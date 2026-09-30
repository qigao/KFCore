from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from apache_text_student import (
    PredicateTextStudent,
    PredicateTextStudentConfig,
)
from make_apache_object_embeddings import (
    OBJECT_BANK_SCHEMA,
    OBJECT_TEMPLATES,
    encode_object_bank,
    ordered_strings_sha256,
    sha256_file,
    validate_object_bank_evidence,
    validate_object_bank_provenance,
)
from make_predicate_embeddings import tensor_sha256


class FakeTokenizer:
    def __call__(
        self,
        texts,
        *,
        truncation,
        max_length,
    ):
        self.last_truncation = truncation
        self.last_max_length = max_length
        rows = []
        for text in texts:
            code = (
                sum(text.encode("utf-8"))
                % 20
            ) + 3
            rows.append([1, code, 2])
        return {"input_ids": rows}


def tiny_student() -> PredicateTextStudent:
    torch.manual_seed(181)
    return PredicateTextStudent(
        PredicateTextStudentConfig(
            vocab_size=32,
            token_dim=4,
            model_dim=8,
            depth=1,
            heads=2,
            ffn_dim=16,
            output_dim=4,
            max_length=6,
        )
    ).eval()


def manual_encode(
    model: PredicateTextStudent,
    tokenizer: FakeTokenizer,
    prompts: list[str],
) -> torch.Tensor:
    rows = tokenizer(
        prompts,
        truncation=True,
        max_length=model.config.max_length,
    )["input_ids"]
    ids = torch.zeros(
        len(rows),
        model.config.max_length,
        dtype=torch.int64,
    )
    mask = torch.ones_like(
        ids,
        dtype=torch.bool,
    )
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(
            row,
            dtype=torch.int64,
        )
        mask[index, : len(row)] = False
    with torch.inference_mode():
        return model(ids, mask).float()


class ApacheObjectEmbeddingsTest(
    unittest.TestCase
):
    def test_template_sum_and_renormalize_matches_upstream(self):
        model = tiny_student()
        tokenizer = FakeTokenizer()
        labels = ("person", "horse")

        actual = encode_object_bank(
            model,
            tokenizer,
            labels,
            device="cpu",
        )

        first = manual_encode(
            model,
            tokenizer,
            [
                OBJECT_TEMPLATES[0].format(
                    p=label
                )
                for label in labels
            ],
        )
        second = manual_encode(
            model,
            tokenizer,
            [
                OBJECT_TEMPLATES[1].format(
                    p=label
                )
                for label in labels
            ],
        )
        expected = F.normalize(
            first + second,
            dim=-1,
        ).float()

        self.assertTrue(
            torch.equal(
                actual,
                expected,
            )
        )
        self.assertEqual(
            tokenizer.last_max_length,
            model.config.max_length,
        )
        self.assertTrue(
            tokenizer.last_truncation
        )
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(
                    actual,
                    dim=-1,
                ),
                torch.ones(len(labels)),
                atol=1.0e-6,
                rtol=0.0,
            )
        )

    def test_nonreleased_template_contract_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "templates",
        ):
            encode_object_bank(
                tiny_student(),
                FakeTokenizer(),
                ("person",),
                templates=(
                    "{p}",
                    "photo: {p}",
                ),
            )

    def test_provenance_binds_tensor_student_tokenizer_and_order(self):
        labels = ("person", "horse")
        bank = F.normalize(
            torch.randn(2, 512),
            dim=-1,
        ).float()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor_path = root / "objects.pt"
            student_path = root / "student.pt"
            tokenizer_contract = (
                root / "contract.json"
            )
            metadata_path = (
                root / "objects.pt.json"
            )
            torch.save(bank, tensor_path)
            student_path.write_bytes(
                b"student-fixture"
            )
            tokenizer_contract.write_text(
                json.dumps(
                    {
                        "schema": (
                            "kfcore.clip-tokenizer-golden/1"
                        )
                    }
                ),
                encoding="utf-8",
            )
            metadata = {
                "schema": OBJECT_BANK_SCHEMA,
                "templates": list(
                    OBJECT_TEMPLATES
                ),
                "object_labels": list(
                    labels
                ),
                "object_labels_sha256": (
                    ordered_strings_sha256(
                        labels
                    )
                ),
                "output_dim": 512,
                "row_normalized": True,
                "tensor_sha256": tensor_sha256(
                    bank
                ),
                "tensor_file_sha256": (
                    sha256_file(
                        tensor_path
                    )
                ),
                "text_student_sha256": (
                    sha256_file(
                        student_path
                    )
                ),
                "tokenizer_contract_sha256": (
                    sha256_file(
                        tokenizer_contract
                    )
                ),
            }
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            report = (
                validate_object_bank_provenance(
                    bank,
                    tensor_path=tensor_path,
                    metadata_path=metadata_path,
                    object_labels=labels,
                    text_student_path=student_path,
                    tokenizer_contract_path=(
                        tokenizer_contract
                    ),
                )
            )
            self.assertEqual(
                report["tensor_sha256"],
                tensor_sha256(bank),
            )

            with self.assertRaisesRegex(
                ValueError,
                "object-label order",
            ):
                validate_object_bank_provenance(
                    bank,
                    tensor_path=tensor_path,
                    metadata_path=metadata_path,
                    object_labels=tuple(
                        reversed(labels)
                    ),
                    text_student_path=student_path,
                    tokenizer_contract_path=(
                        tokenizer_contract
                    ),
                )

            student_path.write_bytes(
                b"different-student"
            )
            with self.assertRaisesRegex(
                ValueError,
                "text_student_sha256",
            ):
                validate_object_bank_provenance(
                    bank,
                    tensor_path=tensor_path,
                    metadata_path=metadata_path,
                    object_labels=labels,
                    text_student_path=student_path,
                    tokenizer_contract_path=(
                        tokenizer_contract
                    ),
                )

    def test_portable_evidence_rejects_template_and_hash_drift(self):
        labels = ["person", "horse"]
        meta = {
            "schema": OBJECT_BANK_SCHEMA,
            "templates": list(
                OBJECT_TEMPLATES
            ),
            "object_labels": labels,
            "object_labels_sha256": (
                ordered_strings_sha256(
                    labels
                )
            ),
            "output_dim": 512,
            "row_normalized": True,
            "tensor_sha256": "1" * 64,
            "tensor_file_sha256": "2" * 64,
            "text_student_sha256": "3" * 64,
            "tokenizer_contract_sha256": (
                "4" * 64
            ),
        }
        validate_object_bank_evidence(
            meta,
            object_labels=labels,
        )

        changed = copy.deepcopy(meta)
        changed["templates"][1] = "photo {p}"
        with self.assertRaisesRegex(
            ValueError,
            "templates",
        ):
            validate_object_bank_evidence(
                changed
            )

        changed = copy.deepcopy(meta)
        changed["text_student_sha256"] = (
            "not-a-hash"
        )
        with self.assertRaisesRegex(
            ValueError,
            "text_student_sha256",
        ):
            validate_object_bank_evidence(
                changed
            )


if __name__ == "__main__":
    unittest.main()
