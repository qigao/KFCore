from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from apache_text_bank import (
    NPZ_FORMAT,
    PREDICATE_BANK_SCHEMA,
    RELEASED_OBJECT_TEMPLATES,
    RELEASED_PREDICATE_TEMPLATES,
    TEXT_BANK_DERIVATION_SCHEMA,
    encode_labels_with_templates,
    load_predicate_text_bank,
    ordered_strings_sha256,
    rebuild_text_banks,
    sha256_file,
    tokenize_texts,
)
from apache_text_student import (
    PredicateTextStudent,
    PredicateTextStudentConfig,
)


class FakeTokenizer:
    def __call__(
        self,
        texts,
        *,
        truncation,
        max_length,
    ):
        assert truncation is True
        rows = []
        for text in texts:
            # Stable, content-sensitive token rows without external tokenizer deps.
            raw = [
                1,
                *[
                    2 + (byte % 29)
                    for byte in text.encode("utf-8")
                ],
                31,
            ]
            rows.append(
                raw[:max_length]
            )
        return {
            "input_ids": rows,
        }


class FakeStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            max_length=8,
            output_dim=4,
        )

    def forward(
        self,
        input_ids,
        padding_mask=None,
    ):
        if padding_mask is None:
            padding_mask = (
                input_ids == 0
            )
        keep = (
            ~padding_mask
        ).to(torch.float32)
        values = input_ids.to(
            torch.float32
        )
        features = torch.stack(
            (
                (values * keep).sum(-1),
                ((values % 5) * keep).sum(-1),
                ((values % 7) * keep).sum(-1),
                keep.sum(-1),
            ),
            dim=-1,
        )
        return F.normalize(
            features,
            dim=-1,
        )


def tiny_config() -> PredicateTextStudentConfig:
    return PredicateTextStudentConfig(
        vocab_size=64,
        token_dim=8,
        model_dim=16,
        depth=2,
        heads=4,
        ffn_dim=32,
        output_dim=12,
        max_length=8,
    )


def write_tiny_checkpoint(
    root: Path,
) -> Path:
    torch.manual_seed(198)
    model = PredicateTextStudent(
        tiny_config()
    )
    path = root / "student.pt"
    torch.save(
        {
            "cfg": {
                "vocab_size": 64,
                "d_tok": 8,
                "dim": 16,
                "depth": 2,
                "heads": 4,
                "ffn_dim": 32,
                "out_dim": 12,
                "max_len": 8,
            },
            "state_dict": (
                model.state_dict()
            ),
        },
        path,
    )
    return path


def write_tokenizer_bundle(
    root: Path,
) -> Path:
    directory = (
        root / "tokenizer"
    )
    directory.mkdir()
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {
                "fixture": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (directory / "tokenizer_config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    return directory


def sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


class ApacheTextBankTest(
    unittest.TestCase
):
    def test_template_encoding_is_sum_then_single_normalize(self):
        model = FakeStudent()
        tokenizer = FakeTokenizer()
        labels = (
            "on",
            "holding",
        )
        templates = (
            "{p}",
            "photo: {p}",
        )

        actual = (
            encode_labels_with_templates(
                model,
                tokenizer,
                labels,
                templates,
                batch_size=1,
            )
        )

        expected_sum = None
        for template in templates:
            texts = [
                template.format(
                    p=label
                )
                for label in labels
            ]
            ids, padding = (
                tokenize_texts(
                    tokenizer,
                    texts,
                    max_length=8,
                )
            )
            encoded = model(
                ids,
                padding,
            )
            expected_sum = (
                encoded
                if expected_sum is None
                else expected_sum
                + encoded
            )
        assert expected_sum is not None
        expected = F.normalize(
            expected_sum,
            dim=-1,
        )

        self.assertTrue(
            torch.equal(
                actual,
                expected,
            )
        )
        self.assertTrue(
            torch.allclose(
                actual.norm(
                    dim=-1
                ),
                torch.ones(2),
                rtol=0.0,
                atol=1.0e-6,
            )
        )

    def test_named_predicate_bank_requires_order_and_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "pred.npz"
            embeddings = np.ones(
                (2, 512),
                dtype=np.float16,
            )
            np.savez_compressed(
                path,
                embeddings=embeddings,
                predicates=np.asarray(
                    [
                        "holding",
                        "on",
                    ]
                ),
                templates=np.asarray(
                    RELEASED_PREDICATE_TEMPLATES
                ),
            )
            tensor, report = (
                load_predicate_text_bank(
                    path,
                    (
                        "holding",
                        "on",
                    ),
                )
            )
            self.assertEqual(
                tuple(tensor.shape),
                (2, 512),
            )
            self.assertEqual(
                report["schema"],
                PREDICATE_BANK_SCHEMA,
            )
            self.assertEqual(
                report[
                    "predicate_order_sha256"
                ],
                ordered_strings_sha256(
                    (
                        "holding",
                        "on",
                    )
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "names/order",
            ):
                load_predicate_text_bank(
                    path,
                    (
                        "on",
                        "holding",
                    ),
                )

            wrong_templates = (
                root / "wrong.npz"
            )
            np.savez_compressed(
                wrong_templates,
                embeddings=embeddings,
                predicates=np.asarray(
                    [
                        "holding",
                        "on",
                    ]
                ),
                templates=np.asarray(
                    [
                        "{p}",
                        "a photo of {p}",
                    ]
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "3-template ensemble",
            ):
                load_predicate_text_bank(
                    wrong_templates,
                    (
                        "holding",
                        "on",
                    ),
                )

    def test_rebuild_is_byte_deterministic_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student = (
                write_tiny_checkpoint(
                    root
                )
            )
            tokenizer_dir = (
                write_tokenizer_bundle(
                    root
                )
            )
            student_sha = (
                sha256_file(
                    student
                )
            )

            one_pred = (
                root / "one_pred.npz"
            )
            one_obj = (
                root / "one_obj.npz"
            )
            one_evidence = (
                root / "one.json"
            )
            two_pred = (
                root / "two_pred.npz"
            )
            two_obj = (
                root / "two_obj.npz"
            )
            two_evidence = (
                root / "two.json"
            )

            report_one = (
                rebuild_text_banks(
                    student_checkpoint=student,
                    tokenizer_dir=(
                        tokenizer_dir
                    ),
                    predicates=(
                        "holding",
                        "on",
                    ),
                    object_labels=(
                        "person",
                        "horse",
                    ),
                    predicate_output=(
                        one_pred
                    ),
                    object_output=one_obj,
                    evidence_output=(
                        one_evidence
                    ),
                    expected_student_sha256=(
                        student_sha
                    ),
                    require_released_config=False,
                    tokenizer_factory=(
                        lambda _: FakeTokenizer()
                    ),
                    transformers_version="test",
                    batch_size=1,
                )
            )
            report_two = (
                rebuild_text_banks(
                    student_checkpoint=student,
                    tokenizer_dir=(
                        tokenizer_dir
                    ),
                    predicates=(
                        "holding",
                        "on",
                    ),
                    object_labels=(
                        "person",
                        "horse",
                    ),
                    predicate_output=(
                        two_pred
                    ),
                    object_output=two_obj,
                    evidence_output=(
                        two_evidence
                    ),
                    expected_student_sha256=(
                        student_sha
                    ),
                    require_released_config=False,
                    tokenizer_factory=(
                        lambda _: FakeTokenizer()
                    ),
                    transformers_version="test",
                    batch_size=1,
                )
            )

            one_pred_bytes = (
                one_pred.read_bytes()
            )
            two_pred_bytes = (
                two_pred.read_bytes()
            )
            one_obj_bytes = (
                one_obj.read_bytes()
            )
            two_obj_bytes = (
                two_obj.read_bytes()
            )
            one_evidence_bytes = (
                one_evidence.read_bytes()
            )
            two_evidence_bytes = (
                two_evidence.read_bytes()
            )

        self.assertEqual(
            one_pred_bytes,
            two_pred_bytes,
        )
        self.assertEqual(
            one_obj_bytes,
            two_obj_bytes,
        )
        self.assertEqual(
            one_evidence_bytes,
            two_evidence_bytes,
        )
        self.assertEqual(
            report_one,
            report_two,
        )
        self.assertEqual(
            report_one["schema"],
            TEXT_BANK_DERIVATION_SCHEMA,
        )
        self.assertEqual(
            report_one[
                "student_checkpoint_sha256"
            ],
            student_sha,
        )
        self.assertEqual(
            report_one[
                "encoding_semantics"
            ],
            "encode-each-template,sum-template-vectors,l2-normalize-once",
        )
        self.assertEqual(
            report_one[
                "predicate_bank"
            ]["templates"],
            list(
                RELEASED_PREDICATE_TEMPLATES
            ),
        )
        self.assertEqual(
            report_one[
                "object_bank"
            ]["templates"],
            list(
                RELEASED_OBJECT_TEMPLATES
            ),
        )
        self.assertEqual(
            report_one[
                "predicate_bank"
            ]["npz_format"],
            NPZ_FORMAT,
        )
        self.assertEqual(
            report_one[
                "object_bank"
            ]["npz_format"],
            NPZ_FORMAT,
        )
        self.assertEqual(
            report_one[
                "predicate_bank"
            ]["artifact_sha256"],
            sha256_bytes(
                one_pred_bytes
            ),
        )
        self.assertEqual(
            report_one[
                "object_bank"
            ]["artifact_sha256"],
            sha256_bytes(
                one_obj_bytes
            ),
        )
        self.assertEqual(
            report_one[
                "predicate_bank"
            ]["shape"],
            [2, 12],
        )
        self.assertEqual(
            report_one[
                "object_bank"
            ]["shape"],
            [2, 12],
        )

    def test_student_sha_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student = (
                write_tiny_checkpoint(
                    root
                )
            )
            tokenizer_dir = (
                write_tokenizer_bundle(
                    root
                )
            )
            with self.assertRaisesRegex(
                ValueError,
                "checkpoint SHA-256",
            ):
                rebuild_text_banks(
                    student_checkpoint=student,
                    tokenizer_dir=tokenizer_dir,
                    predicates=(
                        "holding",
                        "on",
                    ),
                    object_labels=(
                        "person",
                        "horse",
                    ),
                    predicate_output=(
                        root / "pred.npz"
                    ),
                    object_output=(
                        root / "obj.npz"
                    ),
                    evidence_output=(
                        root / "evidence.json"
                    ),
                    expected_student_sha256=(
                        "0" * 64
                    ),
                    require_released_config=False,
                    tokenizer_factory=(
                        lambda _: FakeTokenizer()
                    ),
                    transformers_version="test",
                )

    def test_tokenizer_bundle_requires_portable_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student = (
                write_tiny_checkpoint(
                    root
                )
            )
            empty = (
                root / "tokenizer"
            )
            empty.mkdir()

            with self.assertRaisesRegex(
                ValueError,
                "tokenizer.json",
            ):
                rebuild_text_banks(
                    student_checkpoint=student,
                    tokenizer_dir=empty,
                    predicates=(
                        "holding",
                        "on",
                    ),
                    object_labels=(
                        "person",
                        "horse",
                    ),
                    predicate_output=(
                        root / "pred.npz"
                    ),
                    object_output=(
                        root / "obj.npz"
                    ),
                    evidence_output=(
                        root / "evidence.json"
                    ),
                    expected_student_sha256=(
                        sha256_file(
                            student
                        )
                    ),
                    require_released_config=False,
                    tokenizer_factory=(
                        lambda _: FakeTokenizer()
                    ),
                    transformers_version="test",
                )


if __name__ == "__main__":
    unittest.main()
