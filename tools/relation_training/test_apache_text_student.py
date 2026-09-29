from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from apache_text_student import (
    CLIP_VOCAB_SIZE,
    PredicateTextStudent,
    PredicateTextStudentConfig,
    config_from_checkpoint,
    load_apache_checkpoint,
)
from export_predicate_text_student import (
    check_text_student_parity,
    export_text_student,
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


class PredicateTextStudentTest(unittest.TestCase):
    def test_reference_defaults(self):
        config = PredicateTextStudentConfig()
        self.assertEqual(config.vocab_size, CLIP_VOCAB_SIZE)
        self.assertEqual(config.token_dim, 128)
        self.assertEqual(config.model_dim, 256)
        self.assertEqual(config.depth, 6)
        self.assertEqual(config.heads, 4)
        self.assertEqual(config.ffn_dim, 1024)
        self.assertEqual(config.output_dim, 512)
        self.assertEqual(config.max_length, 32)

    def test_forward_is_normalized_and_padding_invariant(self):
        torch.manual_seed(71)
        model = PredicateTextStudent(tiny_config()).eval()
        ids = torch.tensor(
            [
                [1, 2, 3, 0, 0, 0, 0, 0],
                [1, 2, 3, 17, 18, 19, 20, 21],
            ],
            dtype=torch.int64,
        )
        mask = torch.tensor(
            [
                [False, False, False, True, True, True, True, True],
                [False, False, False, True, True, True, True, True],
            ]
        )
        with torch.inference_mode():
            output = model(ids, mask)
        self.assertEqual(tuple(output.shape), (2, 12))
        self.assertTrue(
            torch.allclose(
                output.norm(dim=-1),
                torch.ones(2),
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output[0],
                output[1],
                atol=1.0e-6,
            )
        )

    def test_checkpoint_uses_exact_apache_state_keys(self):
        torch.manual_seed(72)
        model = PredicateTextStudent(tiny_config())
        state = model.state_dict()
        expected = {
            "token_embedding.weight",
            "tok_proj.weight",
            "positional",
            "blocks.0.norm1.weight",
            "blocks.0.attn.in_proj_weight",
            "blocks.0.norm2.weight",
            "blocks.0.ffn.0.weight",
            "ln_final.weight",
            "head.weight",
        }
        self.assertTrue(expected.issubset(state))

        payload = {
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
            "state_dict": state,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text_student.pt"
            torch.save(payload, path)
            loaded = load_apache_checkpoint(str(path))
        for name, value in state.items():
            self.assertTrue(
                torch.equal(
                    value,
                    loaded.state_dict()[name],
                ),
                msg=f"checkpoint tensor drift: {name}",
            )

    def test_checkpoint_config_rejects_missing_reference_fields(self):
        with self.assertRaises(ValueError):
            config_from_checkpoint(
                {
                    "cfg": {
                        "vocab_size": 64,
                        "dim": 16,
                    }
                }
            )

    def test_forward_contract_rejects_invalid_inputs(self):
        model = PredicateTextStudent(tiny_config())
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 9, dtype=torch.int64))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 8, dtype=torch.int32))
        with self.assertRaises(ValueError):
            model(
                torch.zeros(1, 8, dtype=torch.int64),
                torch.zeros(1, 7, dtype=torch.bool),
            )

    def test_onnx_parity_and_dynamic_predicate_batch(self):
        torch.manual_seed(73)
        model = PredicateTextStudent(tiny_config()).eval()
        ids = torch.tensor(
            [
                [1, 2, 3, 0, 0, 0, 0, 0],
                [4, 5, 6, 7, 0, 0, 0, 0],
            ],
            dtype=torch.int64,
        )
        mask = ids == 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.onnx"
            reference = export_text_student(
                model,
                path,
                ids,
                mask,
            )
            delta = check_text_student_parity(
                path,
                ids,
                mask,
                reference,
            )
            self.assertLessEqual(delta, 1.0e-4)

            import onnxruntime as ort
            session = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
            )
            single_ids = ids[:1].numpy()
            single_mask = mask[:1].numpy()
            result = session.run(
                ["predicate_embedding"],
                {
                    "input_ids": single_ids,
                    "padding_mask": single_mask,
                },
            )[0]
            self.assertEqual(result.shape, (1, 12))


if __name__ == "__main__":
    unittest.main()
