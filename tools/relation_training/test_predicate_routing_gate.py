from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from apache_vocab_head import ApacheVocabHead
from export_predicate_routing_gate import (
    PredicateRoutingGateExport,
    check_gate_parity,
    export_gate,
)


class PredicateRoutingGateExportTest(unittest.TestCase):
    def test_dynamic_vocabulary_gate_onnx_parity(self):
        torch.manual_seed(91)
        head = ApacheVocabHead(
            d_model=16,
            text_dim=12,
            gate_hidden=8,
        ).eval()
        model = PredicateRoutingGateExport(head).eval()

        W3 = torch.nn.functional.normalize(
            torch.randn(3, 12),
            dim=-1,
        )
        W1 = W3[:1].clone()
        W9 = torch.nn.functional.normalize(
            torch.randn(9, 12),
            dim=-1,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routing-gate.onnx"
            reference = export_gate(
                model,
                path,
                W3,
            )
            self.assertEqual(
                tuple(reference.shape),
                (3,),
            )
            delta = check_gate_parity(
                path,
                model,
                [W1, W3, W9],
            )

        self.assertLessEqual(delta, 1.0e-6)

    def test_routing_gate_is_permutation_equivariant(self):
        torch.manual_seed(92)
        head = ApacheVocabHead(
            d_model=16,
            text_dim=12,
            gate_hidden=8,
        ).eval()
        W = torch.nn.functional.normalize(
            torch.randn(5, 12),
            dim=-1,
        )
        permutation = torch.tensor([3, 1, 4, 0, 2])

        with torch.inference_mode():
            alpha = head.routing_alpha(W)
            permuted = head.routing_alpha(
                W[permutation]
            )

        self.assertTrue(
            torch.allclose(
                permuted,
                alpha[permutation],
                atol=0.0,
                rtol=0.0,
            )
        )

    def test_routing_gate_outputs_are_valid_probabilities(self):
        torch.manual_seed(93)
        head = ApacheVocabHead(
            d_model=16,
            text_dim=12,
            gate_hidden=8,
        ).eval()
        W = torch.nn.functional.normalize(
            torch.randn(17, 12),
            dim=-1,
        )
        with torch.inference_mode():
            alpha = head.routing_alpha(W)

        self.assertTrue(torch.isfinite(alpha).all())
        self.assertTrue((alpha > 0.0).all())
        self.assertTrue((alpha < 1.0).all())


if __name__ == "__main__":
    unittest.main()
