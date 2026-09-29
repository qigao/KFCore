from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from benchmark import RelationVocabulary
from semantic_ontology import (
    build_metadata,
    semantic_soft_positive_weights,
)
from make_predicate_embeddings import (
    DEFAULT_TEMPLATES,
    aggregate_prompt_embeddings,
    gram_diagnostics,
    identity_predicate_embeddings,
    predicate_prompts,
    symmetric_whiten_predicates,
    tensor_sha256,
    write_outputs,
)


class PredicatePrototypeTest(unittest.TestCase):
    def test_dimension_matched_identity_is_orthonormal(self):
        embeddings = identity_predicate_embeddings(
            3,
            dimension=8,
        )
        self.assertEqual(tuple(embeddings.shape), (3, 8))
        self.assertTrue(
            torch.allclose(
                embeddings @ embeddings.T,
                torch.eye(3),
            )
        )
        self.assertTrue(
            torch.allclose(
                embeddings.norm(dim=-1),
                torch.ones(3),
            )
        )

    def test_identity_rejects_dimension_smaller_than_vocabulary(self):
        with self.assertRaises(ValueError):
            identity_predicate_embeddings(
                4,
                dimension=3,
            )

    def test_prompt_generation_normalizes_underscores(self):
        prompts = predicate_prompts(
            ("in_front_of", "holding"),
            DEFAULT_TEMPLATES,
        )
        self.assertEqual(
            prompts[0][0],
            "in front of",
        )
        self.assertIn(
            "in front of",
            prompts[0][1],
        )
        self.assertEqual(len(prompts[1]), len(DEFAULT_TEMPLATES))

    def test_prompt_aggregation_normalizes_each_prompt_then_mean(self):
        encoded = torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 5.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )
        result = aggregate_prompt_embeddings(
            encoded,
            predicate_count=2,
            template_count=3,
        )
        self.assertEqual(tuple(result.shape), (2, 3))
        self.assertTrue(
            torch.allclose(
                result.norm(dim=-1),
                torch.ones(2),
                atol=1.0e-6,
            )
        )

        expected_first = torch.tensor([1.0, 1.0, 1.0])
        expected_first = expected_first / expected_first.norm()
        self.assertTrue(
            torch.allclose(
                result[0],
                expected_first,
                atol=1.0e-6,
            )
        )

    def test_symmetric_whitening_removes_row_correlation(self):
        correlated = torch.tensor(
            [
                [1.0, 0.8, 0.0, 0.0],
                [0.9, 1.0, 0.2, 0.0],
                [0.2, 0.4, 1.0, 0.3],
            ],
            dtype=torch.float32,
        )
        before = gram_diagnostics(correlated)
        whitened = symmetric_whiten_predicates(correlated)
        after = gram_diagnostics(whitened)

        self.assertEqual(tuple(whitened.shape), (3, 4))
        self.assertGreater(before["max_abs_off_diagonal"], 0.5)
        self.assertLess(after["max_abs_off_diagonal"], 1.0e-5)
        self.assertLess(after["mean_abs_off_diagonal"], 1.0e-5)
        self.assertTrue(
            torch.allclose(
                whitened @ whitened.T,
                torch.eye(3),
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        )

    def test_symmetric_whitening_rejects_rank_deficient_rows(self):
        duplicate = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        with self.assertRaises(ValueError):
            symmetric_whiten_predicates(duplicate)

    def test_semantic_soft_positive_weights_use_sparse_symmetric_neighbors(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.98, 0.20, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        weights, cosine = semantic_soft_positive_weights(
            embeddings,
            top_k=1,
            temperature=0.1,
            min_weight=0.05,
        )

        self.assertEqual(tuple(weights.shape), (4, 4))
        self.assertTrue(torch.allclose(weights.diag(), torch.ones(4)))
        self.assertTrue(torch.allclose(weights, weights.T))
        self.assertGreater(float(weights[0, 1]), 0.05)
        self.assertEqual(float(weights[0, 2]), 0.0)
        self.assertEqual(float(weights[0, 3]), 0.0)
        self.assertTrue(torch.allclose(cosine, cosine.T))

    def test_semantic_soft_positive_weights_respect_minimum_weight(self):
        embeddings = torch.eye(3)
        weights, _ = semantic_soft_positive_weights(
            embeddings,
            top_k=2,
            temperature=0.1,
            min_weight=0.05,
        )
        self.assertTrue(torch.equal(weights, torch.eye(3)))

    def test_semantic_ontology_metadata_records_named_edges(self):
        vocabulary = RelationVocabulary(
            predicates=("contain", "inside_of", "ride"),
        )
        weights = torch.eye(3)
        weights[0, 1] = 0.4
        weights[1, 0] = 0.4
        cosine = torch.tensor(
            [
                [1.0, 0.8, 0.1],
                [0.8, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        metadata = build_metadata(
            vocabulary,
            weights,
            cosine,
            source_tensor_sha256="source-hash",
            top_k=2,
            temperature=0.1,
            min_weight=0.05,
        )
        self.assertEqual(metadata["edge_count"], 1)
        edge = metadata["edges"][0]
        self.assertEqual(edge["left"], "contain")
        self.assertEqual(edge["right"], "inside_of")
        self.assertAlmostEqual(edge["weight"], 0.4)

    def test_write_outputs_records_stable_tensor_hash(self):
        vocabulary = RelationVocabulary(
            predicates=("above", "behind", "holding"),
        )
        embeddings = identity_predicate_embeddings(
            3,
            dimension=8,
        )
        expected_hash = tensor_sha256(embeddings)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "prototypes.pt"
            write_outputs(
                out,
                embeddings,
                vocabulary=vocabulary,
                mode="identity",
                model_id=None,
                templates=(),
            )

            restored = torch.load(
                out,
                map_location="cpu",
                weights_only=True,
            )
            metadata = json.loads(
                out.with_suffix(".pt.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(torch.equal(restored, embeddings))
        self.assertEqual(metadata["tensor_sha256"], expected_hash)
        self.assertEqual(metadata["shape"], [3, 8])
        self.assertEqual(
            metadata["vocabulary_sha256"],
            vocabulary.sha256(),
        )


if __name__ == "__main__":
    unittest.main()
