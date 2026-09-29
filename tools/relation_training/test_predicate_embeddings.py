from __future__ import annotations

import unittest

import torch

from benchmark import RelationVocabulary
from make_predicate_embeddings import (
    aggregate_prompt_embeddings,
    build_prompts,
    identity_embeddings,
    predicate_text,
)


class PredicateEmbeddingTest(unittest.TestCase):
    def test_predicate_text_replaces_underscores(self):
        self.assertEqual(
            predicate_text("holding_hands"),
            "holding hands",
        )

    def test_prompt_ensemble_is_predicate_major_and_deterministic(self):
        prompts = build_prompts(
            ("holding_hands", "inside_of"),
            (
                "{predicate}",
                "one object is {predicate} another object",
            ),
        )
        self.assertEqual(
            prompts,
            [
                "holding hands",
                "one object is holding hands another object",
                "inside of",
                "one object is inside of another object",
            ],
        )

    def test_prompt_aggregation_normalizes_each_predicate(self):
        prompt_embeddings = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        result = aggregate_prompt_embeddings(
            prompt_embeddings,
            predicate_count=2,
            template_count=2,
        )
        self.assertEqual(tuple(result.shape), (2, 3))
        norms = torch.linalg.vector_norm(result, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms)))
        self.assertGreater(float(result[0, 0]), float(result[0, 2]))
        self.assertGreater(float(result[1, 1]), float(result[1, 0]))

    def test_identity_mode_preserves_closed_vocabulary_contract(self):
        vocabulary = RelationVocabulary(
            predicates=("at", "holds", "wears")
        )
        result = identity_embeddings(vocabulary)
        self.assertTrue(torch.equal(result, torch.eye(3)))

    def test_prompt_template_without_placeholder_fails(self):
        with self.assertRaises(ValueError):
            build_prompts(("at",), ("a relation",))


if __name__ == "__main__":
    unittest.main()
