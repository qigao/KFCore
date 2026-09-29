from __future__ import annotations

import unittest

from tensorrt_dynamic_vocab_profile import (
    dynamic_vocabulary_profiles,
    trtexec_shape_flags,
)


class DynamicVocabularyTensorRtProfileTest(unittest.TestCase):
    def metadata(self) -> dict[str, object]:
        return {
            "model_type": "relation.open-vocabulary",
            "vocabulary_dynamic": True,
            "vocabulary_graph_input": True,
            "image_size": 448,
            "max_boxes": 32,
            "query_dim": 512,
            "default_predicate_count": 243,
        }

    def test_reference_profile_keeps_only_vocabulary_axis_dynamic(self):
        profiles = dynamic_vocabulary_profiles(
            self.metadata(),
            max_vocab=19103,
        )
        self.assertEqual(
            profiles["image"]["min"],
            [1, 3, 448, 448],
        )
        self.assertEqual(
            profiles["image"]["min"],
            profiles["image"]["max"],
        )
        self.assertEqual(
            profiles["boxes"]["min"],
            profiles["boxes"]["max"],
        )
        self.assertEqual(
            profiles["W"]["min"],
            [1, 512],
        )
        self.assertEqual(
            profiles["W"]["opt"],
            [243, 512],
        )
        self.assertEqual(
            profiles["W"]["max"],
            [19103, 512],
        )
        self.assertEqual(
            profiles["alpha"]["min"],
            [1],
        )
        self.assertEqual(
            profiles["alpha"]["opt"],
            [243],
        )
        self.assertEqual(
            profiles["alpha"]["max"],
            [19103],
        )

    def test_profile_allows_explicit_optimum(self):
        profiles = dynamic_vocabulary_profiles(
            self.metadata(),
            opt_vocab=512,
            max_vocab=2048,
        )
        self.assertEqual(
            profiles["W"]["opt"],
            [512, 512],
        )
        self.assertEqual(
            profiles["W"]["max"],
            [2048, 512],
        )

    def test_profile_rejects_wrong_model_or_invalid_range(self):
        invalid = self.metadata()
        invalid["model_type"] = "relation.open-vocabulary-encoder"
        with self.assertRaises(ValueError):
            dynamic_vocabulary_profiles(invalid)
        with self.assertRaises(ValueError):
            dynamic_vocabulary_profiles(
                self.metadata(),
                opt_vocab=1024,
                max_vocab=512,
            )

    def test_trtexec_flags_cover_all_five_inputs(self):
        profiles = dynamic_vocabulary_profiles(
            self.metadata(),
            max_vocab=1024,
        )
        flags = trtexec_shape_flags(profiles)
        for selector in ("minShapes", "optShapes", "maxShapes"):
            self.assertIn("image:", flags[selector])
            self.assertIn("boxes:", flags[selector])
            self.assertIn("box_counts:", flags[selector])
            self.assertIn("W:", flags[selector])
            self.assertIn("alpha:", flags[selector])


if __name__ == "__main__":
    unittest.main()
