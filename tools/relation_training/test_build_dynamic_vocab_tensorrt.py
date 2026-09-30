from __future__ import annotations

import unittest

from build_dynamic_vocab_tensorrt import (
    validate_profile_against_graph,
)


class DynamicVocabularyTensorRtBuildContractTest(unittest.TestCase):
    def test_dynamic_vocab_axis_accepts_min_opt_max(self):
        validate_profile_against_graph(
            (-1, 512),
            (
                (1, 512),
                (243, 512),
                (19103, 512),
            ),
            name="W",
        )
        validate_profile_against_graph(
            (-1,),
            (
                (1,),
                (243,),
                (19103,),
            ),
            name="alpha",
        )

    def test_fixed_graph_axes_reject_profile_drift(self):
        with self.assertRaises(ValueError):
            validate_profile_against_graph(
                (1, 3, 448, 448),
                (
                    (1, 3, 448, 448),
                    (1, 3, 448, 448),
                    (1, 3, 512, 512),
                ),
                name="image",
            )

    def test_profile_rank_must_match_graph_rank(self):
        with self.assertRaises(ValueError):
            validate_profile_against_graph(
                (-1, 512),
                (
                    (1,),
                    (1,),
                    (1,),
                ),
                name="W",
            )


if __name__ == "__main__":
    unittest.main()
