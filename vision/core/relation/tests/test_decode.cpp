#include "decode.hpp"

#include "kfcore/relation/error.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace kfcore::relation;

namespace
{

RelateAnythingOptions options()
{
    RelateAnythingOptions value;
    value.predicates = {"beside", "holding"};
    value.threshold = 0.5F;
    value.top_k = 8U;
    return value;
}

std::vector<Region> regions()
{
    return {
        {0.0F, 0.0F, 10.0F, 10.0F, 0.9F, std::uint64_t{11}},
        {20.0F, 0.0F, 30.0F, 10.0F, 0.8F, std::uint64_t{22}},
        {40.0F, 0.0F, 50.0F, 10.0F, 0.2F, std::uint64_t{33}},
    };
}

} // namespace

spec("scene relation decoding")
{
    it("keeps legacy default and exposes explicit Apache 40-box options")
    {
        const RelateAnythingOptions legacy;
        const auto apache = apache_released_relate_anything_options();

        check(legacy.max_boxes == kLegacyRelationMaxBoxes);
        check(apache.max_boxes == kApacheReleasedMaxBoxes);
        check(kLegacyRelationMaxBoxes == std::size_t{32U});
        check(kApacheReleasedMaxBoxes == std::size_t{40U});
    }

    it("fuses predicate and pair logits and keeps the best predicate per pair")
    {
        const float pred[] = {
            -1.0F, 2.0F,
            3.0F, 1.0F,
        };
        const float pair[] = {1.0F, -4.0F};
        const std::int64_t sub[] = {0, 1};
        const std::int64_t obj[] = {1, 2};
        const std::uint8_t valid[] = {1, 1};

        auto cfg = options();
        cfg.threshold = 0.7F;
        const detail::RawRelationOutputs raw {
            pred, pair, sub, obj, valid, 2U, 2U
        };

        const auto edges =
            detail::decode_relation_outputs(raw, regions(), cfg);

        check(edges.size() == std::size_t{1U});
        check(edges[0].subject_index == std::size_t{0U});
        check(edges[0].object_index == std::size_t{1U});
        check(edges[0].predicate_index == std::size_t{1U});
        check(edges[0].score > 0.95F);
        check(edges[0].subject_track_id == std::uint64_t{11});
        check(edges[0].object_track_id == std::uint64_t{22});
    }

    it("drops invalid self and out-of-range pairs")
    {
        const float pred[] = {
            5.0F, 1.0F,
            5.0F, 1.0F,
            5.0F, 1.0F,
            5.0F, 1.0F,
        };
        const float pair[] = {1.0F, 1.0F, 1.0F, 1.0F};
        const std::int64_t sub[] = {0, 1, 0, 2};
        const std::int64_t obj[] = {1, 1, 7, 0};
        const std::uint8_t valid[] = {0, 1, 1, 1};

        const detail::RawRelationOutputs raw {
            pred, pair, sub, obj, valid, 4U, 2U
        };
        const auto edges =
            detail::decode_relation_outputs(raw, regions(), options());

        check(edges.size() == std::size_t{1U});
        check(edges[0].subject_index == std::size_t{2U});
        check(edges[0].object_index == std::size_t{0U});
    }

    it("uses detector confidence for ranking without changing relation score")
    {
        const float pred[] = {
            2.0F, 0.0F,
            2.0F, 0.0F,
        };
        const float pair[] = {0.0F, 0.0F};
        const std::int64_t sub[] = {0, 0};
        const std::int64_t obj[] = {2, 1};
        const std::uint8_t valid[] = {1, 1};

        auto cfg = options();
        cfg.top_k = 2U;
        const detail::RawRelationOutputs raw {
            pred, pair, sub, obj, valid, 2U, 2U
        };
        const auto edges =
            detail::decode_relation_outputs(raw, regions(), cfg);

        check(edges.size() == std::size_t{2U});
        check(edges[0].object_index == std::size_t{1U});
        check(edges[1].object_index == std::size_t{2U});
        check(std::fabs(edges[0].score - edges[1].score) < 1.0e-6F);
    }

    it("applies calibration in logit space")
    {
        const float pred[] = {0.0F, -1.0F};
        const float pair[] = {0.0F};
        const std::int64_t sub[] = {0};
        const std::int64_t obj[] = {1};
        const std::uint8_t valid[] = {1};

        auto cfg = options();
        cfg.threshold = 0.7F;
        cfg.calibration_a = 2.0F;
        cfg.calibration_b = 1.0F;
        const detail::RawRelationOutputs raw {
            pred, pair, sub, obj, valid, 1U, 2U
        };
        const auto edges =
            detail::decode_relation_outputs(raw, regions(), cfg);

        check(edges.size() == std::size_t{1U});
        check(edges[0].score > 0.73F);
        check(edges[0].score < 0.74F);
    }

    it("rejects a vocabulary width mismatch")
    {
        const float pred[] = {0.0F};
        const float pair[] = {0.0F};
        const std::int64_t sub[] = {0};
        const std::int64_t obj[] = {1};
        const std::uint8_t valid[] = {1};
        const detail::RawRelationOutputs raw {
            pred, pair, sub, obj, valid, 1U, 1U
        };

        check_throws_as(
            detail::decode_relation_outputs(raw, regions(), options()),
            RelationError);
    }
}
