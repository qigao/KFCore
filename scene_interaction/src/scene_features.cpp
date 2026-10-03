#include "scene_features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfcore::scene_interaction::detail
{
namespace
{

using TrackedDetection = yolo::TrackedDetection;

void validate_box(const yolo::BoxF& box, std::int32_t width, std::int32_t height)
{
    if (!std::isfinite(box.left) || !std::isfinite(box.top) ||
        !std::isfinite(box.right) || !std::isfinite(box.bottom) ||
        box.left < 0.0F || box.top < 0.0F ||
        box.right > static_cast<float>(width) ||
        box.bottom > static_cast<float>(height) ||
        box.right <= box.left || box.bottom <= box.top)
    {
        throw std::invalid_argument(
            "scene behavior object boxes must be finite positive boxes inside the image");
    }
}

float normalized_area(const yolo::BoxF& box, float image_area) noexcept
{
    return ((box.right - box.left) * (box.bottom - box.top)) / image_area;
}

float box_iou(const yolo::BoxF& left, const yolo::BoxF& right) noexcept
{
    const float intersection_left = (std::max)(left.left, right.left);
    const float intersection_top = (std::max)(left.top, right.top);
    const float intersection_right = (std::min)(left.right, right.right);
    const float intersection_bottom = (std::min)(left.bottom, right.bottom);
    const float intersection_width =
        (std::max)(0.0F, intersection_right - intersection_left);
    const float intersection_height =
        (std::max)(0.0F, intersection_bottom - intersection_top);
    const float intersection = intersection_width * intersection_height;
    const float left_area =
        (left.right - left.left) * (left.bottom - left.top);
    const float right_area =
        (right.right - right.left) * (right.bottom - right.top);
    const float denominator = left_area + right_area - intersection;
    return denominator > 0.0F ? intersection / denominator : 0.0F;
}

void write_geometry(std::vector<float>& values, std::size_t predicate_count,
                    const TrackedDetection& subject,
                    const TrackedDetection& object,
                    std::int32_t width, std::int32_t height)
{
    const yolo::BoxF& subject_box = subject.detection.box;
    const yolo::BoxF& object_box = object.detection.box;
    validate_box(subject_box, width, height);
    validate_box(object_box, width, height);

    const float image_width = static_cast<float>(width);
    const float image_height = static_cast<float>(height);
    const float subject_center_x =
        (subject_box.left + subject_box.right) * 0.5F;
    const float subject_center_y =
        (subject_box.top + subject_box.bottom) * 0.5F;
    const float object_center_x =
        (object_box.left + object_box.right) * 0.5F;
    const float object_center_y =
        (object_box.top + object_box.bottom) * 0.5F;

    const float dx = (object_center_x - subject_center_x) / image_width;
    const float dy = (object_center_y - subject_center_y) / image_height;
    const float distance =
        (std::min)(1.0F, std::sqrt(dx * dx + dy * dy) /
                             std::sqrt(2.0F));
    const float image_area = image_width * image_height;

    values[predicate_count + 0U] = dx;
    values[predicate_count + 1U] = dy;
    values[predicate_count + 2U] = distance;
    values[predicate_count + 3U] =
        normalized_area(subject_box, image_area);
    values[predicate_count + 4U] =
        normalized_area(object_box, image_area);
    values[predicate_count + 5U] = box_iou(subject_box, object_box);
}

struct AccumulatedObservation
{
    PairObservation observation;
    std::size_t subject_index = 0U;
    std::size_t object_index = 0U;
};

} // namespace

std::size_t scene_input_size(std::size_t predicate_count)
{
    if (predicate_count == 0U)
    {
        throw std::invalid_argument(
            "scene behavior predicate_count must be positive");
    }
    if (predicate_count >
        (std::numeric_limits<std::size_t>::max)() -
            kSceneGeometryFeatureCount)
    {
        throw std::length_error("scene behavior input size overflow");
    }
    return predicate_count + kSceneGeometryFeatureCount;
}

std::vector<PairObservation>
encode_pair_observations(const pipelines::SceneGraphFrame& frame,
                         std::size_t predicate_count)
{
    const std::size_t input_size = scene_input_size(predicate_count);

    if (frame.objects.image_width <= 0 || frame.objects.image_height <= 0 ||
        frame.relations.image_width != frame.objects.image_width ||
        frame.relations.image_height != frame.objects.image_height)
    {
        throw std::invalid_argument(
            "scene graph object and relation frames must share positive image geometry");
    }

    std::map<PairKey, AccumulatedObservation> accumulated;
    for (const relation::RelationEdge& edge : frame.relations.edges)
    {
        if (edge.subject_index >= frame.objects.detections.size() ||
            edge.object_index >= frame.objects.detections.size() ||
            edge.subject_index == edge.object_index)
        {
            throw std::invalid_argument(
                "scene relation edge references invalid object indices");
        }
        if (edge.predicate_index >= predicate_count ||
            !std::isfinite(edge.score) || edge.score < 0.0F ||
            edge.score > 1.0F)
        {
            throw std::invalid_argument(
                "scene relation edge has invalid predicate or score");
        }

        const TrackedDetection& subject =
            frame.objects.detections[edge.subject_index];
        const TrackedDetection& object =
            frame.objects.detections[edge.object_index];

        if (edge.subject_track_id != subject.track_id ||
            edge.object_track_id != object.track_id)
        {
            throw std::invalid_argument(
                "scene relation edge track identity does not match object table");
        }
        if (!subject.track_id || !object.track_id)
        {
            continue;
        }
        if (subject.detection.class_id < 0 ||
            object.detection.class_id < 0)
        {
            throw std::invalid_argument(
                "scene behavior object class IDs must not be negative");
        }

        if (frame.objects.tracking_epoch == 0U)
        {
            throw std::invalid_argument(
                "tracked SceneGraphFrame must carry a positive tracking epoch");
        }
        const PairKey key{
            *subject.track_id,
            *object.track_id,
            frame.objects.tracking_epoch,
        };
        if (key.subject_track_id == key.object_track_id)
        {
            throw std::invalid_argument(
                "scene relation endpoints must not share one stable track ID");
        }
        auto [it, inserted] = accumulated.try_emplace(key);
        AccumulatedObservation& entry = it->second;
        if (inserted)
        {
            entry.observation.pair = key;
            entry.observation.subject_class_id =
                subject.detection.class_id;
            entry.observation.object_class_id =
                object.detection.class_id;
            entry.observation.values.assign(input_size, 0.0F);
            entry.subject_index = edge.subject_index;
            entry.object_index = edge.object_index;
            write_geometry(entry.observation.values, predicate_count,
                           subject, object,
                           frame.objects.image_width,
                           frame.objects.image_height);
        }
        else if (entry.subject_index != edge.subject_index ||
                 entry.object_index != edge.object_index ||
                 entry.observation.subject_class_id !=
                     subject.detection.class_id ||
                 entry.observation.object_class_id !=
                     object.detection.class_id)
        {
            throw std::invalid_argument(
                "one tracked pair maps to multiple object identities in one frame");
        }

        float& predicate_value =
            entry.observation.values[edge.predicate_index];
        predicate_value = (std::max)(predicate_value, edge.score);
    }

    std::vector<PairObservation> result;
    result.reserve(accumulated.size());
    for (auto& [key, entry] : accumulated)
    {
        (void)key;
        result.push_back(std::move(entry.observation));
    }
    return result;
}

} // namespace kfcore::scene_interaction::detail
