#include "kfcore/pipelines/whole_body.hpp"

#include "kfcore/pose/error.hpp"
#include "kfcore/yolo/error.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace kfcore::pipelines
{

struct WholeBodyPipeline::Impl final
{
    std::unique_ptr<yolo::YoloDetector> detector;
    std::unique_ptr<pose::Rtmw> pose_model;
    WholeBodyPipelineOptions options;
};

WholeBodyPipeline::WholeBodyPipeline(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

WholeBodyPipeline::~WholeBodyPipeline() = default;

std::unique_ptr<WholeBodyPipeline> WholeBodyPipeline::create(
    std::unique_ptr<yolo::YoloDetector> detector,
    std::unique_ptr<pose::Rtmw> pose_model,
    const WholeBodyPipelineOptions& options)
{
    if (!detector || !pose_model)
    {
        throw std::invalid_argument(
            "WholeBodyPipeline requires detector and pose model instances");
    }
    if (!std::isfinite(options.minimum_person_score) ||
        options.minimum_person_score < 0.0F ||
        options.minimum_person_score > 1.0F)
    {
        throw std::invalid_argument(
            "WholeBodyPipeline minimum person score must be within [0,1]");
    }

    auto impl = std::make_unique<Impl>();
    impl->detector = std::move(detector);
    impl->pose_model = std::move(pose_model);
    impl->options = options;
    return std::unique_ptr<WholeBodyPipeline>(
        new WholeBodyPipeline(std::move(impl)));
}

WholeBodyFrame WholeBodyPipeline::process(const image::ImageView& image)
{
    if (!impl_ || !impl_->detector || !impl_->pose_model)
    {
        throw std::logic_error("WholeBodyPipeline state is unavailable");
    }

    const yolo::DetectionFrame detections = impl_->detector->detect(image);
    std::vector<pose::RectF> boxes;
    std::vector<yolo::Detection> selected;
    boxes.reserve(detections.detections.size());
    selected.reserve(detections.detections.size());

    for (const yolo::Detection& detection : detections.detections)
    {
        if (detection.class_id != impl_->options.person_class_id ||
            detection.score < impl_->options.minimum_person_score)
        {
            continue;
        }
        const float width = detection.box.right - detection.box.left;
        const float height = detection.box.bottom - detection.box.top;
        if (!std::isfinite(width) || !std::isfinite(height) ||
            width <= 0.0F || height <= 0.0F)
        {
            continue;
        }
        boxes.push_back({detection.box.left, detection.box.top, width, height});
        selected.push_back(detection);
    }

    const std::vector<pose::WholeBodyPose> poses =
        impl_->pose_model->infer(image, boxes);
    if (poses.size() != selected.size())
    {
        throw std::runtime_error(
            "WholeBodyPipeline pose result count does not match selected persons");
    }

    WholeBodyFrame result;
    result.image_width = detections.image_width;
    result.image_height = detections.image_height;
    result.people.reserve(poses.size());
    for (std::size_t index = 0U; index < poses.size(); ++index)
    {
        result.people.push_back({selected[index], poses[index]});
    }
    return result;
}

} // namespace kfcore::pipelines
