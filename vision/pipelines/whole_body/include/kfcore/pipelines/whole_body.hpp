#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/pose/rtmw.hpp"
#include "kfcore/yolo/detector.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace kfcore::pipelines
{

struct WholeBodyPerson
{
    yolo::Detection detection;
    pose::WholeBodyPose pose;
};

struct WholeBodyFrame
{
    std::int32_t image_width = 0;
    std::int32_t image_height = 0;
    std::vector<WholeBodyPerson> people;
};

struct WholeBodyPipelineOptions
{
    std::int32_t person_class_id = 0;
    float minimum_person_score = 0.0F;
};

class WholeBodyPipeline final
{
public:
    ~WholeBodyPipeline();

    WholeBodyPipeline(const WholeBodyPipeline&) = delete;
    WholeBodyPipeline& operator=(const WholeBodyPipeline&) = delete;

    [[nodiscard]] static std::unique_ptr<WholeBodyPipeline>
    create(std::unique_ptr<yolo::YoloDetector> detector,
           std::unique_ptr<pose::Rtmw> pose_model,
           const WholeBodyPipelineOptions& options = {});

    [[nodiscard]] WholeBodyFrame process(const image::ImageView& image);

private:
    struct Impl;
    explicit WholeBodyPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::pipelines
