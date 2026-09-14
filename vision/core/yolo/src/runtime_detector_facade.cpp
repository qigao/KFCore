#include "kfcore/yolo/detector.hpp"

namespace kfcore::yolo
{

std::unique_ptr<YoloDetector> YoloDetector::load(
    runtime::Runtime& runtime,
    const runtime::ModelPackage& package,
    const runtime::ExecutionPolicy& policy,
    const YoloDetectorOptions& options)
{
    return load(package, runtime.backends(), policy, options);
}

} // namespace kfcore::yolo
