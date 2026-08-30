#include "kfcore/yolo/onnx.hpp"

#include "kfcore/yolo/error.hpp"
#include "tinytest.hpp"

#include <functional>
#include <string>
#include <type_traits>

using namespace kfcore::yolo;

namespace
{

void check_error(const std::function<void()>& operation, YoloErrorCode code,
                 const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const YoloError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}
} // namespace

static_assert(!std::is_copy_constructible_v<OnnxDetector>);
static_assert(!std::is_copy_assignable_v<OnnxDetector>);
static_assert(std::is_move_constructible_v<OnnxDetector>);

spec("YOLOv8 ONNX Runtime detector validation")
{
    it("rejects invalid resource and thread options before reading a model")
    {
        OnnxDetectorOptions options;
        options.intra_op_threads = -1;
        check_error([&] { (void)OnnxDetector::load("missing.onnx", options); },
                    YoloErrorCode::InvalidArgument, "thread");

        options.intra_op_threads = 0;
        options.max_output_bytes = 0U;
        check_error([&] { (void)OnnxDetector::load("missing.onnx", options); },
                    YoloErrorCode::ResourceLimitExceeded, "limit");

        options.max_output_bytes = 1U;
        options.max_detections = 0U;
        check_error([&] { (void)OnnxDetector::load("missing.onnx", options); },
                    YoloErrorCode::ResourceLimitExceeded, "detections");

        options.max_detections = 1U;
        options.border_value = 256.0F;
        check_error([&] { (void)OnnxDetector::load("missing.onnx", options); },
                    YoloErrorCode::InvalidArgument, "border");
    }

    it("rejects empty and missing model assets")
    {
        check_error([&] { (void)OnnxDetector::load({}, {}); },
                    YoloErrorCode::FileIo, "path");
        check_error([&] { (void)OnnxDetector::load("missing.onnx", {}); },
                    YoloErrorCode::FileIo, "regular file");
    }
}
