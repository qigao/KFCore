#include "kfcore/yolo/tensorrt.hpp"
#include "tinytest.hpp"

#include <filesystem>

using namespace kfcore::yolo;

spec("TensorRT YOLO integration")
{
    it("rejects a missing engine file with a typed error")
    {
        EngineOptions options;
        check_throws_as(Engine::load(std::filesystem::path("Z:/kfcore/missing.engine"), options),
                        YoloError);
    }
}
