#include "kfcore/runtime_onnx/runtime.hpp"
#include "tinytest.hpp"

#include <filesystem>
#include <string>
#include <type_traits>
#include <utility>

using namespace kfcore::runtime_onnx;

static_assert(!std::is_copy_constructible_v<Environment>);
static_assert(!std::is_copy_assignable_v<Environment>);
static_assert(!std::is_move_constructible_v<Environment>);
static_assert(!std::is_move_assignable_v<Environment>);
static_assert(!std::is_copy_constructible_v<Session>);
static_assert(!std::is_copy_assignable_v<Session>);
static_assert(!std::is_copy_constructible_v<OwnedSession>);
static_assert(!std::is_copy_assignable_v<OwnedSession>);
static_assert(std::is_same_v<decltype(std::declval<const OwnedSession&>().session()),
                             const Session&>);
static_assert(std::is_same_v<
              decltype(std::declval<const Session&>().declared_input_dimensions(0U)),
              const std::vector<std::int64_t>&>);
static_assert(std::is_same_v<
              decltype(std::declval<const Session&>().declared_output_dimensions(0U)),
              const std::vector<std::int64_t>&>);

spec("model-neutral ONNX Runtime API")
{
    it("rejects an empty model asset before creating a session")
    {
        Environment environment("KFCoreRuntimeOnnxTest");
        ModelContract contract { "test-model",
                                 { { "input", ElementType::Float32, { 1, 1 } } },
                                 { { "output", ElementType::Float32, { 1, 1 } } } };

        bool threw = false;
        try
        {
            Session session(environment, std::filesystem::path {}, std::move(contract), {});
        }
        catch (const Error& error)
        {
            threw = true;
            check(error.code() == ErrorCode::InvalidModelAsset);
            check(std::string(error.what()).find("test-model") != std::string::npos);
        }
        check_true(threw);
    }
}
