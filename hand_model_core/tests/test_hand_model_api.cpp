#include "kfcore/hand_models/core.hpp"

#include <tinytest.h>

#include <type_traits>

using namespace kfcore::hand_models;

spec("hand model core public pipeline API")
{
    it("exposes a backend interface and a non-copyable Hand pipeline")
    {
        check_true(std::is_abstract_v<HandInferenceBackend>);
        check_false(std::is_copy_constructible_v<HandPipeline>);
        check_false(std::is_copy_assignable_v<HandPipeline>);
    }
}
