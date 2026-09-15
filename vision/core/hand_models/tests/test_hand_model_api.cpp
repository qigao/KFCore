#include "kfcore/hand_models/runtime.hpp"
#include "kfcore/hand_models/tracking.hpp"

#include "tinytest.hpp"

#include <type_traits>

using namespace kfcore::hand_models;

spec("hand model public API")
{
    it("exposes concrete detector and stateful tracker without pipeline interfaces")
    {
        check_false(std::is_abstract_v<HandDetector>);
        check_false(std::is_copy_constructible_v<HandDetector>);
        check_false(std::is_copy_constructible_v<HandTracker>);
        check_false(std::is_copy_assignable_v<HandTracker>);
    }
}
