#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <stdexcept>
#include <string>

namespace kfcore::runtime
{

const std::string& ModelPackage::model_type() const noexcept
{
    static const std::string unavailable =
        "focused-qualification-stub";
    return unavailable;
}

ResolvedModel Runtime::load_model(
    const ModelPackage&,
    const ExecutionPolicy&) const
{
    throw std::logic_error(
        "focused GT-box qualification uses "
        "OpenVocabularyRelation::load_resolved");
}

} // namespace kfcore::runtime
