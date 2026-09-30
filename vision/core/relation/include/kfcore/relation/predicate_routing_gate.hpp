#pragma once

#include "kfcore/relation/open_vocabulary_relation.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <cstddef>
#include <memory>
#include <string_view>

namespace kfcore::relation
{

inline constexpr std::string_view kPredicateRoutingGateModelType =
    "relation.predicate-routing-gate";

struct PredicateRoutingGateOptions
{
    std::size_t embedding_dim = 512U;
    std::size_t max_predicates = 19103U;
    std::size_t max_tensor_bytes = 64U * 1024U * 1024U;
    std::size_t max_vocabulary_bytes = 64U * 1024U * 1024U;
};

class PredicateRoutingGate final
{
public:
    ~PredicateRoutingGate();

    PredicateRoutingGate(const PredicateRoutingGate&) = delete;
    PredicateRoutingGate& operator=(const PredicateRoutingGate&) = delete;

    [[nodiscard]] static std::unique_ptr<PredicateRoutingGate>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const PredicateRoutingGateOptions& options = {});

    [[nodiscard]] PredicateVocabulary
    apply(PredicateVocabulary vocabulary);

    [[nodiscard]] std::size_t embedding_dim() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute&
    execution_route() const noexcept;

private:
    struct Impl;
    explicit PredicateRoutingGate(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::relation
