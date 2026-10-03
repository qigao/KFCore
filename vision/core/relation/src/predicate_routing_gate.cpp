#include "kfcore/relation/predicate_routing_gate.hpp"

#include "kfcore/relation/error.hpp"
#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::relation
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw RelationError(
        RelationErrorCode::InvalidArgument,
        "PredicateRoutingGate: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw RelationError(
        RelationErrorCode::ModelContractMismatch,
        "PredicateRoutingGate model contract: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& detail)
{
    throw RelationError(
        RelationErrorCode::RuntimeFailure,
        "PredicateRoutingGate runtime: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw RelationError(
        RelationErrorCode::ResourceLimitExceeded,
        "PredicateRoutingGate resource limit: " + detail);
}

std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    const char* subject)
{
    if (left != 0U &&
        right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_resource(
            std::string(subject) + " byte count overflow");
    }
    return left * right;
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag)
        : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw_invalid(
                "calls on one routing gate instance must not overlap");
        }
    }

    ~UseGuard()
    {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag& flag_;
};

const runtime::TensorDescriptor& require_tensor(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const char* name,
    bool is_input)
{
    const auto match = std::find_if(
        tensors.begin(),
        tensors.end(),
        [&](const runtime::TensorDescriptor& tensor)
        {
            return tensor.name == name &&
                   tensor.is_input == is_input;
        });
    if (match == tensors.end())
    {
        throw_contract(
            std::string("missing ") +
            (is_input ? "input '" : "output '") +
            name + "'");
    }
    return *match;
}

void validate_options(
    const PredicateRoutingGateOptions& options)
{
    if (options.embedding_dim == 0U ||
        options.max_predicates == 0U ||
        options.max_tensor_bytes == 0U ||
        options.max_vocabulary_bytes == 0U)
    {
        throw_invalid(
            "dimensions and resource limits must be positive");
    }
}

void validate_contract(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const PredicateRoutingGateOptions& options)
{
    std::size_t input_count = 0U;
    std::size_t output_count = 0U;
    for (const auto& tensor : tensors)
    {
        tensor.is_input ? ++input_count : ++output_count;
    }
    if (input_count != 1U || output_count != 1U)
    {
        throw_contract(
            "routing gate requires exactly 1 input and 1 output");
    }

    const auto& bank = require_tensor(tensors, "W", true);
    const auto& alpha = require_tensor(tensors, "alpha", false);

    if (bank.data_type != runtime::DataType::Float32 ||
        bank.shape.size() != 2U)
    {
        throw_contract("W must be FP32 [V,D]");
    }
    if (bank.shape[0] != -1)
    {
        throw_contract(
            "W vocabulary axis must be dynamic");
    }
    if (bank.shape[1] !=
        static_cast<std::int64_t>(options.embedding_dim))
    {
        throw_contract(
            "W embedding width does not match configured dimension");
    }

    if (alpha.data_type != runtime::DataType::Float32 ||
        alpha.shape.size() != 1U ||
        alpha.shape[0] != -1)
    {
        throw_contract(
            "alpha must be FP32 [V] with a dynamic vocabulary axis");
    }
}

} // namespace

struct PredicateRoutingGate::Impl final
{
    Impl(runtime::ResolvedModel resolved_value,
         PredicateRoutingGateOptions options_value,
         std::string provenance_value)
        : resolved(std::move(resolved_value))
        , options(std::move(options_value))
        , provenance(std::move(provenance_value))
        , context(resolved.model->create_context())
    {
        validate_contract(
            resolved.model->tensors(),
            options);
    }

    runtime::ResolvedModel resolved;
    PredicateRoutingGateOptions options;
    std::string provenance;
    std::unique_ptr<runtime::ExecutionContext> context;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

PredicateRoutingGate::PredicateRoutingGate(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

PredicateRoutingGate::~PredicateRoutingGate() = default;

std::unique_ptr<PredicateRoutingGate>
PredicateRoutingGate::load(
    runtime::Runtime& runtime,
    const runtime::ModelPackage& package,
    const runtime::ExecutionPolicy& policy,
    const PredicateRoutingGateOptions& options)
{
    validate_options(options);
    if (package.model_type() !=
        kPredicateRoutingGateModelType)
    {
        throw_contract(
            "ModelPackage model_type must be "
            "'relation.predicate-routing-gate'");
    }

    try
    {
        auto resolved =
            runtime.load_model(package, policy);
        std::string provenance =
            package.id() + ":" + package.version();
        if (!resolved.route.artifact.sha256.empty())
        {
            provenance += ":" +
                resolved.route.artifact.sha256;
        }

        return std::unique_ptr<PredicateRoutingGate>(
            new PredicateRoutingGate(
                std::make_unique<Impl>(
                    std::move(resolved),
                    options,
                    std::move(provenance))));
    }
    catch (const RelationError&)
    {
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource(
            "routing-gate allocation failed");
    }
}

PredicateVocabulary PredicateRoutingGate::apply(
    PredicateVocabulary vocabulary)
{
    if (!impl_)
    {
        throw_invalid("model state is unavailable");
    }
    UseGuard guard(impl_->in_use);

    if (vocabulary.predicates.size() >
        impl_->options.max_predicates)
    {
        throw_resource(
            "predicate count exceeds configured maximum");
    }

    PredicateVocabulary candidate =
        normalize_predicate_vocabulary(
            std::move(vocabulary),
            impl_->options.embedding_dim,
            impl_->options.max_vocabulary_bytes);

    const std::size_t count =
        candidate.predicates.size();
    const std::size_t input_bytes =
        checked_multiply(
            candidate.embeddings.size(),
            sizeof(float),
            "predicate routing input");
    const std::size_t output_bytes =
        checked_multiply(
            count,
            sizeof(float),
            "predicate routing output");
    if (input_bytes > impl_->options.max_tensor_bytes ||
        output_bytes > impl_->options.max_tensor_bytes)
    {
        throw_resource(
            "routing-gate tensors exceed configured tensor byte limit");
    }

    std::vector<float> alpha(count, 0.0F);

    const runtime::TensorShape bank_shape {
        static_cast<std::int64_t>(count),
        static_cast<std::int64_t>(
            impl_->options.embedding_dim),
    };
    const runtime::TensorShape alpha_shape {
        static_cast<std::int64_t>(count),
    };

    std::vector<runtime::TensorView> inputs {
        {
            "W",
            runtime::DataType::Float32,
            bank_shape,
            candidate.embeddings.data(),
            input_bytes,
            runtime::MemoryKind::Host,
            {},
        },
    };
    std::vector<runtime::MutableTensorView> outputs {
        {
            "alpha",
            runtime::DataType::Float32,
            alpha_shape,
            alpha.data(),
            output_bytes,
            runtime::MemoryKind::Host,
            {},
        },
    };

    try
    {
        impl_->context->run(inputs, outputs);
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }

    for (float value : alpha)
    {
        if (!std::isfinite(value) ||
            value < 0.0F ||
            value > 1.0F)
        {
            throw_runtime(
                "routing gate returned alpha outside [0,1]");
        }
    }

    candidate.spatial_weights = std::move(alpha);
    candidate.routing_gate_provenance =
        impl_->provenance;
    return candidate;
}

std::size_t
PredicateRoutingGate::embedding_dim() const noexcept
{
    return impl_ ?
        impl_->options.embedding_dim :
        0U;
}

const runtime::ExecutionRoute&
PredicateRoutingGate::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty {};
    return impl_ ? impl_->resolved.route : empty;
}

} // namespace kfcore::relation
