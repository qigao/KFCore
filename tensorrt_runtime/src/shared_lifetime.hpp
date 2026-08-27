#pragma once

#include <memory>
#include <utility>

namespace kfcore::tensorrt::detail
{

template <typename T>
using SharedLifetimeAnchor = std::unique_ptr<std::shared_ptr<T>>;

template <typename T>
SharedLifetimeAnchor<T> make_shared_lifetime_anchor(std::shared_ptr<T> owner)
{
    return std::make_unique<std::shared_ptr<T>>(std::move(owner));
}

template <typename T>
SharedLifetimeAnchor<T> copy_shared_lifetime_anchor(const SharedLifetimeAnchor<T>& source)
{
    return source ? std::make_unique<std::shared_ptr<T>>(*source) : nullptr;
}

template <typename T>
std::shared_ptr<T>* abandon_shared_lifetime_anchor(SharedLifetimeAnchor<T>& owner) noexcept
{
    // The caller intentionally leaks this heap anchor only when external-resource cleanup cannot
    // be made safe. Releasing it keeps the retained control-block reference alive.
    return owner.release();
}

} // namespace kfcore::tensorrt::detail
