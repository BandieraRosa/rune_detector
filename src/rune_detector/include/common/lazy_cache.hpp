#pragma once

#include <optional>

/**
 * @brief 延迟求值缓存
 *
 */
template <typename T>
class LazyCache
{
 public:
  LazyCache() = default;

  LazyCache(const LazyCache&) = delete;
  LazyCache& operator=(const LazyCache&) = delete;

  LazyCache(LazyCache&&) noexcept = default;
  LazyCache& operator=(LazyCache&&) noexcept = default;

  template <typename Generator>
  const T& Get(Generator&& gen) const
  {
    if (!value_)
    {
      value_.emplace(std::forward<Generator>(gen)());
    }
    return *value_;
  }

  [[nodiscard]] bool HasValue() const noexcept { return value_.has_value(); }

 private:
  mutable std::optional<T> value_;
};
