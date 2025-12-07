/**
 * @brief 线程安全的延迟求值缓存
 *
 * @tparam T          缓存值类型
 * @tparam ThreadSafe 是否线程安全
 */
#include <concepts>
#include <functional>
#include <mutex>
#include <optional>
template <typename T, bool ThreadSafe = false>
class LazyCache
{
 public:
  LazyCache() = default;

  // 禁用拷贝
  LazyCache(const LazyCache&) = delete;
  LazyCache& operator=(const LazyCache&) = delete;

  // 允许移动
  LazyCache(LazyCache&&) noexcept = default;
  LazyCache& operator=(LazyCache&&) noexcept = default;

  /**
   * @brief 获取缓存值，若未计算则调用 generator 生成并缓存
   *
   * @param generator 用于生成缓存值的可调用对象
   * @return const T& 缓存值的引用
   */
  template <std::invocable Generator>
    requires std::same_as<std::invoke_result_t<Generator>, T>
  const T& get(Generator&& generator) const
  {
    if constexpr (ThreadSafe)
    {
      std::call_once(
          flag_, [this, &generator]
          { value_.emplace(std::invoke(std::forward<Generator>(generator))); });
    }
    else
    {
      if (!value_.has_value())
      {
        value_.emplace(std::invoke(std::forward<Generator>(generator)));
      }
    }
    return *value_;
  }

  /**
   * @brief 检查是否已缓存值
   */
  [[nodiscard]] bool has_value() const noexcept { return value_.has_value(); }

  /**
   * @brief 清除缓存值
   */
  void reset() noexcept
  {
    value_.reset();
    if constexpr (ThreadSafe)
    {
      flag_.~once_flag();
      new (&flag_) std::once_flag{};
    }
  }

 private:
  mutable std::optional<T> value_;
  mutable std::once_flag flag_;  // 仅 ThreadSafe 使用
};