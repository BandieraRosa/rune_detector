#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

/**
 * @brief 参数展开辅助宏
 */
#define PROPERTY_WRAPPER_EXPAND_PARAMS(...) __VA_ARGS__

/**
 * @brief 定义属性宏
 *
 * @param PROPERTY_NAME 属性名
 * @param READ_SCOPE 读取访问权限
 * @param WRITE_SCOPE 写入访问权限
 * @param TYPE 属性类型
 *
 * @note 会生成 get/set/isSet/clear 方法
 */
#define DEFINE_PROPERTY(PROPERTY_NAME, READ_SCOPE, WRITE_SCOPE, TYPE)                 \
 private:                                                                             \
  using _##PROPERTY_NAME##_type = PROPERTY_WRAPPER_EXPAND_PARAMS TYPE;                \
  std::optional<_##PROPERTY_NAME##_type> _##PROPERTY_NAME##_prop;                     \
  READ_SCOPE:                                                                         \
  const _##PROPERTY_NAME##_type& get##PROPERTY_NAME() const                           \
  {                                                                                   \
    if (!_##PROPERTY_NAME##_prop.has_value()) throw std::runtime_error("属性未设置"); \
    return *_##PROPERTY_NAME##_prop;                                                  \
  }                                                                                   \
  _##PROPERTY_NAME##_type& get##PROPERTY_NAME()                                       \
  {                                                                                   \
    if (!_##PROPERTY_NAME##_prop.has_value()) throw std::runtime_error("属性未设置"); \
    return *_##PROPERTY_NAME##_prop;                                                  \
  }                                                                                   \
  bool isSet##PROPERTY_NAME() const                                           \
  {                                                                                   \
    return _##PROPERTY_NAME##_prop.has_value();                                       \
  }                                                                                   \
  WRITE_SCOPE:                                                                        \
  template <typename U>                                                               \
  void set##PROPERTY_NAME(U&& value)                                                  \
  {                                                                                   \
    _##PROPERTY_NAME##_prop.emplace(std::forward<U>(value));                          \
  }                                                                                   \
  void clear##PROPERTY_NAME()  { _##PROPERTY_NAME##_prop.reset(); }

/**
 * @brief 定义带初始值的属性宏
 *
 * @param PROPERTY_NAME 属性名
 * @param READ_SCOPE 读取访问权限
 * @param WRITE_SCOPE 写入访问权限
 * @param TYPE 属性类型
 * @param VALUE 初始值
 *
 * @note 会生成 get/set/isSet/clear 方法，并在声明时初始化
 */
#define DEFINE_PROPERTY_WITH_INIT(PROPERTY_NAME, READ_SCOPE, WRITE_SCOPE, TYPE,         \
                                  VALUE...)                                             \
 private:                                                                               \
  using _##PROPERTY_NAME##_type = PROPERTY_WRAPPER_EXPAND_PARAMS TYPE;                  \
  std::optional<_##PROPERTY_NAME##_type> _##PROPERTY_NAME##_prop{std::in_place, VALUE}; \
  READ_SCOPE:                                                                           \
  const _##PROPERTY_NAME##_type& get##PROPERTY_NAME() const                             \
  {                                                                                     \
    if (!_##PROPERTY_NAME##_prop.has_value()) throw std::runtime_error("属性未设置");   \
    return *_##PROPERTY_NAME##_prop;                                                    \
  }                                                                                     \
  _##PROPERTY_NAME##_type& get##PROPERTY_NAME()                                         \
  {                                                                                     \
    if (!_##PROPERTY_NAME##_prop.has_value()) throw std::runtime_error("属性未设置");   \
    return *_##PROPERTY_NAME##_prop;                                                    \
  }                                                                                     \
  bool isSet##PROPERTY_NAME() const                                             \
  {                                                                                     \
    return _##PROPERTY_NAME##_prop.has_value();                                         \
  }                                                                                     \
  WRITE_SCOPE:                                                                          \
  template <typename U>                                                                 \
  void set##PROPERTY_NAME(U&& value)                                                    \
  {                                                                                     \
    _##PROPERTY_NAME##_prop.emplace(std::forward<U>(value));                            \
  }                                                                                     \
  void clear##PROPERTY_NAME()  { _##PROPERTY_NAME##_prop.reset(); }
