/**
 * @file rune_param.h
 * @brief 神符组合体参数模块头文件
 * @author 张峰玮 (3480409161@qq.com)
 * @date 2025-08-24
 */

#pragma once
#include <rclcpp/rclcpp.hpp>

#include "common/param.hpp"
namespace rune_detector
{
/**
 * @brief RuneParam 参数模块
 *
 * 用于管理神符整体相关的参数。
 */
struct RuneParam : Param
{
  void LoadFromNode(rclcpp::Node& node) override
  {
    const std::string PREFIX = "rune_combo.";
    auto get_param = [&](const std::string& key, auto& var, const auto& default_val)
    {
      if (!node.has_parameter(PREFIX + key))
      {
        var = node.declare_parameter<std::decay_t<decltype(var)>>(PREFIX + key,
                                                                  default_val);
      }
      else
      {
        node.get_parameter(PREFIX + key, var);
      }
    };
  }
};

//! RuneParam 参数实例
inline RuneParam rune_param;
}  // namespace rune_detector