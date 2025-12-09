/**
 * @file rune_fan_hump_param.h
 * @author 张峰玮 (3480409161@qq.com)
 * @brief 神符扇叶突起参数定义
 * @date 2025-7-15
 * @details 定义了用于扇叶顶部、底部中心和底部侧边突起检测的参数结构体 RuneFanHumpParam。
 *          包含滤波器参数、角度偏差限制、最小间隔及距离比例等，并支持 YML 配置管理。
 */

#pragma once

#include <opencv2/core/types.hpp>

#include "common/param.hpp"

/**
 * @struct RuneFanHumpParam
 * @brief 神符扇叶突起参数
 * @details
 * 包含顶部突起、底部中心突起和侧边突起的各种检测参数，用于计算凸包轮廓的角点特征。
 */
struct RuneFanHumpParam : public Param
{
  //=============顶部突起==========================
  float TOP_HUMP_FILTER_LEN_RATIO = 0.040;  //!< 滤波器长度与轮廓点数比例
  float TOP_HUMP_FILTER_SIGMA = 5.0;        //!< 滤波器标准差
  int TOP_HUMP_MIN_INTERVAL = 10;           //!< 顶部突起角点最小间隔
  float TOP_HUMP_MAX_COLLINEAR_DELTA = 25;  //!< 判断共线的最大容许角度偏差（度）
  float TOP_HUMP_MAX_ALIGNMENT_DELTA = 20;  //!< 判断同向的最大容许角度偏差（度）
  float TOP_HUMP_MAX_DIRECTION_DELTA =
      15;  //!< 扇叶中心点指向突起点向量与突起点方向的最大偏差（度）
  float TOP_HUMP_MAX_DISTANCE_RATIO = 1.5;             //!< 中心点到侧边点距离的最大比例
  float TOP_HUMP_MAX_VERTICAL_DISTANCE_RATIO = 0.025;  //!< 最大垂直距离与轮廓长度比例
  float TOP_HUMP_MIN_LINE_DIRECTION_DELTA = 45;        //!< 突起点连线与其方向夹角（度）

  //=============底部中心突起======================
  int BOTTOM_CENTER_HUMP_MIN_INTERVAL = 10;     //!< 不同突起点中心点的最小间隔
  int BOTTOM_CENTER_HUMP_MIN_ANGLE = 30;        //!< 突起方向与扇叶底边的最小角度（度）
  int BOTTOM_CENTER_HUMP_MAX_DELTA_ANGLE = 20;  //!< 突起方向最大偏差角（度）

  //=============底部侧边突起======================
  float SIDE_HUMP_MAX_ANGLE_DELTA = 90;  //!< 侧边突起点与顶部突起点的最大角度差（度）

  void LoadFromNode(rclcpp::Node& node) override
  {
    const std::string PREFIX = "rune_fan_hump.";
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

    get_param("TOP_HUMP_FILTER_LEN_RATIO", TOP_HUMP_FILTER_LEN_RATIO, 0.040f);
    get_param("TOP_HUMP_FILTER_SIGMA", TOP_HUMP_FILTER_SIGMA, 5.0f);
    get_param("TOP_HUMP_MIN_INTERVAL", TOP_HUMP_MIN_INTERVAL, 10);
    get_param("TOP_HUMP_MAX_COLLINEAR_DELTA", TOP_HUMP_MAX_COLLINEAR_DELTA, 25.0f);
    get_param("TOP_HUMP_MAX_ALIGNMENT_DELTA", TOP_HUMP_MAX_ALIGNMENT_DELTA, 20.0f);
    get_param("TOP_HUMP_MAX_DIRECTION_DELTA", TOP_HUMP_MAX_DIRECTION_DELTA, 15.0f);
    get_param("TOP_HUMP_MAX_DISTANCE_RATIO", TOP_HUMP_MAX_DISTANCE_RATIO, 1.5f);
    get_param("TOP_HUMP_MAX_VERTICAL_DISTANCE_RATIO",
              TOP_HUMP_MAX_VERTICAL_DISTANCE_RATIO, 0.025f);
    get_param("TOP_HUMP_MIN_LINE_DIRECTION_DELTA", TOP_HUMP_MIN_LINE_DIRECTION_DELTA,
              45.0f);

    get_param("BOTTOM_CENTER_HUMP_MIN_INTERVAL", BOTTOM_CENTER_HUMP_MIN_INTERVAL, 10);
    get_param("BOTTOM_CENTER_HUMP_MIN_ANGLE", BOTTOM_CENTER_HUMP_MIN_ANGLE, 30);
    get_param("BOTTOM_CENTER_HUMP_MAX_DELTA_ANGLE", BOTTOM_CENTER_HUMP_MAX_DELTA_ANGLE,
              20);

    get_param("SIDE_HUMP_MAX_ANGLE_DELTA", SIDE_HUMP_MAX_ANGLE_DELTA, 90.0f);
  }
};

// 全局默认参数实例
inline RuneFanHumpParam rune_fan_hump_param;  //!< 默认神符扇叶突起参数
