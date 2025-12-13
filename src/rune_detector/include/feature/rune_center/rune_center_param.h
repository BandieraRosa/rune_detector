/**
 * @file rune_center_param.h
 * @brief 神符中心参数头文件
 * @author 张峰玮 (3480409161@qq.com)
 * @date 2025-08-24
 */

#pragma once
#include <opencv2/core/types.hpp>
#include <rclcpp/rclcpp.hpp>

#include "common/param.hpp"

namespace rune_detector
{
/**
 * @brief 神符中心参数模块
 *
 * 包含神符中心的尺寸、圆形度、轮廓比率及坐标系变换矩阵等参数。
 */
struct RuneCenterParam : public Param
{
  float MAX_AREA = 1000.f;             //!< 最大面积
  float MIN_AREA = 10.f;               //!< 最小面积
  float MAX_SIDE_RATIO = 2.5f;         //!< 最大边长比率
  float MIN_SIDE_RATIO = 0.3f;         //!< 最小边长比率
  float MAX_SUB_AREA_RATIO = 0.2f;     //!< 子轮廓最大面积占比
  float MIN_CONVEX_AREA_RATIO = 0.9f;  //!< 与凸包轮廓的最小面积占比
  float MAX_DEFECT_AREA_RATIO = 0.3f;  //!< 最大缺陷的面积占比
  float MIN_ROUNDNESS = 0.2f;          //!< 最小圆形度
  float MAX_ROUNDNESS = 0.9f;          //!< 最大圆形度
  float MIN_AREA_FOR_RATIO = 20.f;     //!< 启用面积比例判断的最小面积
  float DEFAULT_SIDE = 20.f;           //!< 默认边长(用于强制构造时)

  double CENTER_CONCENTRICITY_RATIO = 0.08;  //!< 父子轮廓同心度与最大轮廓的比值
  cv::Matx31d TRANSLATION = cv::Matx31d(0, 0, -165.44);           //!< 坐标系平移矩阵
  cv::Matx33d ROTATION = cv::Matx33d(1, 0, 0, 0, 1, 0, 0, 0, 1);  //!< 坐标系旋转矩阵

  void LoadFromNode(rclcpp::Node& node) override
  {
    const std::string PREFIX = "rune_center.";
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

    get_param("MAX_AREA", MAX_AREA, 1000.0f);
    get_param("MIN_AREA", MIN_AREA, 10.0f);
    get_param("MAX_SIDE_RATIO", MAX_SIDE_RATIO, 2.5f);
    get_param("MIN_SIDE_RATIO", MIN_SIDE_RATIO, 0.3f);
    get_param("MAX_SUB_AREA_RATIO", MAX_SUB_AREA_RATIO, 0.2f);
    get_param("MIN_CONVEX_AREA_RATIO", MIN_CONVEX_AREA_RATIO, 0.9f);
    get_param("MAX_DEFECT_AREA_RATIO", MAX_DEFECT_AREA_RATIO, 0.3f);
    get_param("MIN_ROUNDNESS", MIN_ROUNDNESS, 0.2f);
    get_param("MAX_ROUNDNESS", MAX_ROUNDNESS, 0.9f);
    get_param("MIN_AREA_FOR_RATIO", MIN_AREA_FOR_RATIO, 20.0f);
    get_param("DEFAULT_SIDE", DEFAULT_SIDE, 20.0f);
    get_param("CENTER_CONCENTRICITY_RATIO", CENTER_CONCENTRICITY_RATIO, 0.08);
    {
      std::vector<double> def = {0, 0, -165.44};
      get_param("TRANSLATION", def, def);
      RestoreVec3(def, TRANSLATION);
    }
    {
      std::vector<double> def = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      get_param("ROTATION", def, def);
      RestoreMat33(def, ROTATION);
    }
  }
};

inline RuneCenterParam rune_center_param;  //!< 神符中心参数实例

/**
 * @brief 神符中心绘制参数
 *
 * 包含神符中心绘制时的颜色、线条粗细及默认半径。
 */
struct RuneCenterDrawParam : Param
{
  cv::Scalar color = cv::Scalar(0, 255, 0);  //!< 颜色
  int thickness = 2;                         //!< 线条粗细
  double default_radius = 150.0;             //!< 默认半径

  //   YML_INIT(RuneCenterDrawParam, YML_ADD_PARAM(color); YML_ADD_PARAM(thickness);
  //            YML_ADD_PARAM(default_radius););
  void LoadFromNode(rclcpp::Node& node) override
  {
    const std::string PREFIX = "rune_center_draw.";
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

    get_param("thickness", thickness, 2);
    get_param("default_radius", default_radius, 150.0);
    {
      std::vector<double> def = {0, 255, 0};
      get_param("color", def, def);
      RestoreScalar(def, color);
    }
  }
};

inline RuneCenterDrawParam rune_center_draw_param;  //!< 神符中心绘制参数实例

}  // namespace rune_detector