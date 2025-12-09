/**
 * @file rune_target_param.h
 * @author 张峰玮 (3480409161@qq.com)
 * @brief 神符靶心参数定义模块
 * @date 2025-08-24
 *
 * @details
 * 定义神符靶心参数结构体
 * RuneTargetParam，用于已激活与未激活靶心的面积、边长比、周长比、PNP 角点、缺口参数等。
 * 同时定义靶心绘制参数结构体
 * RuneTargetDrawParam，包括激活与未激活靶心的颜色、线宽、点半径及字体信息。
 */

#pragma once

#include <opencv2/core/types.hpp>
#include <rclcpp/rclcpp.hpp>

#include "common/param.hpp"

namespace rune_detector
{
//! 神符靶心参数模块
struct RuneTargetParam
{
  //--------------------[通用]------------------------
  //! 靶心半径 / mm，仅用于可视化
  float RADIUS = 150.f;

  //--------------------[已激活靶心]--------------------
  float ACTIVE_MIN_AREA = 60.f;                      //!< 最小面积
  float ACTIVE_MAX_AREA = 6000.f;                    //!< 最大面积
  float ACTIVE_MIN_SIDE_RATIO = 0.99f;               //!< 最小边长比率
  float ACTIVE_MAX_SIDE_RATIO = 1.55f;               //!< 最大边长比率
  float ACTIVE_MIN_AREA_RATIO = 0.8f;                //!< 最小面积比率
  float ACTIVE_MAX_AREA_RATIO = 1.20f;               //!< 最大面积比率
  float ACTIVE_MIN_PERI_RATIO = 0.35f;               //!< 最小周长比率
  float ACTIVE_MAX_PERI_RATIO = 0.80f;               //!< 最大周长比率
  float ACTIVE_MAX_CONVEX_AREA_RATIO = 0.9f;         //!< 最大凸包面积比率
  float ACTIVE_MAX_CONVEX_PERI_RATIO = 0.11f;        //!< 最大凸包周长比率
  float ACTIVE_MIN_AREA_RATIO_SUB = 0.70f;           //!< 子轮廓最小面积比
  float ACTIVE_MAX_AREA_RATIO_SUB_TEN_RING = 0.30f;  //!< 十环时子轮廓面积最大比
  float ACTIVE_DEFAULT_SIDE = 60.f;                  //!< 默认边长
  std::vector<cv::Point3f> ACTIVE_3D = {cv::Point3f(0, 0, 0)};  //!< PNP 中心点 3D 坐标

  //--------------------[未激活靶心]--------------------
  float INACTIVE_MIN_AREA = 60.f;                                 //!< 最小面积
  float INACTIVE_MAX_AREA = 6000.f;                               //!< 最大面积
  float INACTIVE_MIN_SIDE_RATIO = 0.99f;                          //!< 最小边长比率
  float INACTIVE_MAX_SIDE_RATIO = 1.55f;                          //!< 最大边长比率
  float INACTIVE_MIN_AREA_RATIO = 0.8f;                           //!< 最小面积比率
  float INACTIVE_MAX_AREA_RATIO = 1.20f;                          //!< 最大面积比率
  float INACTIVE_MIN_PERI_RATIO = 0.35f;                          //!< 最小周长比率
  float INACTIVE_MAX_PERI_RATIO = 0.80f;                          //!< 最大周长比率
  std::vector<cv::Point3f> INACTIVE_3D = {cv::Point3f(0, 0, 0)};  //!< PNP 中心点 3D 坐标

  //--------------------[缺口检测参数]--------------------
  float GAP_MIN_AREA_RATIO = 0.025f;        //!< 缺陷面积最小比
  float GAP_MAX_AREA_RATIO = 0.20f;         //!< 缺陷面积最大比
  float GAP_MIN_SIDE_RATIO = 1.55f;         //!< 缺陷最小长宽比
  float GAP_MAX_SIDE_RATIO = 8.0f;          //!< 缺陷最大长宽比
  float GAP_CIRCLE_RADIUS_RATIO = 0.7037f;  //!< 缺陷圆半径与最外层圆半径比
  float GAP_MAX_DISTANCE_RATIO = 0.80f;     //!< 缺陷中心距最大比例
  float GAP_MIN_DISTANCE_RATIO = 0.50f;     //!< 缺陷中心距最小比例
  std::vector<cv::Point3d> GAP_3D = {       //!< PNP 缺口角点 3D 坐标
      cv::Point3d(-67.175, -67.175, 0), cv::Point3d(0, -95, 0),
      cv::Point3d(67.175, -67.175, 0),  cv::Point3d(95, 0, 0),
      cv::Point3d(67.175, 67.175, 0),   cv::Point3d(0, 95, 0),
      cv::Point3d(-67.175, 67.175, 0),  cv::Point3d(-95, 0, 0)};

  cv::Matx33d ROTATION = cv::Matx33d::eye();  //!< 靶心坐标系相对于神符中心的旋转矩阵
  cv::Matx31d TRANSLATION =
      cv::Matx31d(0, -700, 0);  //!< 靶心坐标系相对于神符中心的平移矩阵

  /**
   * @brief 从 ROS2 Node 参数加载配置
   *
   * @param node ROS2 节点
   * @param ns   参数命名空间前缀
   */
  void LoadFromNode(rclcpp::Node& node, const std::string& ns = "rune_target")
  {
    const auto PREFIX = ns.empty() ? "" : ns + ".";

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
    // 标量参数
    get_param("RADIUS", RADIUS, 150.f);

    get_param("ACTIVE_MIN_AREA", ACTIVE_MIN_AREA, 60.f);
    get_param("ACTIVE_MAX_AREA", ACTIVE_MAX_AREA, 6000.f);
    get_param("ACTIVE_MIN_SIDE_RATIO", ACTIVE_MIN_SIDE_RATIO, 0.99f);
    get_param("ACTIVE_MAX_SIDE_RATIO", ACTIVE_MAX_SIDE_RATIO, 1.55f);
    get_param("ACTIVE_MIN_AREA_RATIO", ACTIVE_MIN_AREA_RATIO, 0.8f);
    get_param("ACTIVE_MAX_AREA_RATIO", ACTIVE_MAX_AREA_RATIO, 1.20f);
    get_param("ACTIVE_MIN_PERI_RATIO", ACTIVE_MIN_PERI_RATIO, 0.35f);
    get_param("ACTIVE_MAX_PERI_RATIO", ACTIVE_MAX_PERI_RATIO, 0.80f);
    get_param("ACTIVE_MAX_CONVEX_AREA_RATIO", ACTIVE_MAX_CONVEX_AREA_RATIO, 0.9f);
    get_param("ACTIVE_MAX_CONVEX_PERI_RATIO", ACTIVE_MAX_CONVEX_PERI_RATIO, 0.11f);
    get_param("ACTIVE_MIN_AREA_RATIO_SUB", ACTIVE_MIN_AREA_RATIO_SUB, 0.70f);
    get_param("ACTIVE_MAX_AREA_RATIO_SUB_TEN_RING", ACTIVE_MAX_AREA_RATIO_SUB_TEN_RING,
              0.30f);
    get_param("ACTIVE_DEFAULT_SIDE", ACTIVE_DEFAULT_SIDE, 60.f);

    get_param("INACTIVE_MIN_AREA", INACTIVE_MIN_AREA, 60.f);
    get_param("INACTIVE_MAX_AREA", INACTIVE_MAX_AREA, 6000.f);
    get_param("INACTIVE_MIN_SIDE_RATIO", INACTIVE_MIN_SIDE_RATIO, 0.99f);
    get_param("INACTIVE_MAX_SIDE_RATIO", INACTIVE_MAX_SIDE_RATIO, 1.55f);
    get_param("INACTIVE_MIN_AREA_RATIO", INACTIVE_MIN_AREA_RATIO, 0.8f);
    get_param("INACTIVE_MAX_AREA_RATIO", INACTIVE_MAX_AREA_RATIO, 1.20f);
    get_param("INACTIVE_MIN_PERI_RATIO", INACTIVE_MIN_PERI_RATIO, 0.35f);
    get_param("INACTIVE_MAX_PERI_RATIO", INACTIVE_MAX_PERI_RATIO, 0.80f);

    get_param("GAP_MIN_AREA_RATIO", GAP_MIN_AREA_RATIO, 0.025f);
    get_param("GAP_MAX_AREA_RATIO", GAP_MAX_AREA_RATIO, 0.20f);
    get_param("GAP_MIN_SIDE_RATIO", GAP_MIN_SIDE_RATIO, 1.55f);
    get_param("GAP_MAX_SIDE_RATIO", GAP_MAX_SIDE_RATIO, 8.0f);
    get_param("GAP_CIRCLE_RADIUS_RATIO", GAP_CIRCLE_RADIUS_RATIO, 0.7037f);
    get_param("GAP_MAX_DISTANCE_RATIO", GAP_MAX_DISTANCE_RATIO, 0.80f);
    get_param("GAP_MIN_DISTANCE_RATIO", GAP_MIN_DISTANCE_RATIO, 0.50f);

    // 3D 点 / 矩阵 / 向量，使用扁平 double 向量参数
    {
      std::vector<double> default_active3d = {0.0, 0.0, 0.0};
      get_param("ACTIVE_3D", default_active3d, default_active3d);
      Params::restore_points3(default_active3d, ACTIVE_3D);
    }
    {
      std::vector<double> default_inactive3d = {0.0, 0.0, 0.0};
      get_param("INACTIVE_3D", default_inactive3d, default_inactive3d);
      Params::restore_points3(default_inactive3d, INACTIVE_3D);
    }
    {
      std::vector<double> default_gap3d = {
          {//!< PNP 缺口角点 3D 坐标
           -67.175, -67.175, 0, 0, -95, 0, 67.175,  -67.175, 0, 95,  0, 0,
           67.175,  67.175,  0, 0, 95,  0, -67.175, 67.175,  0, -95, 0, 0}};
      get_param("GAP_3D", default_gap3d, default_gap3d);
      Params::restore_points3(default_gap3d, GAP_3D);
    }
    {
      std::vector<double> default_rot = Params::flatten_mat33(ROTATION);
      get_param("ROTATION", default_rot, default_rot);
      Params::restore_mat33(default_rot, ROTATION);
    }
    {
      std::vector<double> default_trans = {0.0, -700.0, 0.0};
      get_param("TRANSLATION", default_trans, default_trans);
      Params::restore_vec3(default_trans, TRANSLATION);
    }
  }
};

inline RuneTargetParam rune_target_param;  //!< 全局参数实例

//! 靶心绘制参数模块
struct RuneTargetDrawParam
{
  //! 已激活靶心绘制参数
  struct Active
  {
    cv::Scalar color = cv::Scalar(0, 255, 0);  //!< 颜色
    int thickness = 2;                         //!< 线条粗细
    int point_radius = 3;                      //!< 点半径
    double default_circle_radius = 150.0;      //!< 默认圆圈大小
  } active;

  //! 未激活靶心绘制参数
  struct Inactive
  {
    cv::Scalar color = cv::Scalar(0, 255, 0);           //!< 颜色
    int thickness = 2;                                  //!< 线条粗细
    int point_radius = 3;                               //!< 点半径
    double default_circle_radius = 150.0;               //!< 默认圆圈大小
    double font_scale = 0.5;                            //!< 角点标注字体大小
    int font_thickness = 1;                             //!< 角点标注字体粗细
    cv::Scalar font_color = cv::Scalar(255, 255, 255);  //!< 角点标注字体颜色
  } inactive;

  /**
   * @brief 从 ROS2 Node 参数加载绘制配置
   *
   * @param node ROS2 节点
   * @param ns   参数命名空间前缀
   */
  void LoadFromNode(rclcpp::Node& node, const std::string& ns = "rune_target_draw")
  {
    const auto PREFIX = ns.empty() ? "" : ns + ".";

    auto get_param = [&](const std::string& key, auto& var, const auto& default_val)
    {
      // 如果参数未声明，则声明；如果已声明，则获取
      // 这里的 has_parameter 是为了防止重复声明报错
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

    // 加载已激活靶心绘制参数
    {
      std::vector<double> default_color = Params::flatten_scalar(active.color);
      get_param("active.color", default_color, default_color);
      Params::restore_scalar(default_color, active.color);

      get_param("active.thickness", active.thickness, 2);
      get_param("active.point_radius", active.point_radius, 3);
      get_param("active.default_circle_radius", active.default_circle_radius, 150.0);
    }

    // 加载未激活靶心绘制参数
    {
      std::vector<double> default_color = Params::flatten_scalar(inactive.color);
      get_param("inactive.color", default_color, default_color);
      Params::restore_scalar(default_color, inactive.color);

      get_param("inactive.thickness", inactive.thickness, 2);
      get_param("inactive.point_radius", inactive.point_radius, 3);
      get_param("inactive.default_circle_radius", inactive.default_circle_radius, 150.0);
      get_param("inactive.font_scale", inactive.font_scale, 0.5);
      get_param("inactive.font_thickness", inactive.font_thickness, 1);

      std::vector<double> default_font_color =
          Params::flatten_scalar(inactive.font_color);
      get_param("inactive.font_color", default_font_color, default_font_color);
      Params::restore_scalar(default_font_color, inactive.font_color);
    }
  }
};

inline RuneTargetDrawParam rune_target_draw_param;  //!< 全局绘制参数实例

}  // namespace rune_detector