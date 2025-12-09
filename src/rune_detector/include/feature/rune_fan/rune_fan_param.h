#pragma once

#include "common/param.hpp"

/**
 * @brief RuneFanParam 参数模块
 *
 * 神符扇叶识别参数，包括已激活与未激活扇叶的尺寸、面积、比例、3D坐标及坐标系变换矩阵等。
 */
struct RuneFanParam : public Param
{
  //------------------【已激活扇叶参数】-------------------
  float ACTIVE_MAX_SIDE_RATIO = 2.0;   //!< 已激活扇叶的最大长边短边比
  float ACTIVE_MIN_AREA = 100.0;       //!< 已激活扇叶的最小面积
  float ACTIVE_MAX_AREA = 4000.0;      //!< 已激活扇叶的最大面积
  float ACTIVE_MIN_AREA_RATIO = 0.05;  //!< 已激活扇叶面积占最小外接矩形比例下限
  float ACTIVE_MAX_AREA_RATIO = 0.60;  //!< 已激活扇叶面积占最小外接矩形比例上限
  float ACTIVE_MAX_AREA_PERIMETER_RATIO = 0.030;   //!< 已激活扇叶最大面积周长比
  float ACTIVE_MIN_AREA_PERIMETER_RATIO = 0.0002;  //!< 已激活扇叶最小面积周长比

  std::vector<cv::Point3d> ACTIVE_TOP_3D = {
      cv::Point3d(-174, -32, 0), cv::Point3d(0, 0, 0),
      cv::Point3d(174, -32, 0)};  //!< 顶部角点3D坐标
  std::vector<cv::Point3d> ACTIVE_BOTTOM_CENTER_3D = {
      cv::Point3d(0, 350, 0)};  //!< 底部中心角点3D坐标
  std::vector<cv::Point3d> ACTIVE_SIDE_3D = {
      cv::Point3d(-186, 173, 0), cv::Point3d(186, 173, 0)};  //!< 侧面角点3D坐标
  std::vector<cv::Point3d> ACTIVE_BOTTOM_SIDE_3D = {
      cv::Point3d(-57, 350, 0), cv::Point3d(57, 350, 0)};  //!< 底部侧面角点3D坐标

  cv::Matx31d ACTIVE_TRANSLATION =
      cv::Matx31d(0, -505, 0);                        //!< 扇叶坐标系相对神符中心平移矩阵
  cv::Matx33d ACTIVE_ROTATION = cv::Matx33d(1, 0, 0,  //!< 扇叶坐标系相对神符中心旋转矩阵
                                            0, 1, 0, 0, 0, 1);

  float ACTIVE_MIN_AREA_INCOMPLETE = 30.0;                  //!< 残缺扇叶最小面积
  float ACTIVE_MAX_AREA_PERIMETER_RATIO_INCOMPLETE = 0.07;  //!< 残缺扇叶最大面积周长比
  float ACTIVE_MAX_DIRECTION_DELTA_INCOMPLETE = 45.0;       //!< 残缺扇叶方向最大偏移角度

  //-----------------【未激活扇叶参数】--------------------
  float INACTIVE_MAX_AREA_GROWTH_RATIO = 1.5f;      //!< 匹配箭头时最大面积增长比例
  float INACTIVE_MAX_RECT_PROJECTION_RATIO = 1.5f;  //!< 箭头匹配时矩形投影比值
  float INACTIVE_MIN_AREA_RATIO = 0.70;       //!< 灯臂凸包与最小外接矩形最小面积比例
  float INACTIVE_MERGE_DISTANCE_RATIO = 0.5;  //!< 轮廓合并距离阈值比例
  float INACTIVE_MAX_SIDE_RATIO = 6.0;        //!< 灯臂最大长边短边比
  float INACTIVE_MIN_SIDE_RATIO = 2.0;        //!< 灯臂最小长边短边比
  float INACTIVE_MIN_AREA = 100.0;            //!< 灯臂最小面积
  float INACTIVE_MAX_AREA = 4000.0;           //!< 灯臂最大面积
  float INACTIVE_MAX_DISTANCE_RATIO = 6.0;    //!< 灯臂与其他特征最大距离比例
  std::vector<cv::Point3d> INACTIVE_3D = {cv::Point3d(-30, 0, 0), cv::Point3d(30, 0, 0),
                                          cv::Point3d(30, 330, 0),
                                          cv::Point3d(-30, 330, 0)};  //!< 灯臂角点3D坐标
  cv::Matx31d INACTIVE_TRANSLATION =
      cv::Matx31d(0, -505, 0);  //!< 灯臂坐标系相对神符中心平移矩阵
  cv::Matx33d INACTIVE_ROTATION =
      cv::Matx33d(1, 0, 0,  //!< 灯臂坐标系相对神符中心旋转矩阵
                  0, 1, 0, 0, 0, 1);

  void LoadFromNode(rclcpp::Node& node) override
  {
    const std::string PREFIX = "rune_fan.";
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

    // 已激活扇叶标量
    get_param("ACTIVE_MAX_SIDE_RATIO", ACTIVE_MAX_SIDE_RATIO, 2.0f);
    get_param("ACTIVE_MIN_AREA", ACTIVE_MIN_AREA, 100.0f);
    get_param("ACTIVE_MAX_AREA", ACTIVE_MAX_AREA, 4000.0f);
    get_param("ACTIVE_MIN_AREA_RATIO", ACTIVE_MIN_AREA_RATIO, 0.05f);
    get_param("ACTIVE_MAX_AREA_RATIO", ACTIVE_MAX_AREA_RATIO, 0.60f);
    get_param("ACTIVE_MAX_AREA_PERIMETER_RATIO", ACTIVE_MAX_AREA_PERIMETER_RATIO, 0.030f);
    get_param("ACTIVE_MIN_AREA_PERIMETER_RATIO", ACTIVE_MIN_AREA_PERIMETER_RATIO,
              0.0002f);
    get_param("ACTIVE_MIN_AREA_INCOMPLETE", ACTIVE_MIN_AREA_INCOMPLETE, 30.0f);
    get_param("ACTIVE_MAX_AREA_PERIMETER_RATIO_INCOMPLETE",
              ACTIVE_MAX_AREA_PERIMETER_RATIO_INCOMPLETE, 0.07f);
    get_param("ACTIVE_MAX_DIRECTION_DELTA_INCOMPLETE",
              ACTIVE_MAX_DIRECTION_DELTA_INCOMPLETE, 45.0f);

    // 已激活扇叶 3D 点
    {
      std::vector<double> def = {-174, -32, 0, 0, 0, 0, 174, -32, 0};
      get_param("ACTIVE_TOP_3D", def, def);
      RestorePoints3(def, ACTIVE_TOP_3D);
    }
    {
      std::vector<double> def = {0, 350, 0};
      get_param("ACTIVE_BOTTOM_CENTER_3D", def, def);
      RestorePoints3(def, ACTIVE_BOTTOM_CENTER_3D);
    }
    {
      std::vector<double> def = {-186, 173, 0, 186, 173, 0};
      get_param("ACTIVE_SIDE_3D", def, def);
      RestorePoints3(def, ACTIVE_SIDE_3D);
    }
    {
      std::vector<double> def = {-57, 350, 0, 57, 350, 0};
      get_param("ACTIVE_BOTTOM_SIDE_3D", def, def);
      RestorePoints3(def, ACTIVE_BOTTOM_SIDE_3D);
    }

    // 已激活扇叶变换矩阵
    {
      std::vector<double> def = {0, -505, 0};
      get_param("ACTIVE_TRANSLATION", def, def);
      RestoreVec3(def, ACTIVE_TRANSLATION);
    }
    {
      std::vector<double> def = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      get_param("ACTIVE_ROTATION", def, def);
      RestoreMat33(def, ACTIVE_ROTATION);
    }

    // 未激活扇叶标量
    get_param("INACTIVE_MAX_AREA_GROWTH_RATIO", INACTIVE_MAX_AREA_GROWTH_RATIO, 1.5f);
    get_param("INACTIVE_MAX_RECT_PROJECTION_RATIO", INACTIVE_MAX_RECT_PROJECTION_RATIO,
              1.5f);
    get_param("INACTIVE_MIN_AREA_RATIO", INACTIVE_MIN_AREA_RATIO, 0.70f);
    get_param("INACTIVE_MERGE_DISTANCE_RATIO", INACTIVE_MERGE_DISTANCE_RATIO, 0.5f);
    get_param("INACTIVE_MAX_SIDE_RATIO", INACTIVE_MAX_SIDE_RATIO, 6.0f);
    get_param("INACTIVE_MIN_SIDE_RATIO", INACTIVE_MIN_SIDE_RATIO, 2.0f);
    get_param("INACTIVE_MIN_AREA", INACTIVE_MIN_AREA, 100.0f);
    get_param("INACTIVE_MAX_AREA", INACTIVE_MAX_AREA, 4000.0f);
    get_param("INACTIVE_MAX_DISTANCE_RATIO", INACTIVE_MAX_DISTANCE_RATIO, 6.0f);

    // 未激活扇叶 3D 点
    {
      std::vector<double> def = {-30, 0, 0, 30, 0, 0, 30, 330, 0, -30, 330, 0};
      get_param("INACTIVE_3D", def, def);
      RestorePoints3(def, INACTIVE_3D);
    }

    // 未激活扇叶变换矩阵
    {
      std::vector<double> def = {0, -505, 0};
      get_param("INACTIVE_TRANSLATION", def, def);
      RestoreVec3(def, INACTIVE_TRANSLATION);
    }
    {
      std::vector<double> def = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      get_param("INACTIVE_ROTATION", def, def);
      RestoreMat33(def, INACTIVE_ROTATION);
    }
  }
};

inline RuneFanParam rune_fan_param;  //!< 全局RuneFan参数实例

/**
 * @brief 神符扇叶绘制参数
 */
struct RuneFanDrawParam : public Param
{
  /**
   * @brief 已激活扇叶绘制参数
   */
  struct Active
  {
    cv::Scalar color = cv::Scalar(0, 255, 0);            //!< 绘制颜色
    int thickness = 2;                                   //!< 线条粗细
    int point_radius = 3;                                //!< 点半径
    double font_scale = 0.5;                             //!< 文字大小
    int font_thickness = 1;                              //!< 文字粗细
    cv::Scalar font_color = cv::Scalar(255, 255, 255);   //!< 文字颜色
    int arrow_thickness = 2;                             //!< 箭头粗细
    double arrow_length = 50.0;                          //!< 箭头长度
    cv::Scalar arrow_color = cv::Scalar(255, 255, 255);  //!< 箭头颜色

  } active;

  /**
   * @brief 未激活扇叶绘制参数
   */
  struct Inactive
  {
    cv::Scalar color = cv::Scalar(0, 255, 0);            //!< 绘制颜色
    int thickness = 2;                                   //!< 线条粗细
    int point_radius = 3;                                //!< 点半径
    double font_scale = 0.5;                             //!< 文字大小
    int font_thickness = 1;                              //!< 文字粗细
    cv::Scalar font_color = cv::Scalar(255, 255, 255);   //!< 文字颜色
    int arrow_thickness = 2;                             //!< 箭头粗细
    double arrow_length = 50.0;                          //!< 箭头长度
    cv::Scalar arrow_color = cv::Scalar(255, 255, 255);  //!< 箭头颜色

  } inactive;

  void LoadFromNode(rclcpp::Node& node) override
  {
    const std::string PREFIX = "rune_fan_draw.";
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

    // 已激活扇叶绘制参数
    {
      std::vector<double> def_color = {0, 255, 0};
      get_param("active_color", def_color, def_color);
      RestoreScalar(def_color, active.color);

      get_param("active_thickness", active.thickness, 2);
      get_param("active_point_radius", active.point_radius, 3);
      get_param("active_font_scale", active.font_scale, 0.5);
      get_param("active_font_thickness", active.font_thickness, 1);

      std::vector<double> def_font_color = {255, 255, 255};
      get_param("active_font_color", def_font_color, def_font_color);
      RestoreScalar(def_font_color, active.font_color);

      get_param("active_arrow_thickness", active.arrow_thickness, 2);
      get_param("active_arrow_length", active.arrow_length, 50.0);

      std::vector<double> def_arrow_color = {255, 255, 255};
      get_param("active_arrow_color", def_arrow_color, def_arrow_color);
      RestoreScalar(def_arrow_color, active.arrow_color);
    }

    // 未激活扇叶绘制参数
    {
      std::vector<double> def_color = {0, 255, 0};
      get_param("inactive_color", def_color, def_color);
      RestoreScalar(def_color, inactive.color);

      get_param("inactive_thickness", inactive.thickness, 2);
      get_param("inactive_point_radius", inactive.point_radius, 3);
      get_param("inactive_font_scale", inactive.font_scale, 0.5);
      get_param("inactive_font_thickness", inactive.font_thickness, 1);

      std::vector<double> def_font_color = {255, 255, 255};
      get_param("inactive_font_color", def_font_color, def_font_color);
      RestoreScalar(def_font_color, inactive.font_color);

      get_param("inactive_arrow_thickness", inactive.arrow_thickness, 2);
      get_param("inactive_arrow_length", inactive.arrow_length, 50.0);

      std::vector<double> def_arrow_color = {255, 255, 255};
      get_param("inactive_arrow_color", def_arrow_color, def_arrow_color);
      RestoreScalar(def_arrow_color, inactive.arrow_color);
    }
  }
};

inline RuneFanDrawParam rune_fan_draw_param;  //!< 全局RuneFan绘制参数实例
