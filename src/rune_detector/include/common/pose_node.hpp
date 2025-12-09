#pragma once

#include <geometry_msgs/msg/detail/pose__struct.hpp>

/**
 * @brief 位姿节点类型别名
 */
using PoseNode = geometry_msgs::msg::Pose;  //!< 位姿节点类型

/**
 * @brief 坐标系类型
 */
struct CoordFrame
{
  static std::string WORLD;   //!< 世界坐标系标识
  static std::string CAMERA;  //!< 相机坐标系标识
  static std::string JOINT;   //!< 转轴坐标系标识
  static std::string GYRO;    //!< 陀螺仪坐标系标识
};

//! 世界坐标系
inline std::string CoordFrame::WORLD = "odom";
//! 相机坐标系
inline std::string CoordFrame::CAMERA = "camera_optical_frame";
//! 转轴坐标系
inline std::string CoordFrame::JOINT = "pitch_link";
//! 陀螺仪坐标系
inline std::string CoordFrame::GYRO = "gimbal_odom";
