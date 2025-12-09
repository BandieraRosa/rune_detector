/**
 * @file geom_utils.hpp
 * @author 张峰玮 (3480409161@qq.com)
 * @brief 常用几何与向量计算工具头文件 (已移除 type_utils 依赖)
 * @date 2025-7-15
 */

#pragma once

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <cmath>
#include <geometry_msgs/msg/pose.hpp>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <vector>

//! 角度制
enum AngleMode : bool
{
  RAD = true,
  DEG = false
};

//! 欧拉角转轴枚举
enum class EulerAxis : uint8_t
{
  X = 0,  //!< X 轴
  Y = 1,  //!< Y 轴
  Z = 2   //!< Z 轴
};

// --------------------[内部辅助：分量获取]--------------------
namespace detail
{
// 针对 cv::Point_ (2D)
template <typename T>
inline T get_x(const cv::Point_<T>& p)
{
  return p.x;
}
template <typename T>
inline T get_y(const cv::Point_<T>& p)
{
  return p.y;
}
template <typename T>
inline T get_z(const cv::Point_<T>&)
{
  return T(0);
}  // 2D 点 Z 轴为 0

// 针对 cv::Point3_ (3D)
template <typename T>
inline T get_x(const cv::Point3_<T>& p)
{
  return p.x;
}
template <typename T>
inline T get_y(const cv::Point3_<T>& p)
{
  return p.y;
}
template <typename T>
inline T get_z(const cv::Point3_<T>& p)
{
  return p.z;
}

// 针对 cv::Vec (N维)
template <typename T, int N>
inline T get_x(const cv::Vec<T, N>& v)
{
  return v[0];
}
template <typename T, int N>
inline T get_y(const cv::Vec<T, N>& v)
{
  return v[1];
}
template <typename T, int N>
inline T get_z(const cv::Vec<T, N>& v)
{
  if constexpr (N >= 3)
  {
    return v[2];
  }
  else
  {
    return T(0);
  }
}
}  // namespace detail

// -------------------- 角度规范化函数 --------------------
template <typename T>
inline T normalize_degree(T degrees)
{
  while (degrees > 180)
  {
    degrees -= 360;
  }
  while (degrees <= -180)
  {
    degrees += 360;
  }
  return degrees;
}

template <typename T>
inline T normalize_radian(T radians)
{
  while (radians > CV_PI)
  {
    radians -= 2 * CV_PI;
  }
  while (radians <= -CV_PI)
  {
    radians += 2 * CV_PI;
  }
  return radians;
}

// ------------------------【常用变换公式】------------------------

/**
 * @brief 角度转换为弧度
 */
template <typename Tp>
inline Tp deg2rad(Tp deg)
{
  return deg * static_cast<Tp>(CV_PI) / static_cast<Tp>(180);
}

/**
 * @brief 弧度转换为角度
 */
template <typename Tp>
inline Tp rad2deg(Tp rad)
{
  return rad * static_cast<Tp>(180) / static_cast<Tp>(CV_PI);
}

/**
 * @brief Point类型转换为Matx类型
 */
template <typename Tp>
inline cv::Matx<Tp, 3, 1> point2matx(cv::Point3_<Tp> point)
{
  return cv::Matx<Tp, 3, 1>(point.x, point.y, point.z);
}

/**
 * @brief Matx类型转换为Point类型
 */
template <typename Tp>
inline cv::Point3_<Tp> matx2point(cv::Matx<Tp, 3, 1> matx)
{
  return cv::Point3_<Tp>(matx(0), matx(1), matx(2));
}

/**
 * @brief Matx类型转换为Vec类型
 */
template <typename Tp>
inline cv::Vec<Tp, 3> matx2vec(cv::Matx<Tp, 3, 1> matx)
{
  return cv::Vec<Tp, 3>(matx(0), matx(1), matx(2));
}

// ------------------------【几何距离计算】------------------------

/**
 * @brief 计算向量之间的欧氏距离 (兼容 2D 和 3D)
 *
 * @tparam T1 第一个向量类型 (支持 cv::Point, cv::Point3, cv::Vec)
 * @tparam T2 第二个向量类型
 * @param v1 第一个向量
 * @param v2 第二个向量
 * @return 欧氏距离
 */
template <typename T1, typename T2>
inline auto get_dist(const T1& v1, const T2& v2)
{
  // 使用 detail 命名空间的 getter，支持 2D/3D 混合
  const auto DX = detail::get_x(v1) - detail::get_x(v2);
  const auto DY = detail::get_y(v1) - detail::get_y(v2);
  const auto DZ = detail::get_z(v1) - detail::get_z(v2);
  return std::sqrt(DX * DX + DY * DY + DZ * DZ);
}

/**
 * @brief 计算单位向量
 */
template <typename Tp>
inline cv::Point_<Tp> get_unit_vector(const cv::Point_<Tp>& v)
{
  return v / sqrt(v.x * v.x + v.y * v.y);
}

/**
 * @brief 求两个向量的最小夹角
 */
template <typename Tp1, typename Tp2>
inline Tp1 get_vector_min_angle(const cv::Point_<Tp1>& v1, const cv::Point_<Tp2>& v2,
                                AngleMode mode = RAD)
{
  Tp1 cos_theta = (v1.x * v2.x + v1.y * v2.y) / (std::sqrt(v1.x * v1.x + v1.y * v1.y) *
                                                 std::sqrt(v2.x * v2.x + v2.y * v2.y));
  Tp1 theta = std::acos(cos_theta);
  if (std::isnan(theta))
  {
    return Tp1(0);
  }
  return mode ? theta : rad2deg(theta);
}

/**
 * @brief 计算点集的方向向量 (逐差法)
 */
template <typename Tp>
inline cv::Point2f get_points_direction_vector(const std::vector<cv::Point_<Tp>>& points)
{
  cv::Point2f direction{};
  if (points.size() <= 1)
  {
    return direction;
  }

  size_t half_size = points.size() / 2;
  for (size_t i = 0; i < half_size; ++i)
  {
    direction -= static_cast<cv::Point2f>(points[i]);
  }
  for (size_t i = half_size + (points.size() % 2); i < points.size(); ++i)
  {
    direction += static_cast<cv::Point2f>(points[i]);
  }

  direction /= static_cast<float>(half_size * (half_size + (points.size() % 2)));
  float norm = std::sqrt(direction.x * direction.x + direction.y * direction.y);
  if (norm > std::numeric_limits<float>::epsilon())
  {
    direction /= norm;
  }

  return direction;
}

/**
 * @brief 计算两直线交点 (Point 输入)
 */
template <typename Tp>
inline cv::Point_<Tp> get_line_intersection(const cv::Point_<Tp>& p1,
                                            const cv::Point_<Tp>& p2,
                                            const cv::Point_<Tp>& p3,
                                            const cv::Point_<Tp>& p4)
{
  Tp a1 = p2.y - p1.y;
  Tp b1 = p1.x - p2.x;
  Tp c1 = a1 * p1.x + b1 * p1.y;

  Tp a2 = p4.y - p3.y;
  Tp b2 = p3.x - p4.x;
  Tp c2 = a2 * p3.x + b2 * p3.y;

  Tp delta = a1 * b2 - a2 * b1;

  if (std::abs(delta) < std::numeric_limits<double>::epsilon())
  {
    return cv::Point_<Tp>(std::numeric_limits<Tp>::quiet_NaN(),
                          std::numeric_limits<Tp>::quiet_NaN());
  }
  Tp x = static_cast<int>((b2 * c1 - b1 * c2) / delta);
  Tp y = static_cast<int>((a1 * c2 - a2 * c1) / delta);
  return cv::Point_<Tp>(x, y);
}

/**
 * @brief 计算两直线交点 (Vec 输入)
 */
template <typename Tp1, typename Tp2>
inline auto get_line_intersection(const cv::Vec<Tp1, 4>& line1,
                                  const cv::Vec<Tp2, 4>& line2)
{
  auto a1 = line1[1], b1 = -line1[0], c1 = line1[0] * line1[3] - line1[1] * line1[2];
  auto a2 = line2[1], b2 = -line2[0], c2 = line2[0] * line2[3] - line2[1] * line2[2];
  if (auto delta = a1 * b2 - a2 * b1;
      std::abs(delta) < std::numeric_limits<decltype(delta)>::epsilon())
  {
    using ResultT = decltype((b1 * c2 - b2 * c1) / delta);
    return cv::Point_<ResultT>(std::numeric_limits<ResultT>::quiet_NaN(),
                               std::numeric_limits<ResultT>::quiet_NaN());
  }
  else
  {
    return cv::Point_((b1 * c2 - b2 * c1) / delta, (a2 * c1 - a1 * c2) / delta);
  }
}

/**
 * @brief 计算投影向量
 */
template <typename Tp>
inline cv::Point_<Tp> get_projection_vector(const cv::Point_<Tp>& v1,
                                            const cv::Point_<Tp>& v2)
{
  return (v1.x * v2.x + v1.y * v2.y) / (v2.x * v2.x + v2.y * v2.y) * v2;
}

/**
 * @brief 计算投影长度
 */
template <typename Tp>
inline Tp get_projection(const cv::Point_<Tp>& v1, const cv::Point_<Tp>& v2)
{
  return (v1.x * v2.x + v1.y * v2.y) / sqrt(v2.x * v2.x + v2.y * v2.y);
}

/**
 * @brief 平面向量外积计算 (2D)
 *
 * @param v1 第一个向量
 * @param v2 第二个向量
 * @return 外积结果 , 若 retval = 0,则共线
 */
template <typename Vec1, typename Vec2>
inline auto get_cross(const Vec1& v1, const Vec2& v2)
{
  const auto V1_X = detail::get_x(v1);
  const auto V1_Y = detail::get_y(v1);
  const auto V2_X = detail::get_x(v2);
  const auto V2_Y = detail::get_y(v2);
  return V1_X * V2_Y - V1_Y * V2_X;
}

/**
 * @brief 欧拉角转换为旋转矩阵
 */
template <typename Tp>
inline cv::Matx<Tp, 3, 3> euler2mat(Tp val, EulerAxis axis)
{
  Tp s = std::sin(val), c = std::cos(val);
  switch (axis)
  {
    case EulerAxis::X:
      return {1, 0, 0, 0, c, -s, 0, s, c};
    case EulerAxis::Y:
      return {c, 0, s, 0, 1, 0, -s, 0, c};
    case EulerAxis::Z:
      return {c, -s, 0, s, c, 0, 0, 0, 1};
    default:
      throw std::invalid_argument("Bad argument of the \"axis\": " +
                                  std::to_string(static_cast<int>(axis)));
      return cv::Matx<Tp, 3, 3>::eye();
  }
}

/**
 * @brief 获取两条直线的交点
 */
template <typename Tp1, typename Tp2>
inline cv::Point2f get_cross_point(const cv::Vec<Tp1, 4>& LineA,
                                   const cv::Vec<Tp2, 4>& LineB)
{
  double ka = NAN, kb = NAN;
  ka =
      static_cast<double>(LineA[3] - LineA[1]) / static_cast<double>(LineA[2] - LineA[0]);
  kb =
      static_cast<double>(LineB[3] - LineB[1]) / static_cast<double>(LineB[2] - LineB[0]);
  cv::Point2f cross_point;
  cross_point.x = static_cast<float>(
      (ka * LineA[0] - LineA[1] - kb * LineB[0] + LineB[1]) / (ka - kb));
  cross_point.y = static_cast<float>(
      (ka * kb * (LineA[0] - LineB[0]) + ka * LineB[1] - kb * LineA[1]) / (ka - kb));
  return cross_point;
}

/**
 * @brief 点到直线距离
 */
template <typename Tp1, typename Tp2>
inline auto get_dist(const cv::Vec<Tp1, 4>& line, const cv::Point_<Tp2>& pt,
                     bool direc = true)
{
  auto retval =
      (line(1) * pt.x - line(0) * pt.y + line(0) * line(3) - line(1) * line(2)) /
      std::sqrt(line(0) * line(0) + line(1) * line(1));
  return direc ? retval : std::abs(retval);
}

/**
 * @brief 将陀螺仪欧拉角转化为旋转矩阵
 */
template <typename Tp>
inline cv::Matx<Tp, 3, 3> gyro_euler2_rot_mat(Tp yaw, Tp pitch)
{
  return euler2mat(deg2rad(yaw), EulerAxis::Y) *
         euler2mat(deg2rad(-1 * pitch), EulerAxis::X);
}

inline void pose_to_open_cv(const geometry_msgs::msg::Pose& pose, cv::Vec3d& rvec,
                            cv::Vec3d& tvec)
{
  // 平移向量直接赋值
  tvec[0] = pose.position.x;
  tvec[1] = pose.position.y;
  tvec[2] = pose.position.z;

  // 旋转：四元数 -> 矩阵 -> 向量
  tf2::Quaternion q(pose.orientation.x, pose.orientation.y, pose.orientation.z,
                    pose.orientation.w);
  tf2::Matrix3x3 m(q);

  cv::Matx33d rmat;
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      rmat(i, j) = m[i][j];
    }
  }

  cv::Rodrigues(rmat, rvec);
}

/**
 * @brief 将 OpenCV 的 rvec 和 tvec 转换为 ROS Pose
 */
inline geometry_msgs::msg::Pose open_cv_to_pose(const cv::Vec3d& rvec,
                                                const cv::Vec3d& tvec)
{
  geometry_msgs::msg::Pose pose;

  // 平移
  pose.position.x = tvec[0];
  pose.position.y = tvec[1];
  pose.position.z = tvec[2];

  // 旋转：向量 -> 矩阵 -> 四元数
  cv::Matx33d rmat;
  cv::Rodrigues(rvec, rmat);

  tf2::Matrix3x3 m(rmat(0, 0), rmat(0, 1), rmat(0, 2), rmat(1, 0), rmat(1, 1), rmat(1, 2),
                   rmat(2, 0), rmat(2, 1), rmat(2, 2));

  tf2::Quaternion q;
  m.getRotation(q);

  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();

  return pose;
}
