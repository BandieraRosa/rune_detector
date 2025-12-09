#pragma once

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <rclcpp/node.hpp>
#include <vector>

class Param
{
 public:
  /**
   * @brief 从 ROS2 Node 参数加载配置
   *
   * @param node ROS2 节点
   */
  virtual void LoadFromNode(rclcpp::Node& node);

  template <typename Point3T>
  inline std::vector<double> FlattenPoints3(const std::vector<Point3T>& pts)
  {
    std::vector<double> v;
    v.reserve(pts.size() * 3);
    for (const auto& p : pts)
    {
      v.push_back(static_cast<double>(p.x));
      v.push_back(static_cast<double>(p.y));
      v.push_back(static_cast<double>(p.z));
    }
    return v;
  }

  // 从扁平向量还原 Point3f/Point3d 列表
  template <typename Point3T>
  inline void RestorePoints3(const std::vector<double>& data, std::vector<Point3T>& pts)
  {
    if (data.size() % 3 != 0 || data.empty())
    {
      return;
    }

    pts.clear();
    pts.reserve(data.size() / 3);
    for (std::size_t i = 0; i + 2 < data.size(); i += 3)
    {
      pts.emplace_back(static_cast<typename Point3T::value_type>(data[i + 0]),
                       static_cast<typename Point3T::value_type>(data[i + 1]),
                       static_cast<typename Point3T::value_type>(data[i + 2]));
    }
  }

  inline std::vector<double> FlattenMat33(const cv::Matx33d& m)
  {
    std::vector<double> v(9);
    for (int r = 0; r < 3; ++r)
    {
      for (int c = 0; c < 3; ++c)
      {
        v[r * 3 + c] = m(r, c);
      }
    }
    return v;
  }

  inline void RestoreMat33(const std::vector<double>& data, cv::Matx33d& m)
  {
    if (data.size() != 9)
    {
      return;
    }
    for (int r = 0; r < 3; ++r)
    {
      for (int c = 0; c < 3; ++c)
      {
        m(r, c) = data[r * 3 + c];
      }
    }
  }

  inline std::vector<double> FlattenVec3(const cv::Matx31d& v)
  {
    return {v(0), v(1), v(2)};
  }

  inline void RestoreVec3(const std::vector<double>& data, cv::Matx31d& v)
  {
    if (data.size() != 3)
    {
      return;
    }
    v(0) = data[0];
    v(1) = data[1];
    v(2) = data[2];
  }

  inline std::vector<double> FlattenScalar(const cv::Scalar& s)
  {
    // BGR(A)
    return {s[0], s[1], s[2], s[3]};
  }

  inline void RestoreScalar(const std::vector<double>& data, cv::Scalar& s)
  {
    if (data.size() < 3)
    {
      return;
    }
    s[0] = data[0];
    s[1] = data[1];
    s[2] = data[2];
    if (data.size() >= 4)
    {
      s[3] = data[3];
    }
  }
};