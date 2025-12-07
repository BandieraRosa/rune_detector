#pragma once
#include <bitset>
#include <memory>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

namespace rune_detector
{

template <typename T = int>
class ContourWrapper
{
 public:
  /**
   * @brief 构造函数
   *
   * @param contour 轮廓点集
   */
  ContourWrapper(const std::vector<cv::Point_<int>>& contour);

  /**
   * @brief 移动构造函数
   *
   * @param contour 轮廓点集
   */
  ContourWrapper(std::vector<cv::Point_<int>>&& contour);

  // 禁用拷贝构造函数
  ContourWrapper(const ContourWrapper&) = delete;
  ContourWrapper& operator=(const ContourWrapper&) = delete;
  // 允许移动
  ContourWrapper(ContourWrapper&&) = default;
  ContourWrapper& operator=(ContourWrapper&&) = default;
};

}  // namespace rune_detector