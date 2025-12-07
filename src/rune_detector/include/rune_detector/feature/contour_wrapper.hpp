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

 public:
  /**
   * @brief 创建轮廓对象
   *
   * @param contour 轮廓点集
   */
  inline static std::shared_ptr<ContourWrapper<int>> make_contour(
      const std::vector<cv::Point_<int>>& contour);

  /**
   * @brief 创建轮廓对象
   *
   * @param contour 轮廓点集
   */
  inline static std::shared_ptr<ContourWrapper<int>> make_contour(
      std::vector<cv::Point_<int>>&& contour);

  /**
   * @brief 获取轮廓点集
   */
  const std::vector<cv::Point_<int>>& points() const;

  /**
   * @brief 获取轮廓面积
   */
  double area() const;

  /**
   * @brief 获取轮廓周长
   */
  double perimeter(bool close = true) const;

  /**
   * @brief 获取轮廓中心点
   */
  cv::Point_<float> center() const;

  /**
   * @brief 获取轮廓的正外接矩形
   */
  cv::Rect boundingRect() const;

  /**
   * @brief 获取轮廓的最小外接矩形
   */
  cv::RotatedRect minAreaRect() const;

  /**
   * @brief 获取轮廓的拟合圆
   */
  std::tuple<cv::Point2f, float> fittedCircle() const;

  /**
   * @brief 获取轮廓的拟合椭圆
   */
  cv::RotatedRect fittedEllipse() const;

  /**
   * @brief 获取轮廓的凸包
   */
  const std::vector<cv::Point_<int>>& convexHull() const;

  /**
   * @brief 获取凸包的索引
   */
  const std::vector<int>& convexHullIdx() const;

  /**
   * @brief 获取凸包的面积
   */
  float convexArea() const;

  /**
   * @brief 生成信息字符串
   */
  std::string infoString() const;

  /**
   * @brief 隐式转换
   */
  operator const std::vector<cv::Point_<int>>&() const;

  /**
   * @brief 获取轮廓的凸包轮廓——接口
   */
  static std::shared_ptr<ContourWrapper<int>> getConvexHull(
      const std::vector<std::shared_ptr<const ContourWrapper<int>>>& contours);

};

}  // namespace rune_detector