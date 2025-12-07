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
  inline static std::shared_ptr<ContourWrapper<T>> MakeContour(
      const std::vector<cv::Point_<T>>& contour);

  /**
   * @brief 创建轮廓对象
   *
   * @param contour 轮廓点集
   */
  inline static std::shared_ptr<ContourWrapper<T>> MakeContour(
      std::vector<cv::Point_<T>>&& contour);

  /**
   * @brief 获取轮廓点集
   */
  const std::vector<cv::Point_<T>>& Points() const;

  /**
   * @brief 获取轮廓面积
   */
  double Area() const;

  /**
   * @brief 获取轮廓周长
   */
  double Perimeter(bool close = true) const;

  /**
   * @brief 获取轮廓中心点
   */
  cv::Point_<float> Center() const;

  /**
   * @brief 获取轮廓的正外接矩形
   */
  cv::Rect BoundingRect() const;

  /**
   * @brief 获取轮廓的最小外接矩形
   */
  cv::RotatedRect MinAreaRect() const;

  /**
   * @brief 获取轮廓的拟合圆
   */
  std::tuple<cv::Point2f, float> FittedCircle() const;

  /**
   * @brief 获取轮廓的拟合椭圆
   */
  cv::RotatedRect FittedEllipse() const;

  /**
   * @brief 获取轮廓的凸包
   */
  const std::vector<cv::Point_<T>>& ConvexHull() const;

  /**
   * @brief 获取凸包的索引
   */
  const std::vector<int>& ConvexHullIdx() const;

  /**
   * @brief 获取凸包的面积
   */
  float ConvexArea() const;

  /**
   * @brief 生成信息字符串
   */
  std::string InfoString() const;

  /**
   * @brief 隐式转换
   */
  operator const std::vector<cv::Point_<T>>&() const;

  /**
   * @brief 获取轮廓的凸包轮廓——接口
   */
  static std::shared_ptr<ContourWrapper<T>> GetConvexHull(
      const std::vector<std::shared_ptr<const ContourWrapper<T>>>& contours);

 private:
  // 标志位缓存
  mutable std::bitset<16> cache_flags_;

  //----------------【直接存储】---------------------
  std::vector<cv::Point_<T>> __points;
  mutable double __area = 0;
  mutable double __perimeter_close = 0;
  mutable double __perimeter_open = 0;
  mutable cv::Point_<float> __center = cv::Point_<float>(0, 0);
  mutable double __convex_area = 0;  //!< 凸包面积

  //----------------【缓存存储】---------------------
  mutable std::unique_ptr<cv::Rect> __bounding_rect;
  mutable std::unique_ptr<cv::RotatedRect> __min_area_rect;
  mutable std::unique_ptr<std::tuple<cv::Point2f, float>> __fitted_circle;
  mutable std::unique_ptr<cv::RotatedRect> __fitted_ellipse;
  mutable std::unique_ptr<std::vector<cv::Point_<T>>> __convex_hull;
  mutable std::unique_ptr<std::vector<int>> __convex_hull_idx;
  // 标志位定义
  enum CacheFlags
  {
    AREA_CALC = 0,              //!< 面积计算
    PERIMETER_CLOSE_CALC = 1,   //!< 周长计算
    PERIMETER_OPEN_CALC = 2,    //!< 开放式周长计算
    CENTER_CALC = 3,            //!< 中心点计算
    BOUNDING_RECT_CALC = 4,     //!< 外接矩形计算
    MIN_AREA_RECT_CALC = 5,     //!< 最小外接矩形计算
    FITTED_CIRCLE_CALC = 6,     //!< 拟合圆计算
    FITTED_ELLIPSE_CALC = 7,    //!< 拟合椭圆计算
    SMOOTHED_CONTOUR_CALC = 8,  //!< 平滑轮廓计算
    CONVEX_HULL_CALC = 9,       //!< 凸包轮廓计算
    CONVEX_HULL_IDX_CALC = 10,  //!< 凸包索引计算
    CONVEX_HULL_AREA_CALC = 11  //!< 凸包面积计算
  };
};

}  // namespace rune_detector