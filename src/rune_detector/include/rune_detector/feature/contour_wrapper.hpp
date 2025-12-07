#pragma once
#include <bitset>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/logging.hpp>
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
  ContourWrapper(const std::vector<cv::Point_<T>>& contour);

  /**
   * @brief 移动构造函数
   *
   * @param contour 轮廓点集
   */
  ContourWrapper(std::vector<cv::Point_<T>>&& contour);

  // 禁用拷贝构造函数
  ContourWrapper(const ContourWrapper&) = delete;
  ContourWrapper& operator=(const ContourWrapper&) = delete;
  // 允许移动
  ContourWrapper(ContourWrapper&&) = default;
  ContourWrapper& operator=(ContourWrapper&&) = default;

 private:
  // 标志位缓存
  mutable std::bitset<16> cache_flags_;

  //----------------【直接存储】---------------------
  std::vector<cv::Point_<T>> points_;
  mutable double area_ = 0;
  mutable double perimeter_close_ = 0;
  mutable double perimeter_open_ = 0;
  mutable cv::Point_<float> center_ = cv::Point_<float>(0, 0);
  mutable double convex_area_ = 0;  //!< 凸包面积

  //----------------【缓存存储】---------------------
  mutable std::unique_ptr<cv::Rect> bounding_rect_;
  mutable std::unique_ptr<cv::RotatedRect> min_area_rect_;
  mutable std::unique_ptr<std::tuple<cv::Point2f, float>> fitted_circle_;
  mutable std::unique_ptr<cv::RotatedRect> fitted_ellipse_;
  mutable std::unique_ptr<std::vector<cv::Point_<T>>> convex_hull_;
  mutable std::unique_ptr<std::vector<int>> convex_hull_idx_;
  // 标志位定义
  enum CacheFlags : uint8_t
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

 private:
  /**
   * @brief 获取多轮廓的凸包轮廓
   */
  static std::shared_ptr<ContourWrapper<T>> GetConvexHullImpl(
      const std::vector<std::shared_ptr<const ContourWrapper<T>>>& contours)
  {
    size_t total_point_size = 0;
    for (const auto& contour : contours)
    {
      total_point_size += contour->points().size();
    }
    std::vector<cv::Point_<T>> all_points;
    all_points.reserve(total_point_size);
    for (const auto& contour : contours)
    {
      const auto& points = contour->points();
      all_points.insert(all_points.end(), points.begin(), points.end());
    }
    std::vector<cv::Point_<T>> convex_hull;
    cv::convexHull(all_points, convex_hull);
    return ContourWrapper<T>::MakeContour(std::move(convex_hull));
  }

 public:
  /**
   * @brief 创建轮廓对象
   *
   * @param contour 轮廓点集
   */
  inline static std::shared_ptr<ContourWrapper<T>> MakeContour(
      const std::vector<cv::Point_<T>>& contour)
  {
    return std::make_shared<ContourWrapper<T>>(contour);
  }

  /**
   * @brief 创建轮廓对象
   *
   * @param contour 轮廓点集
   */
  inline static std::shared_ptr<ContourWrapper<T>> MakeContour(
      std::vector<cv::Point_<T>>&& contour)
  {
    return std::make_shared<ContourWrapper<T>>(std::move(contour));
  }

  /**
   * @brief 获取轮廓点集
   */
  const std::vector<cv::Point_<T>>& Points() const { return points_; }

  /**
   * @brief 获取轮廓面积
   */
  double Area() const
  {
    if (!cache_flags_.test(AREA_CALC))
    {
      area_ = std::abs(cv::contourArea(points_));
      cache_flags_.set(AREA_CALC);
    }
    return area_;
  }

  /**
   * @brief 获取轮廓周长
   */
  double Perimeter(bool close = true) const
  {
    if (close)
    {
      if (!cache_flags_.test(PERIMETER_CLOSE_CALC))
      {
        const auto& contour = this->Points();
        perimeter_close_ = cv::arcLength(contour, true);
        cache_flags_.set(PERIMETER_CLOSE_CALC);
      }
      return perimeter_close_;
    }
    else
    {
      if (!cache_flags_.test(PERIMETER_OPEN_CALC))
      {
        const auto& contour = this->Points();
        perimeter_open_ = cv::arcLength(contour, false);
        cache_flags_.set(PERIMETER_OPEN_CALC);
      }
      return perimeter_open_;
    }
  }

  /**
   * @brief 获取轮廓中心点
   */
  cv::Point_<float> Center() const
  {
    if (!cache_flags_.test(CENTER_CALC))
    {
      const auto& contour = this->Points();
      cv::Scalar mean_val = cv::mean(contour);
      center_ = cv::Point_<float>(static_cast<float>(mean_val[0]),
                                  static_cast<float>(mean_val[1]));
      cache_flags_.set(CENTER_CALC);
    }
    return center_;
  }

  /**
   * @brief 获取轮廓的正外接矩形
   */
  cv::Rect BoundingRect() const
  {
    if (!cache_flags_.test(BOUNDING_RECT_CALC))
    {
      const auto& contour = this->Points();
      bounding_rect_ = std::make_unique<cv::Rect>(cv::boundingRect(contour));
      cache_flags_.set(BOUNDING_RECT_CALC);
    }
    return *bounding_rect_;
  }

  /**
   * @brief 获取轮廓的最小外接矩形
   */
  cv::RotatedRect MinAreaRect() const
  {
    if (!cache_flags_.test(MIN_AREA_RECT_CALC))
    {
      const auto& contour = this->Points();
      min_area_rect_ = std::make_unique<cv::RotatedRect>(cv::minAreaRect(contour));
      cache_flags_.set(MIN_AREA_RECT_CALC);
    }
    return *min_area_rect_;
  }

  /**
   * @brief 获取轮廓的拟合圆
   */
  std::tuple<cv::Point2f, float> FittedCircle() const
  {
    if (!cache_flags_.test(FITTED_CIRCLE_CALC))
    {
      const auto& contour = this->Points();
      if (contour.size() >= 3)
      {
        cv::Point2f center_temp;
        float radius_temp = 0;
        cv::minEnclosingCircle(contour, center_temp, radius_temp);
        fitted_circle_ =
            std::make_unique<std::tuple<cv::Point2f, float>>(center_temp, radius_temp);
        cache_flags_.set(FITTED_CIRCLE_CALC);
      }
      else
      {
        RCLCPP_ERROR(rclcpp::get_logger("rune_detector"),
                     "Contour has less than 3 points, cannot fit a circle.");
        throw std::runtime_error("");
      }
    }
    return *fitted_circle_;
  }

  /**
   * @brief 获取轮廓的拟合椭圆
   */
  cv::RotatedRect FittedEllipse() const
  {
    if (!cache_flags_.test(FITTED_ELLIPSE_CALC))
    {
      const auto& contour = this->Points();
      if (contour.size() >= 5)
      {
        fitted_ellipse_ = std::make_unique<cv::RotatedRect>(cv::fitEllipse(contour));
        cache_flags_.set(FITTED_ELLIPSE_CALC);
      }
      else
      {
        // 点数不足时触发异常
        RCLCPP_ERROR(rclcpp::get_logger("rune_detector"),
                     "Insufficient points for ellipse fitting");
        throw std::runtime_error("");
      }
    }
    return *fitted_ellipse_;
  }

  /**
   * @brief 获取轮廓的凸包
   */
  const std::vector<cv::Point_<T>>& ConvexHull() const
  {
    if (!cache_flags_.test(CONVEX_HULL_CALC))
    {
      const auto& contour = this->Points();
      convex_hull_ = std::make_unique<std::vector<cv::Point_<T>>>();
      cv::convexHull(contour, *convex_hull_);
      cache_flags_.set(CONVEX_HULL_CALC);
    }
    return *convex_hull_;
  }

  /**
   * @brief 获取凸包的索引
   */
  const std::vector<int>& ConvexHullIdx() const
  {
    if (!cache_flags_.test(CONVEX_HULL_IDX_CALC))
    {
      const auto& contour = this->Points();
      convex_hull_idx_ = std::make_unique<std::vector<int>>();
      cv::convexHull(contour, *convex_hull_idx_);
      cache_flags_.set(CONVEX_HULL_IDX_CALC);
    }
    return *convex_hull_idx_;
  }

  /**
   * @brief 获取凸包的面积
   */
  float ConvexArea() const
  {
    if (!cache_flags_.test(CONVEX_HULL_AREA_CALC))
    {
      const auto& convex_contour = this->ConvexHull();
      convex_area_ = std::abs(cv::contourArea(convex_contour));
      cache_flags_.set(CONVEX_HULL_AREA_CALC);
    }
    return convex_area_;
  }

  /**
   * @brief 生成信息字符串
   */
  std::string InfoString() const
  {
    std::ostringstream oss;
    oss << "  Area: " << this->Area() << "\n";
    oss << "  Perimeter (Closed): " << this->Perimeter(true) << "\n";
    oss << "  Center: (" << this->Center().x << ", " << this->Center().y << ")\n";
    return oss.str();
  }

  /**
   * @brief 隐式转换
   */
  operator const std::vector<cv::Point_<T>>&() const { return this->Points(); }

  /**
   * @brief 获取轮廓的凸包轮廓——接口
   */
  static std::shared_ptr<ContourWrapper<T>> GetConvexHull(
      const std::vector<std::shared_ptr<const ContourWrapper<T>>>& contours)
  {
    // 检查所有轮廓是否有效
    for (const auto& contour : contours)
    {
      if (contour == nullptr || contour->Points().empty())
      {
        RCLCPP_ERROR(rclcpp::get_logger("rune_detector"), "Invalid contour");
        throw std::runtime_error("");
      }
    }
    return GetConvexHullImpl(contours);
  }

 public:
  using const_iterator =
      typename std::vector<cv::Point_<T>>::const_iterator;  // 常量迭代器
  using const_reverse_iterator =
      typename std::vector<cv::Point_<T>>::const_reverse_iterator;  // 常量反向迭代器

  const_iterator begin() const noexcept { return points_.begin(); }
  const_iterator end() const noexcept { return points_.end(); }
  const_iterator cbegin() const noexcept { return points_.cbegin(); }
  const_iterator cend() const noexcept { return points_.cend(); }
  const_reverse_iterator rbegin() const noexcept { return points_.rbegin(); }
  const_reverse_iterator rend() const noexcept { return points_.rend(); }
  const_reverse_iterator crbegin() const noexcept { return points_.crbegin(); }
  const_reverse_iterator crend() const noexcept { return points_.crend(); }

  // 常用stl接口
  size_t size() const noexcept { return points_.size(); }
  bool empty() const noexcept { return points_.empty(); }
  const cv::Point_<T>& operator[](size_t idx) const noexcept { return points_[idx]; }
  const cv::Point_<T>& at(size_t idx) const { return points_.at(idx); }
  const cv::Point_<T>& front() const noexcept { return points_.front(); }
  const cv::Point_<T>& back() const noexcept { return points_.back(); }
};

}  // namespace rune_detector