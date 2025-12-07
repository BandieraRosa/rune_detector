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

/**
 * @brief 增强版轮廓检测函数，返回智能轮廓对象集合
 *
 * @param image 输入二值图像 (建议使用 clone 保留原始数据)
 * @param contours 输出轮廓集合 (Contour_cptr)
 * @param hierarchy 输出轮廓层级信息
 * @param mode 轮廓检索模式 (默认为 cv::RETR_TREE)
 * @param method 轮廓近似方法 (默认为 cv::CHAIN_APPROX_NONE)
 * @param offset 轮廓点坐标偏移量 (默认为 (0,0))
 *
 * @note 自动执行抗锯齿处理 (根据 ENABLE_SMOOTH_CONTOUR_CALC 配置)
 */
inline void find_contours(
    cv::InputArray image,
    std::vector<std::shared_ptr<const ContourWrapper<int>>>& contours,
    cv::OutputArray hierarchy, int mode = cv::RETR_TREE,
    int method = cv::CHAIN_APPROX_NONE, const cv::Point& offset = cv::Point(0, 0))
{
  std::vector<std::vector<cv::Point>> raw_contours;
  cv::findContours(image, raw_contours, hierarchy, mode, method, offset);
  contours.reserve(raw_contours.size());
  for (auto&& contour : raw_contours)
  {
    contours.emplace_back(ContourWrapper<int>::MakeContour(std::move(contour)));
  }
}

/**
 * @brief 增强版轮廓检测函数，返回智能轮廓对象集合，使用 unordered_map 保存层级信息
 *
 * @param image 输入二值图像 (建议使用 clone 保留原始数据)
 * @param contours 输出轮廓集合 (Contour_cptr)
 * @param hierarchy 输出轮廓层级映射，格式为:
 *        std::unordered_map<当前轮廓, std::tuple<后一个轮廓, 前一个轮廓, 内嵌轮廓,
 * 父轮廓>>
 * @param mode 轮廓检索模式 (默认为 cv::RETR_TREE)
 * @param method 轮廓近似方法 (默认为 cv::CHAIN_APPROX_NONE)
 * @param offset 轮廓点坐标偏移量 (默认为 (0,0))
 *
 * @note 自动执行抗锯齿处理 (根据 ENABLE_SMOOTH_CONTOUR_CALC 配置)
 */
inline void find_contours(
    cv::InputArray image,
    std::vector<std::shared_ptr<const ContourWrapper<int>>>& contours,
    std::unordered_map<std::shared_ptr<const ContourWrapper<int>>,
                       std::tuple<std::shared_ptr<const ContourWrapper<int>>,
                                  std::shared_ptr<const ContourWrapper<int>>,
                                  std::shared_ptr<const ContourWrapper<int>>,
                                  std::shared_ptr<const ContourWrapper<int>>>>& hierarchy,
    int mode = cv::RETR_TREE, int method = cv::CHAIN_APPROX_NONE,
    const cv::Point& offset = cv::Point(0, 0))
{
  std::vector<std::vector<cv::Point>> raw_contours;
  std::vector<cv::Vec4i> hierarchy_vec;
  cv::findContours(image, raw_contours, hierarchy_vec, mode, method, offset);
  contours.reserve(raw_contours.size());
  for (auto&& contour : raw_contours)
  {
    contours.emplace_back(ContourWrapper<int>::MakeContour(std::move(contour)));
  }

  hierarchy.reserve(raw_contours.size());
  for (size_t i = 0; i < raw_contours.size(); ++i)
  {
    auto& contour = contours[i];
    auto& h = hierarchy_vec[i];
    hierarchy[contour] = std::make_tuple(
        (h[0] != -1) ? contours[h[0]] : nullptr, (h[1] != -1) ? contours[h[1]] : nullptr,
        (h[2] != -1) ? contours[h[2]] : nullptr, (h[3] != -1) ? contours[h[3]] : nullptr);
  }
}

/**
 * @brief 绘制单个 ContourWrapper 轮廓到图像
 *
 * @tparam T 轮廓数据类型 (int / float / double)
 * @param image 目标图像 (BGR)
 * @param contour 智能轮廓指针
 * @param color 绘制颜色 (默认绿色)
 * @param thickness 线宽 (默认 2)
 * @param lineType 线型 (默认抗锯齿)
 */
template <typename T>
void draw_contour(cv::Mat& image, const std::shared_ptr<const ContourWrapper<T>>& contour,
                  const cv::Scalar& color = cv::Scalar(0, 255, 0), int thickness = 2,
                  int lineType = cv::LINE_AA)
{
  if (!contour || contour->empty())
  {
    return;
  }

  const auto& points = contour->Points();
  std::vector<cv::Point> int_points;
  int_points.reserve(points.size());

  if constexpr (std::is_same_v<T, int>)
  {
    std::transform(points.begin(), points.end(), std::back_inserter(int_points),
                   [](const cv::Point_<T>& p) { return cv::Point(p.x, p.y); });
  }
  else
  {
    std::transform(points.begin(), points.end(), std::back_inserter(int_points),
                   [](const cv::Point_<T>& p)
                   { return cv::Point(cvRound(p.x), cvRound(p.y)); });
  }

  cv::drawContours(image, std::vector<std::vector<cv::Point>>{int_points}, 0, color,
                   thickness, lineType);
}

/**
 * @brief 绘制轮廓集合到图像
 *
 * @param image 输入输出图像
 * @param contours 轮廓集合
 * @param contourIdx 绘制的轮廓索引 (-1 表示绘制所有)
 * @param color 绘制颜色
 * @param thickness 绘制线宽
 * @param lineType 绘制线型
 */
inline void draw_contours(
    cv::InputOutputArray image,
    const std::vector<std::shared_ptr<const ContourWrapper<int>>>& contours,
    int contourIdx, const cv::Scalar& color, int thickness = 1, int lineType = cv::LINE_8)
{
  if (contourIdx < -1 || contourIdx >= static_cast<int>(contours.size()))
  {
    throw std::out_of_range("Invalid contour index");
  }

  for (size_t i = 0; i < contours.size(); ++i)
  {
    if (contourIdx == -1 || contourIdx == static_cast<int>(i))
    {
      const auto& contour = contours[i];
      if (contour)
      {
        draw_contour(image.getMatRef(), contour, color, thickness, lineType);
      }
      else
      {
        RCLCPP_WARN(rclcpp::get_logger("rune_detector"), "Contour at index %li is null.",
                    i);
      }
    }
  }
}

/**
 * @brief 删除指定索引轮廓，并更新层级信息
 *
 * @param contours 轮廓集合
 * @param hierarchy 层级向量
 * @param index 要删除的轮廓下标
 * @return true 删除成功, false 索引无效
 */
static bool delete_contour(
    std::vector<std::shared_ptr<const ContourWrapper<int>>>& contours,
    std::vector<cv::Vec4i>& hierarchy, int index)
{
  if (index < 0 || index >= contours.size())
  {
    return false;
  }

  auto update_hierarchy = [&](int idx, int new_val, int pos)
  {
    if (idx != -1)
    {
      hierarchy[idx][pos] = new_val;
    }
  };

  update_hierarchy(hierarchy[index][0], hierarchy[index][1], 1);
  update_hierarchy(hierarchy[index][1], hierarchy[index][0], 0);

  int sub_idx = hierarchy[index][2];
  while (sub_idx != -1)
  {
    hierarchy[sub_idx][3] = hierarchy[index][3];
    sub_idx = hierarchy[sub_idx][1];
  }

  if (hierarchy[hierarchy[index][3]][2] == index)
  {
    hierarchy[hierarchy[index][3]][2] =
        (hierarchy[index][2] != -1)   ? hierarchy[index][2]
        : (hierarchy[index][0] != -1) ? hierarchy[index][0]
        : (hierarchy[index][1] != -1) ? hierarchy[index][1]
                                      : -1;
  }

  for (auto& h : hierarchy)
  {
    for (int i = 0; i < 4; ++i)
    {
      if (h[i] > index)
      {
        --h[i];
      }
    }
  }

  contours.erase(contours.begin() + index);
  hierarchy.erase(hierarchy.begin() + index);

  return true;
}

/**
 * @brief 递归获取指定轮廓的所有子轮廓下标（实现函数）
 *
 * @tparam T 容器类型，需要支持 insert 接口（如 std::vector<int>）
 * @param hierarchy 所有轮廓的层级结构，由 OpenCV findContours 返回
 * @param idx 当前轮廓下标
 * @param sub_contours_idx 用于存储结果的容器
 *
 * @note 遍历当前轮廓的子节点，并递归获取所有子轮廓的索引。
 */
template <typename T>
void get_all_sub_contours_idx_impl(const std::vector<cv::Vec4i>& hierarchy, int idx,
                                   T& sub_contours_idx)
{
  using namespace std;
  using namespace cv;

  if (hierarchy[idx][2] == -1)
  {
    return;
  }
  int front_child_idx = hierarchy[idx][2];
  while (front_child_idx != -1)
  {
    sub_contours_idx.insert(
        sub_contours_idx.end(),
        static_cast<typename T::value_type>(front_child_idx));  //!< 插入子轮廓下标
    get_all_sub_contours_idx_impl(hierarchy, front_child_idx,
                                  sub_contours_idx);  //!< 递归获取子轮廓
    front_child_idx = hierarchy[front_child_idx][1];  //!< 获取下一个兄弟轮廓下标
  }
  if (hierarchy[hierarchy[idx][2]][0] == -1)
  {
    return;
  }
  int back_child_idx = hierarchy[hierarchy[idx][2]][0];
  while (back_child_idx != -1)
  {
    sub_contours_idx.insert(
        sub_contours_idx.end(),
        static_cast<typename T::value_type>(back_child_idx));  //!< 插入子轮廓下标
    get_all_sub_contours_idx_impl(hierarchy, back_child_idx,
                                  sub_contours_idx);  //!< 递归获取子轮廓
    back_child_idx = hierarchy[back_child_idx][0];    //!< 获取下一个兄弟轮廓下标
  }
}

/**
 * @brief 获取指定轮廓的所有子轮廓下标（接口函数）
 *
 * @tparam T 容器类型，需要支持 insert 接口（如 std::vector<int>）
 * @param hierarchy 所有轮廓的层级结构，由 OpenCV findContours 返回
 * @param idx 指定轮廓下标
 * @param sub_contours_idx 用于存储结果的容器
 *
 * @note 调用内部递归实现函数完成子轮廓索引的收集。
 */
template <typename T>
void get_all_sub_contours_idx(const std::vector<cv::Vec4i>& hierarchy, int idx,
                              T& sub_contours_idx)
{
  get_all_sub_contours_idx_impl<T>(hierarchy, idx, sub_contours_idx);
}

}  // namespace rune_detector