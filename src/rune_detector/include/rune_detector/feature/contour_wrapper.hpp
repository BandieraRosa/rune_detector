#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/logging.hpp>
#include <span>
#include <tuple>
#include <vector>

#include "rune_detector/feature/thread_safe_lazy_cache.hpp"

namespace rune_detector
{
/**
 * @brief 轮廓支持的坐标类型：int, float, double
 */
template <typename T>
concept ContourBaseType =
    std::same_as<std::remove_cv_t<T>, int> || std::same_as<std::remove_cv_t<T>, float> ||
    std::same_as<std::remove_cv_t<T>, double>;

/**
 * @brief OpenCV Point 类型概念
 */
template <typename T>
concept PointType = requires(T p) {
  { p.x } -> std::convertible_to<double>;
  { p.y } -> std::convertible_to<double>;
};

/**
 * @class ContourWrapper
 * @brief 高性能轮廓分析器，实现计算结果的延迟加载和智能缓存
 *
 * 1. 基于写时复制(copy-on-write)和延迟加载(lazy initialization)优化内存使用
 * 2. 采用抗锯齿预处理提升几何特征计算精度
 * 3. 自动缓存计算结果，避免重复运算
 * 4. 支持线程安全：const方法可并发调用，非const方法需外部同步
 *
 * @note 构造时会自动进行轮廓抗锯齿处理(approxPolyDP)，后续所有计算基于处理后的轮廓
 */
template <ContourBaseType T = int, bool ThreadSafe = false>
class ContourWrapper
{
 public:
  using ValueType = T;
  using PointType = cv::Point_<T>;
  using PointVec = std::vector<PointType>;

  // 关键点类型：int 坐标时用 float 提高精度，否则保持原类型
  using KeyType = std::conditional_t<std::same_as<T, int>, float, T>;
  using KeyPointType = cv::Point_<KeyType>;

  // 智能指针类型
  using ContourPtr = std::shared_ptr<ContourWrapper>;
  using ContourCPtr = std::shared_ptr<const ContourWrapper>;

  friend class std::allocator<ContourWrapper<ValueType>>;

 private:
  template <typename U>
  using Cache = LazyCache<U, ThreadSafe>;

  PointVec points_;  // 轮廓点集

  mutable Cache<double> area_;
  mutable Cache<double> perimeter_close_;
  mutable Cache<double> perimeter_open_;
  mutable Cache<double> convex_area_;  //!< 凸包面积
  mutable Cache<KeyPointType> center_;

  mutable Cache<cv::Rect> bounding_rect_;
  mutable Cache<cv::RotatedRect> min_area_rect_;
  mutable Cache<std::tuple<cv::Point2f, float>> fitted_circle_;
  mutable Cache<cv::RotatedRect> fitted_ellipse_;
  mutable Cache<PointVec> convex_hull_;
  mutable Cache<std::vector<int>> convex_hull_idx_;

 public:
  /**
   * @brief 构造函数
   * @param contour 轮廓点集
   */
  ContourWrapper(const PointVec& contour);

  /**
   * @brief 移动构造函数
   * @param contour 轮廓点集
   */
  ContourWrapper(PointVec&& contour) noexcept(false);

  // 禁用拷贝构造函数
  ContourWrapper(const ContourWrapper&) = delete;
  ContourWrapper& operator=(const ContourWrapper&) = delete;
  // 允许移动
  ContourWrapper(ContourWrapper&&) noexcept = default;
  ContourWrapper& operator=(ContourWrapper&&) noexcept = default;

  ~ContourWrapper() = default;

  /**
   * @brief 创建轮廓对象的共享指针
   */
  [[nodiscard]] static ContourPtr MakeContour(const PointVec& points);

  [[nodiscard]] static ContourPtr MakeContour(PointVec&& points);

  template <std::input_iterator It>
  [[nodiscard]] static ContourPtr MakeContour(It first, It last);

 private:
  /**
   * @brief 获取多轮廓的凸包轮廓（实现函数）
   */
  static ContourPtr GetConvexHullImpl(
      const std::vector<std::shared_ptr<const ContourWrapper<T>>>& contours);

 public:
  /**
   * @brief 获取轮廓点集
   */
  [[nodiscard]] const PointVec& Points() const noexcept;

  /**
   * @brief 获取轮廓点集的零拷贝只读
   */
  [[nodiscard]] std::span<const PointType> View() const noexcept;

  /**
   * @brief 隐式转换为点集引用
   */
  [[nodiscard]] operator const PointVec&() const noexcept;

  /**
   * @brief 获取轮廓面积
   */
  [[nodiscard]] double Area() const noexcept;

  /**
   * @brief 获取轮廓周长
   */
  [[nodiscard]] double Perimeter(bool closed = true) const noexcept;

  /**
   * @brief 获取轮廓中心点
   */
  [[nodiscard]] KeyPointType Center() const noexcept;

  /**
   * @brief 获取轮廓的正外接矩形
   */
  [[nodiscard]] cv::Rect BoundingRect() const noexcept;

  /**
   * @brief 获取轮廓的最小外接矩形
   */
  [[nodiscard]] cv::RotatedRect MinAreaRect() const noexcept;

  /**
   * @brief 获取轮廓的拟合圆
   */
  [[nodiscard]] std::tuple<cv::Point2f, float> FittedCircle() const;

  /**
   * @brief 获取轮廓的拟合椭圆
   */
  [[nodiscard]] cv::RotatedRect FittedEllipse() const;

  /**
   * @brief 获取轮廓的凸包
   */
  [[nodiscard]] const PointVec& ConvexHull() const;

  /**
   * @brief 获取凸包的索引
   */
  [[nodiscard]] const std::vector<int>& ConvexHullIdx() const;

  /**
   * @brief 获取凸包的面积
   */
  [[nodiscard]] float ConvexArea() const noexcept;

  /**
   * @brief 生成信息字符串
   */
  std::string InfoString() const;

  /**
   * @brief 获取轮廓的凸包轮廓——接口
   */
  static ContourPtr GetConvexHull(
      const std::vector<std::shared_ptr<const ContourWrapper<T>>>& contours);

 public:
  using const_iterator = typename PointVec::const_iterator;
  using const_reverse_iterator = typename PointVec::const_reverse_iterator;

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;
  const_iterator cbegin() const noexcept;
  const_iterator cend() const noexcept;
  const_reverse_iterator rbegin() const noexcept;
  const_reverse_iterator rend() const noexcept;
  const_reverse_iterator crbegin() const noexcept;
  const_reverse_iterator crend() const noexcept;

  size_t size() const noexcept;
  bool empty() const noexcept;
  const PointType& operator[](size_t idx) const noexcept;
  const PointType& at(size_t idx) const;
  const PointType& front() const noexcept;
  const PointType& back() const noexcept;
};

// 非线程安全
using Contour = ContourWrapper<int>;
using ContourF = ContourWrapper<float>;
using ContourD = ContourWrapper<double>;

using ContourPtr = Contour::ContourPtr;
using ContourFPtr = ContourF::ContourPtr;
using ContourDPtr = ContourD::ContourPtr;

using ContourConstPtr = Contour::ContourCPtr;
using ContourFConstPtr = ContourF::ContourCPtr;
using ContourDConstPtr = ContourD::ContourCPtr;

// 线程安全
using ContourTS = ContourWrapper<int, true>;
using ContourFTS = ContourWrapper<float, true>;
using ContourDTS = ContourWrapper<double, true>;

template <typename T>
struct IsContourWrapper : std::false_type
{
};

template <ContourBaseType T, bool TS>
struct IsContourWrapper<ContourWrapper<T, TS>> : std::true_type
{
};

template <typename T>
concept ContourWrapperType = IsContourWrapper<std::remove_cvref_t<T>>::value;

template <typename T>
concept ContourWrapperPtrType = requires {
  typename std::remove_cvref_t<T>::element_type;
  requires ContourWrapperType<typename std::remove_cvref_t<T>::element_type>;
};

/**
 * @brief 轮廓精度转换函数（支持 int/float/double 互转）
 * @tparam OutputType 目标精度类型 (int/float/double)
 * @tparam ContourPtr  输入轮廓指针类型 (自动推导)
 *
 * @param contour 输入轮廓的智能指针
 * @return std::shared_ptr<ContourWrapper<OutputType>> 转换后的轮廓智能指针
 *
 * @note 类型相同时直接返回原指针（无额外开销）
 *       类型不同时生成新轮廓（触发写时复制）
 */
template <ContourBaseType OutputType, ContourWrapperPtrType ContourPtrT>
[[nodiscard]] inline auto convert(const ContourPtrT& contour)
    -> std::shared_ptr<ContourWrapper<OutputType>>;

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
inline void find_contours(cv::InputArray image, std::vector<ContourConstPtr>& contours,
                          cv::OutputArray hierarchy, int mode = cv::RETR_TREE,
                          int method = cv::CHAIN_APPROX_NONE,
                          const cv::Point& offset = cv::Point(0, 0));

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
    cv::InputArray image, std::vector<ContourConstPtr>& contours,
    std::unordered_map<ContourConstPtr, std::tuple<ContourConstPtr, ContourConstPtr,
                                                   ContourConstPtr, ContourConstPtr>>&
        hierarchy,
    int mode = cv::RETR_TREE, int method = cv::CHAIN_APPROX_NONE,
    const cv::Point& offset = cv::Point(0, 0));

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
                  int lineType = cv::LINE_AA);

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
inline void draw_contours(cv::InputOutputArray image,
                          const std::vector<ContourConstPtr>& contours, int contourIdx,
                          const cv::Scalar& color, int thickness = 1,
                          int lineType = cv::LINE_8);

/**
 * @brief 删除指定索引轮廓，并更新层级信息
 *
 * @param contours 轮廓集合
 * @param hierarchy 层级向量
 * @param index 要删除的轮廓下标
 * @return true 删除成功, false 索引无效
 */
[[maybe_unused]] static bool delete_contour(std::vector<ContourConstPtr>& contours,
                                            std::vector<cv::Vec4i>& hierarchy, int index);

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
                                   T& sub_contours_idx);

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
                              T& sub_contours_idx);

template <ContourBaseType T, bool ThreadSafe>
ContourWrapper<T, ThreadSafe>::ContourWrapper(const PointVec& contour) : points_(contour)
{
}

template <ContourBaseType T, bool ThreadSafe>
ContourWrapper<T, ThreadSafe>::ContourWrapper(PointVec&& contour) noexcept(false)
    : points_(std::move(contour))
{
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] auto ContourWrapper<T, ThreadSafe>::MakeContour(const PointVec& points)
    -> ContourPtr
{
  return std::make_shared<ContourWrapper>(points);
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] auto ContourWrapper<T, ThreadSafe>::MakeContour(PointVec&& points)
    -> ContourPtr
{
  return std::make_shared<ContourWrapper>(std::move(points));
}

template <ContourBaseType T, bool ThreadSafe>
template <std::input_iterator It>
[[nodiscard]] auto ContourWrapper<T, ThreadSafe>::MakeContour(It first, It last)
    -> ContourPtr
{
  return std::make_shared<ContourWrapper>(first, last);
}

template <ContourBaseType T, bool ThreadSafe>
auto ContourWrapper<T, ThreadSafe>::GetConvexHullImpl(
    const std::vector<std::shared_ptr<const ContourWrapper<T>>>& contours) -> ContourPtr
{
  size_t total_point_size = 0;
  for (const auto& contour : contours)
  {
    total_point_size += contour->Points().size();
  }
  PointVec all_points;
  all_points.reserve(total_point_size);
  for (const auto& contour : contours)
  {
    const auto& points = contour->Points();
    all_points.insert(all_points.end(), points.begin(), points.end());
  }
  PointVec convex_hull;
  cv::convexHull(all_points, convex_hull);
  return ContourWrapper<T, ThreadSafe>::MakeContour(std::move(convex_hull));
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] const typename ContourWrapper<T, ThreadSafe>::PointVec&
ContourWrapper<T, ThreadSafe>::Points() const noexcept
{
  return points_;
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] std::span<const typename ContourWrapper<T, ThreadSafe>::PointType>
ContourWrapper<T, ThreadSafe>::View() const noexcept
{
  return points_;
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] ContourWrapper<T, ThreadSafe>::operator const PointVec&() const noexcept
{
  return points_;
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] double ContourWrapper<T, ThreadSafe>::Area() const noexcept
{
  return area_.Get([this] { return std::abs(cv::contourArea(points_)); });
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] double ContourWrapper<T, ThreadSafe>::Perimeter(bool closed) const noexcept
{
  if (closed)
  {
    return perimeter_close_.Get([this] { return cv::arcLength(points_, true); });
  }
  return perimeter_open_.Get([this] { return cv::arcLength(points_, false); });
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] typename ContourWrapper<T, ThreadSafe>::KeyPointType
ContourWrapper<T, ThreadSafe>::Center() const noexcept
{
  return center_.Get(
      [this]
      {
        const cv::Scalar MEAN = cv::mean(points_);
        return KeyPointType(static_cast<KeyType>(MEAN[0]), static_cast<KeyType>(MEAN[1]));
      });
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] cv::Rect ContourWrapper<T, ThreadSafe>::BoundingRect() const noexcept
{
  return bounding_rect_.Get([this] { return cv::boundingRect(points_); });
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] cv::RotatedRect ContourWrapper<T, ThreadSafe>::MinAreaRect() const noexcept
{
  return min_area_rect_.Get([this] { return cv::minAreaRect(points_); });
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] std::tuple<cv::Point2f, float> ContourWrapper<T, ThreadSafe>::FittedCircle()
    const
{
  return fitted_circle_.Get(
      [this]() -> std::tuple<cv::Point2f, float>
      {
        if (points_.size() < 3)
        {
          throw std::runtime_error("Insufficient points for circle fitting");
        }
        cv::Point2f center;
        float radius{};
        cv::minEnclosingCircle(points_, center, radius);
        return {center, radius};
      });
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] cv::RotatedRect ContourWrapper<T, ThreadSafe>::FittedEllipse() const
{
  return fitted_ellipse_.Get(
      [this]() -> cv::RotatedRect
      {
        if (points_.size() < 5)
        {
          throw std::runtime_error("Insufficient points for ellipse fitting");
        }
        return cv::fitEllipse(points_);
      });
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] const typename ContourWrapper<T, ThreadSafe>::PointVec&
ContourWrapper<T, ThreadSafe>::ConvexHull() const
{
  return convex_hull_.Get(
      [this]
      {
        PointVec hull;
        cv::convexHull(points_, hull);
        return hull;
      });
}

template <ContourBaseType T, bool ThreadSafe>
[[nodiscard]] const std::vector<int>& ContourWrapper<T, ThreadSafe>::ConvexHullIdx() const
{
  return convex_hull_idx_.Get(
      [this]
      {
        std::vector<int> idx;
        cv::convexHull(points_, idx);
        return idx;
      });
}

template <ContourBaseType T, bool ThreadSafe>
float ContourWrapper<T, ThreadSafe>::ConvexArea() const noexcept
{
  return convex_area_.Get([this] { return std::abs(cv::contourArea(ConvexHull())); });
}

template <ContourBaseType T, bool ThreadSafe>
std::string ContourWrapper<T, ThreadSafe>::InfoString() const
{
  std::ostringstream oss;
  oss << "  Area: " << this->Area() << "\n";
  oss << "  Perimeter (Closed): " << this->Perimeter(true) << "\n";
  oss << "  Center: (" << this->Center().x << ", " << this->Center().y << ")\n";
  return oss.str();
}

template <ContourBaseType T, bool ThreadSafe>
auto ContourWrapper<T, ThreadSafe>::GetConvexHull(
    const std::vector<std::shared_ptr<const ContourWrapper<T>>>& contours) -> ContourPtr
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

template <ContourBaseType T, bool ThreadSafe>
typename ContourWrapper<T, ThreadSafe>::const_iterator
ContourWrapper<T, ThreadSafe>::begin() const noexcept
{
  return points_.begin();
}

template <ContourBaseType T, bool ThreadSafe>
typename ContourWrapper<T, ThreadSafe>::const_iterator
ContourWrapper<T, ThreadSafe>::end() const noexcept
{
  return points_.end();
}

template <ContourBaseType T, bool ThreadSafe>
typename ContourWrapper<T, ThreadSafe>::const_iterator
ContourWrapper<T, ThreadSafe>::cbegin() const noexcept
{
  return points_.cbegin();
}

template <ContourBaseType T, bool ThreadSafe>
typename ContourWrapper<T, ThreadSafe>::const_iterator
ContourWrapper<T, ThreadSafe>::cend() const noexcept
{
  return points_.cend();
}

template <ContourBaseType T, bool ThreadSafe>
typename ContourWrapper<T, ThreadSafe>::const_reverse_iterator
ContourWrapper<T, ThreadSafe>::rbegin() const noexcept
{
  return points_.rbegin();
}

template <ContourBaseType T, bool ThreadSafe>
typename ContourWrapper<T, ThreadSafe>::const_reverse_iterator
ContourWrapper<T, ThreadSafe>::rend() const noexcept
{
  return points_.rend();
}

template <ContourBaseType T, bool ThreadSafe>
typename ContourWrapper<T, ThreadSafe>::const_reverse_iterator
ContourWrapper<T, ThreadSafe>::crbegin() const noexcept
{
  return points_.crbegin();
}

template <ContourBaseType T, bool ThreadSafe>
typename ContourWrapper<T, ThreadSafe>::const_reverse_iterator
ContourWrapper<T, ThreadSafe>::crend() const noexcept
{
  return points_.crend();
}

template <ContourBaseType T, bool ThreadSafe>
size_t ContourWrapper<T, ThreadSafe>::size() const noexcept
{
  return points_.size();
}

template <ContourBaseType T, bool ThreadSafe>
bool ContourWrapper<T, ThreadSafe>::empty() const noexcept
{
  return points_.empty();
}

template <ContourBaseType T, bool ThreadSafe>
const cv::Point_<T>& ContourWrapper<T, ThreadSafe>::operator[](size_t idx) const noexcept
{
  return points_[idx];
}

template <ContourBaseType T, bool ThreadSafe>
const cv::Point_<T>& ContourWrapper<T, ThreadSafe>::at(size_t idx) const
{
  return points_.at(idx);
}

template <ContourBaseType T, bool ThreadSafe>
const cv::Point_<T>& ContourWrapper<T, ThreadSafe>::front() const noexcept
{
  return points_.front();
}

template <ContourBaseType T, bool ThreadSafe>
const cv::Point_<T>& ContourWrapper<T, ThreadSafe>::back() const noexcept
{
  return points_.back();
}

template <ContourBaseType OutputType, ContourWrapperPtrType ContourPtrT>
[[nodiscard]] inline auto convert(const ContourPtrT& contour)
    -> std::shared_ptr<ContourWrapper<OutputType>>
{
  using InputType = typename ContourPtrT::element_type::value_type;

  if constexpr (std::same_as<OutputType, InputType>)
  {
    // 类型相同直接返回原始指针
    return std::static_pointer_cast<ContourWrapper<OutputType>>(contour);
  }
  else
  {
    // 类型转换生成新轮廓
    using OutputPoint = cv::Point_<OutputType>;
    std::vector<OutputPoint> converted;
    converted.reserve(contour->size());
    // 坐标转换（使用完美转发避免拷贝）
    for (const auto& p : contour->Points())
    {
      converted.emplace_back(static_cast<OutputType>(p.x), static_cast<OutputType>(p.y));
    }

    // 构造新轮廓（触发移动语义）
    return ContourWrapper<OutputType>::MakeContour(std::move(converted));
  }

  return nullptr;
}

inline void find_contours(cv::InputArray image, std::vector<ContourConstPtr>& contours,
                          cv::OutputArray hierarchy, int mode, int method,
                          const cv::Point& offset)
{
  std::vector<std::vector<cv::Point>> raw_contours;
  cv::findContours(image, raw_contours, hierarchy, mode, method, offset);
  contours.reserve(raw_contours.size());
  for (auto&& contour : raw_contours)
  {
    contours.emplace_back(ContourWrapper<int>::MakeContour(std::move(contour)));
  }
}

inline void find_contours(
    cv::InputArray image, std::vector<ContourConstPtr>& contours,
    std::unordered_map<ContourConstPtr, std::tuple<ContourConstPtr, ContourConstPtr,
                                                   ContourConstPtr, ContourConstPtr>>&
        hierarchy,
    int mode, int method, const cv::Point& offset)
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

template <typename T>
void draw_contour(cv::Mat& image, const std::shared_ptr<const ContourWrapper<T>>& contour,
                  const cv::Scalar& color, int thickness, int lineType)
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

inline void draw_contours(cv::InputOutputArray image,
                          const std::vector<ContourConstPtr>& contours, int contourIdx,
                          const cv::Scalar& color, int thickness, int lineType)
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

[[maybe_unused]] static bool delete_contour(std::vector<ContourConstPtr>& contours,
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

template <typename T>
void get_all_sub_contours_idx(const std::vector<cv::Vec4i>& hierarchy, int idx,
                              T& sub_contours_idx)
{
  get_all_sub_contours_idx_impl<T>(hierarchy, idx, sub_contours_idx);
}

}  // namespace rune_detector

template class rune_detector::ContourWrapper<int>;
template class rune_detector::ContourWrapper<float>;
template class rune_detector::ContourWrapper<double>;
template class rune_detector::ContourWrapper<int, true>;
template class rune_detector::ContourWrapper<float, true>;
template class rune_detector::ContourWrapper<double, true>;