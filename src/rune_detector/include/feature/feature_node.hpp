#pragma once

#include <geometry_msgs/msg/detail/pose__struct.hpp>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <unordered_map>

#include "common/contour_wrapper.hpp"
#include "common/pose_node.hpp"
#include "common/property_wrapper.hpp"

#define FEATURE_NODE_DEBUG 1

namespace rune_detector
{
/**
 * @brief 特征节点基类
 *
 * @note 用于承载视觉识别系统中的通用特征信息，支持图像缓存、位姿缓存、
 *       子特征节点管理及可选的绘制功能。
 */
class FeatureNode : public std::enable_shared_from_this<FeatureNode>
{
  /// @brief 智能指针类型
  using Ptr = std::shared_ptr<FeatureNode>;

 public:
  /// @brief 特征节点映射表类型（key为标识符，value为节点指针）
  using FeatureNodeMap = std::unordered_map<std::string, Ptr>;
  /// @brief 绘制掩码类型
  using DrawMask = std::uint64_t;

 public:
  /**
   * @brief 图像信息缓存块
   *
   * @note 存储了特征节点识别出的图像级信息，例如轮廓、角点和方向等。
   */
  struct ImageCache
  {
    /// @brief 轮廓组
    DEFINE_PROPERTY(Contours, public, public, (std::vector<ContourConstPtr>));
    /// @brief 角点集
    DEFINE_PROPERTY(Corners, public, public, (std::vector<cv::Point2f>));
    /// @brief 中心位置
    DEFINE_PROPERTY(Center, public, public, (cv::Point2f));
    /// @brief 方向向量
    DEFINE_PROPERTY(Direction, public, public, (cv::Point2f));
    /// @brief 宽度（像素单位）
    DEFINE_PROPERTY(Width, public, public, (float));
    /// @brief 高度（像素单位）
    DEFINE_PROPERTY(Height, public, public, (float));
  };

  /**
   * @brief 陀螺仪数据结构
   *
   * @note 存储传感器测量的角度和角速度信息
   */
  struct GyroData
  {
    /**
     * @brief 转动姿态信息
     *
     * @note 包含偏转角、俯仰角、滚转角及其角速度
     */
    struct Rotation
    {
      float yaw = 0.f;          //!< 偏转角（向右运动为正）
      float pitch = 0.f;        //!< 俯仰角（向下运动为正）
      float roll = 0.f;         //!< 滚转角（顺时针运动为正）
      float yaw_speed = 0.f;    //!< 偏转角速度（向右运动为正）
      float pitch_speed = 0.f;  //!< 俯仰角速度（向下运动为正）
      float roll_speed = 0.f;   //!< 滚转角速度（顺时针运动为正）
    } rotation;                 //!< 转动姿态实例
  };

  /**
   * @brief 位姿信息缓存块
   *
   * @note 存储特征节点对应的位姿数据，例如位姿节点映射和陀螺仪信息。
   */
  struct PoseCache
  {
    /// @brief 位姿节点映射表类型
    using PoseNodeMap = std::unordered_map<std::string, PoseNode>;
    /// @brief 位姿节点映射
    DEFINE_PROPERTY_WITH_INIT(PoseNodes, public, public, (PoseNodeMap), PoseNodeMap{});
    /// @brief 陀螺仪位姿信息
    DEFINE_PROPERTY(GyroData, public, public, (GyroData));
  };

  /// @brief 绘制配置结构体前置声明
  struct DrawConfig;
  /// @brief 绘制配置共享指针
  using DrawConfigPtr = std::shared_ptr<DrawConfig>;
  /// @brief 常量绘制配置共享指针
  using DrawConfigConstPtr = std::shared_ptr<const DrawConfig>;

  /// @brief 子特征节点映射表类型前置声明
  struct ChildFeatureType;

 private:
  /// @brief 图像信息缓存
  DEFINE_PROPERTY_WITH_INIT(ImageCache, public, protected, (ImageCache), ImageCache());
  /// @brief 位姿信息缓存
  DEFINE_PROPERTY_WITH_INIT(PoseCache, public, protected, (PoseCache), PoseCache());
  /// @brief 子特征节点映射表
  DEFINE_PROPERTY_WITH_INIT(ChildFeatures, public, protected, (FeatureNodeMap),
                            FeatureNodeMap());
  /// @brief 构建时间戳（单位：tick）
  DEFINE_PROPERTY(Tick, public, public, (int64_t));

 public:
  /**
   * @brief 构造函数
   */
  FeatureNode() = default;

  /**
   * @brief 析构函数
   */
  virtual ~FeatureNode() = default;

  /**
   * @brief 访问图像信息缓存
   * @return 图像信息缓存的常量引用
   */
  virtual const ImageCache& GetImageCache() const { return this->getImageCache(); }

  /**
   * @brief 访问位姿信息缓存
   * @return 位姿信息缓存的常量引用
   */
  virtual const PoseCache& PoseCache() const { return this->getPoseCache(); }

  /**
   * @brief 获取子特征节点映射
   * @return 子特征节点映射的常量引用
   */
  virtual const FeatureNodeMap& GetChildFeatures() const
  {
    return this->getChildFeatures();
  }

  /**
   * @brief 绘制特征节点
   *
   * @param[in,out] image 绘制目标图像
   * @param[in] config 绘制配置指针
   *
   * @note 默认实现为空，子类可以重写此函数以实现具体绘制逻辑。
   */
  virtual void DrawFeature(cv::Mat& image, const DrawConfigConstPtr& config) const
  {
    (void)image;   // 避免未使用参数警告
    (void)config;  // 避免未使用参数警告
  }

 public:
  /**
   * @brief 子特征类型定义
   *
   * @note 用于标识不同类型的子特征节点，便于在特征节点管理中进行分类和查询。
   */
  struct ChildFeatureType
  {
    static std::string rune_target_;  ///< 神符靶心子特征类型
    static std::string rune_center_;  ///< 神符中心子特征类型
    static std::string rune_fan_;     ///< 神符扇叶子特征类型
  };

  struct DrawConfig
  {
    cv::Scalar color = cv::Scalar(100, 255, 0);  ///< 绘制颜色（BGR 格式）
    int thickness = 2;                           ///< 绘制线条粗细
    DrawMask type = 0;                           ///< 绘制类型掩码，用于控制绘制层级
    bool draw_contours = false;                  ///< 是否绘制轮廓
    bool draw_corners = false;                   ///< 是否绘制角点
    bool draw_center = false;                    ///< 是否绘制中心点
    bool draw_pose_nodes = false;                ///< 是否绘制位姿节点
  };

 protected:
};

/// @brief FeatureNode 类型共享指针
using FeatureNodePtr = std::shared_ptr<FeatureNode>;
/// @brief FeatureNode 常量类型共享指针
using FeatureNodeConstPtr = std::shared_ptr<const FeatureNode>;

// #include "feature_node_child_feature_type.h"  // 子特征节点类型定义
// #include "feature_node_draw_config.h"         // 绘制配置定义

/// @brief 神符靶心子特征类型标识
inline std::string FeatureNode::ChildFeatureType::rune_target_ = "rune_target";

/// @brief 神符中心子特征类型标识
inline std::string FeatureNode::ChildFeatureType::rune_center_ = "rune_center";

/// @brief 神符扇叶子特征类型标识
inline std::string FeatureNode::ChildFeatureType::rune_fan_ = "rune_fan";

}  // namespace rune_detector
