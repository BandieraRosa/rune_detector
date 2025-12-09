#pragma once

#include "feature_node.hpp"

namespace rune_detector
{
/**
 * @brief 追踪特征节点
 *
 * @note 用于在视觉识别过程中维护特征节点的历史追踪信息，
 *       支持追踪节点及其对应时间戳的记录与访问。
 */
class TrackingFeatureNode : public FeatureNode
{
  /// @brief 智能指针类型
  using Ptr = std::shared_ptr<TrackingFeatureNode>;

  /// @brief 历史追踪节点缓存
  DEFINE_PROPERTY_WITH_INIT(HistoryNodes, public, protected,
                            (std::deque<FeatureNodeConstPtr>),
                            std::deque<FeatureNodeConstPtr>{});
  /// @brief 历史追踪时间戳缓存
  DEFINE_PROPERTY_WITH_INIT(HistoryTicks, public, protected, (std::deque<int64_t>),
                            std::deque<int64_t>{});

 public:
  /// @brief 默认构造函数
  TrackingFeatureNode() = default;

  /// @brief 拷贝构造函数（禁用）
  TrackingFeatureNode(const TrackingFeatureNode&) = delete;

  /// @brief 移动构造函数（禁用）
  TrackingFeatureNode(TrackingFeatureNode&&) = delete;

  /// @brief 默认析构函数
  virtual ~TrackingFeatureNode() = default;

  /**
   * @brief 将通用特征节点指针安全转换为追踪特征节点指针
   * @param[in] p_feature 通用特征节点指针
   * @return 追踪特征节点智能指针
   */
  static inline std::shared_ptr<TrackingFeatureNode> Cast(const FeatureNodePtr& p_feature)
  {
    return std::dynamic_pointer_cast<TrackingFeatureNode>(p_feature);
  }

  /**
   * @brief 将通用特征节点常量指针安全转换为追踪特征节点常量指针
   * @param[in] p_feature 通用特征节点常量指针
   * @return 追踪特征节点常量智能指针
   */
  static inline const std::shared_ptr<const TrackingFeatureNode> Cast(
      const FeatureNodeConstPtr& p_feature)
  {
    return std::dynamic_pointer_cast<const TrackingFeatureNode>(p_feature);
  }
};

/// @brief 追踪特征节点智能指针类型
using TrackingFeatureNodePtr = std::shared_ptr<TrackingFeatureNode>;

/// @brief 追踪特征节点常量智能指针类型
using TrackingFeatureNodeConstPtr = std::shared_ptr<const TrackingFeatureNode>;

}  // namespace rune_detector