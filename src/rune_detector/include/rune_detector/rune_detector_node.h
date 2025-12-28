#pragma once

#include <cv_bridge/cv_bridge.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "rune_detector/camera_param.h"
#include "rune_interfaces/msg/rune_detections.hpp"

// rm_vision_core
#include "vc/core/type_expansion.hpp"
#include "vc/detector/rune_detector.h"
#include "vc/detector/rune_detector_param.h"
#include "vc/math/pose_node.hpp"

class RuneDetectorNode : public rclcpp::Node
{
 public:
  /**
   * @brief 构造函数，初始化符文检测节点
   */
  explicit RuneDetectorNode(const rclcpp::NodeOptions& options);

 private:
  /**
   * @brief 相机内参回调：只接收一次 camera_info，用于初始化内参并缓存，然后取消订阅
   */
  void CameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg);

  /**
   * @brief 图像回调：处理图像，使用已缓存的内参进行检测
   */
  void ImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg);

  std::string image_topic_;
  std::string camera_info_topic_;
  std::string detections_topic_;
  std::string target_color_str_;
  int color_threshold_{50};

  rclcpp::Publisher<rune_interfaces::msg::RuneDetections>::SharedPtr detections_pub_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

  std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;

  // Detector
  std::unique_ptr<RuneDetector> detector_;
};
