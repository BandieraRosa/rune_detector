#include "rune_detector/rune_detector_node.h"

using rune_interfaces::msg::RuneDetections;

namespace
{
geometry_msgs::msg::Quaternion rmat_to_quat_msg(const cv::Matx33d& R)
{
  tf2::Matrix3x3 m(R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1),
                   R(2, 2));
  tf2::Quaternion q;
  m.getRotation(q);
  return tf2::toMsg(q);
}
}  // namespace

RuneDetectorNode::RuneDetectorNode(const rclcpp::NodeOptions& options)
    : Node("rune_detector_node", options)
{
  image_topic_ = this->declare_parameter<std::string>("image_topic", "/image_raw");
  camera_info_topic_ =
      this->declare_parameter<std::string>("camera_info_topic", "/camera_info");
  detections_topic_ =
      this->declare_parameter<std::string>("detections_topic", "/rune/detections");
  target_color_str_ = this->declare_parameter<std::string>("target_color", "red");
  color_threshold_ = this->declare_parameter<int>("color_threshold", 0);

  // Publisher
  detections_pub_ =
      this->create_publisher<RuneDetections>(detections_topic_, rclcpp::SensorDataQoS());

  // Detector
  detector_ = RuneDetector::make_detector();

  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::SensorDataQoS(),
      std::bind(&RuneDetectorNode::CameraInfoCallback, this, std::placeholders::_1));

  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&RuneDetectorNode::ImageCallback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "rune_detector_node started.");
  RCLCPP_INFO(
      this->get_logger(), "image_topic=%s camera_info_topic=%s detections_topic=%s",
      image_topic_.c_str(), camera_info_topic_.c_str(), detections_topic_.c_str());
}

void RuneDetectorNode::CameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg)
{
  rune_detector::update_camera_param_from_camera_info(*info_msg);

  cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*info_msg);

  RCLCPP_INFO(this->get_logger(),
              "Received camera_info once. fx=%.3f fy=%.3f cx=%.3f cy=%.3f, frame_id=%s. "
              "Unsubscribing camera_info.",
              static_cast<double>(info_msg->k[0]), static_cast<double>(info_msg->k[4]),
              static_cast<double>(info_msg->k[2]), static_cast<double>(info_msg->k[5]),
              info_msg->header.frame_id.c_str());

  cam_info_sub_.reset();
}

void RuneDetectorNode::ImageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg)
{
  if (!cam_info_)
  {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Waiting for camera_info on topic '%s' ...",
                         camera_info_topic_.c_str());
    return;
  }

  // Convert image to cv::Mat
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(img_msg, "bgr8");
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  PixChannel target_color =
      (target_color_str_ == "blue") ? PixChannel::BLUE : PixChannel::RED;

  int thresh = color_threshold_;
  if (thresh <= 0)
  {
    thresh = (target_color == PixChannel::RED) ? rune_detector_param.GRAY_THRESHOLD_RED
                                               : rune_detector_param.GRAY_THRESHOLD_BLUE;
  }

  RuneDetector::RawDetectionResult raw{};
  bool ok = false;

  try
  {
    ok =
        detector_->detect(cv_ptr->image, target_color, static_cast<uint8_t>(thresh), raw);
  }
  catch (const std::exception& e)
  {
    RCLCPP_WARN(this->get_logger(), "detect exception: %s", e.what());
    ok = false;
  }

  RuneDetections out;
  out.header = img_msg->header;
  out.header.frame_id = "camera_optical_frame";

  if (out.header.frame_id.empty())
  {
    out.header.frame_id = cam_info_->header.frame_id;
  }

  out.valid = ok;

  // default rune types
  for (size_t i = 0; i < 5; ++i)
  {
    out.rune_type[i] = RuneDetections::UNKNOWN;
  }

  if (ok)
  {
    // PoseNode is "group -> camera" in OpenCV PnP convention
    const PoseNode& t_group_cam = raw.group_to_cam;

    // Position: mm -> m
    out.group_pose.position.x = t_group_cam.tvec()(0) / 1000.0;
    out.group_pose.position.y = t_group_cam.tvec()(1) / 1000.0;
    out.group_pose.position.z = t_group_cam.tvec()(2) / 1000.0;

    out.group_pose.orientation = rmat_to_quat_msg(t_group_cam.rmat());

    for (size_t i = 0; i < 5; ++i)
    {
      out.rune_type[i] = raw.rune_type[i];
    }
  }

  detections_pub_->publish(out);
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(RuneDetectorNode)
