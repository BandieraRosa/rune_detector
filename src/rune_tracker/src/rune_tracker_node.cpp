#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "rune_interfaces/msg/rune_detections.hpp"
#include "rune_interfaces/msg/rune_target.hpp"

// rm_vision_core
#include "vc/feature/rune_center_param.h"
#include "vc/feature/rune_data_converter.h"
#include "vc/feature/rune_filter_ekf.h"
#include "vc/feature/rune_target_param.h"
#include "vc/math/pose_node.hpp"

using rune_interfaces::msg::RuneDetections;
using rune_interfaces::msg::RuneTarget;

namespace
{
int64_t stamp_to_ns(const builtin_interfaces::msg::Time& t)
{
  return static_cast<int64_t>(t.sec) * 1000000000LL + static_cast<int64_t>(t.nanosec);
}

cv::Matx33d quat_to_rmat(const geometry_msgs::msg::Quaternion& qmsg)
{
  tf2::Quaternion q;
  tf2::fromMsg(qmsg, q);
  tf2::Matrix3x3 m(q);
  return cv::Matx33d(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0],
                     m[2][1], m[2][2]);
}

geometry_msgs::msg::Quaternion rmat_to_quat(const cv::Matx33d& R)
{
  tf2::Matrix3x3 m(R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1),
                   R(2, 2));
  tf2::Quaternion q;
  m.getRotation(q);
  return tf2::toMsg(q);
}

uint8_t pick_target_index(const RuneDetections& msg, const std::string& policy)
{
  auto want = RuneDetections::UNKNOWN;
  if (policy == "pending")
    want = RuneDetections::PENDING_STRUCK;
  else if (policy == "struck")
    want = RuneDetections::STRUCK;

  if (want != RuneDetections::UNKNOWN)
  {
    for (uint8_t i = 0; i < 5; ++i)
    {
      if (msg.rune_type[i] == want) return i;
    }
  }

  if (policy == "any")
  {
    for (uint8_t i = 0; i < 5; ++i)
    {
      if (msg.rune_type[i] != RuneDetections::UNKNOWN) return i;
    }
  }

  // fallback
  return 0;
}
}  // namespace

class RuneTrackerNode : public rclcpp::Node
{
 public:
  RuneTrackerNode(const rclcpp::NodeOptions& options)
      : Node("rune_tracker_node", options),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_)
  {
    detections_topic_ =
        this->declare_parameter<std::string>("detections_topic", "/rune/detections");
    target_topic_ = this->declare_parameter<std::string>("target_topic", "/rune/target");
    output_frame_ = this->declare_parameter<std::string>("output_frame", "gimbal_odom");
    tf_timeout_ms_ = this->declare_parameter<int>("tf_timeout_ms", 30);
    predict_dt_ = this->declare_parameter<double>("predict_dt", 0.12);
    aim_policy_ = this->declare_parameter<std::string>("aim_policy", "pending");

    target_pub_ =
        this->create_publisher<RuneTarget>(target_topic_, rclcpp::SystemDefaultsQoS());

    detections_sub_ = this->create_subscription<RuneDetections>(
        detections_topic_, rclcpp::SensorDataQoS(),
        std::bind(&RuneTrackerNode::DetectionsCallback, this, std::placeholders::_1));

    converter_ = DataConverter::make_converter();
    ekf_ = RuneFilterEKF_CV::make_filter(RuneFilterDataType::XYZ);

    RCLCPP_INFO(this->get_logger(), "rune_tracker_node started.");
    RCLCPP_INFO(this->get_logger(),
                "detections_topic=%s target_topic=%s output_frame=%s predict_dt=%.3f",
                detections_topic_.c_str(), target_topic_.c_str(), output_frame_.c_str(),
                predict_dt_);
  }

 private:
  void DetectionsCallback(const RuneDetections::SharedPtr msg)
  {
    RuneTarget out;
    out.header = msg->header;
    out.header.frame_id = output_frame_;
    out.tracking = false;
    out.target_index = 0;
    out.target_type = RuneTarget::UNKNOWN;
    out.predict_dt = static_cast<float>(predict_dt_);

    if (!msg->valid)
    {
      // no observation: we can still publish predicted state if filter is initialized
      publish_if_possible(false, out);
      return;
    }

    // 1) Transform group pose from camera frame to output_frame using tf2
    geometry_msgs::msg::PoseStamped in_pose, out_pose;
    in_pose.header = msg->header;
    in_pose.pose = msg->group_pose;

    try
    {
      // transform with timeout
      out_pose = tf_buffer_.transform(in_pose, output_frame_,
                                      tf2::durationFromSec(tf_timeout_ms_ / 1000.0));
    }
    catch (const std::exception& e)
    {
      RCLCPP_WARN(this->get_logger(), "tf transform failed: %s", e.what());
      return;
    }

    // 2) Convert ROS pose (m) -> PoseNode (mm) for rm_vision_core math/model params
    cv::Vec3d tvec_mm(out_pose.pose.position.x * 1000.0,
                      out_pose.pose.position.y * 1000.0,
                      out_pose.pose.position.z * 1000.0);

    cv::Matx33d R = quat_to_rmat(out_pose.pose.orientation);
    cv::Vec3d rvec;
    cv::Rodrigues(R, rvec);

    bool is_gimbal_lock = false;
    // filter form: [x y z yaw pitch roll] (deg)
    cv::Matx61f raw_pos_f = converter_->toFilterForm(
        cv::Vec3f(static_cast<float>(tvec_mm[0]), static_cast<float>(tvec_mm[1]),
                  static_cast<float>(tvec_mm[2])),
        cv::Vec3f(static_cast<float>(rvec[0]), static_cast<float>(rvec[1]),
                  static_cast<float>(rvec[2])),
        is_gimbal_lock, false);

    cv::Matx61d raw_pos_d;
    for (int i = 0; i < 6; ++i) raw_pos_d(i) = static_cast<double>(raw_pos_f(i));

    const int64_t tick_ns = stamp_to_ns(msg->header.stamp);

    // 3) EKF update (observation)
    RuneFilterStrategy::FilterInput fi;
    fi.raw_pos = raw_pos_d;
    fi.tick = tick_ns;
    fi.cam_to_gyro = PoseNode();  // not used in EKF itself
    fi.is_observation = true;

    auto fo = ekf_->filter(fi);
    (void)fo;

    if (!ekf_->isValid())
    {
      return;
    }

    // 4) Predict ahead for latency compensation
    cv::Matx61d pred_pos = ekf_->predictAhead(predict_dt_);

    // 5) Convert predicted filter form -> PoseNode group pose (mm)
    cv::Matx61f pred_pos_f;
    for (int i = 0; i < 6; ++i) pred_pos_f(i) = static_cast<float>(pred_pos(i));

    auto [pred_tvec_mm, pred_rvec] = DataConverter::toTvecAndRvec(pred_pos_f);

    PoseNode group_pose_mm(pred_rvec, pred_tvec_mm);

    // 6) Choose target index by rune_type policy
    uint8_t idx = pick_target_index(*msg, aim_policy_);
    uint8_t rtype = msg->rune_type[idx];

    // 7) Compute rune pose for that sector (same logic as rm_vision_core
    // getRunes:setTempRunePnpData)
    PoseNode rune_pose = group_pose_mm;
    rune_pose.rotate_z(72.0 * static_cast<double>(idx));  // DEG

    // Apply center translation compensation (mm)
    const cv::Vec3d center_off(rune_center_param.TRANSLATION(0),
                               rune_center_param.TRANSLATION(1),
                               rune_center_param.TRANSLATION(2));
    rune_pose.tvec(rune_pose.tvec() + rune_pose.rmat() * center_off);

    // 8) Aim point = target center in output_frame
    const cv::Vec3d target_off(rune_target_param.TRANSLATION(0),
                               rune_target_param.TRANSLATION(1),
                               rune_target_param.TRANSLATION(2));
    cv::Vec3d target_pos_mm = rune_pose.tvec() + rune_pose.rmat() * target_off;

    // Target orientation (optional): R_cam_target = R_cam_rune * R_rune_target
    cv::Matx33d R_target = rune_pose.rmat() * rune_target_param.ROTATION;

    out.tracking = true;
    out.target_index = idx;
    out.target_type = rtype;

    out.target_pose.position.x = target_pos_mm[0] / 1000.0;
    out.target_pose.position.y = target_pos_mm[1] / 1000.0;
    out.target_pose.position.z = target_pos_mm[2] / 1000.0;
    out.target_pose.orientation = rmat_to_quat(R_target);

    target_pub_->publish(out);
  }

  void publish_if_possible(bool, RuneTarget&) {}

 private:
  std::string detections_topic_;
  std::string target_topic_;
  std::string output_frame_;
  int tf_timeout_ms_{30};
  double predict_dt_{0.12};
  std::string aim_policy_{"pending"};

  rclcpp::Subscription<RuneDetections>::SharedPtr detections_sub_;
  rclcpp::Publisher<RuneTarget>::SharedPtr target_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  DataConverter_ptr converter_;
  std::shared_ptr<RuneFilterEKF_CV> ekf_;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(RuneTrackerNode)
