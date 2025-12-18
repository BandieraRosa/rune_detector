#include "feature/rune_combo/rune_combo.h"

#include <opencv2/core.hpp>
#include <stdexcept>

#include "common/geom_utils.hpp"
#include "feature/rune_center/rune_center.h"
#include "feature/rune_center/rune_center_param.h"
#include "feature/rune_fan/rune_fan_active.h"
#include "feature/rune_fan/rune_fan_inactive.h"
#include "feature/rune_fan/rune_fan_param.h"
#include "feature/rune_target/rune_target_active.h"
#include "feature/rune_target/rune_target_inactive.h"
#include "feature/rune_target/rune_target_param.h"

namespace rune_detector
{

RuneCombo_ptr RuneCombo::MakeFeature(const PoseNode& pnp_data, const RuneType& type,
                                     const GyroData& gyro_data, int64 tick,
                                     const cv::Mat& k, const cv::Mat& d,
                                     const cv::Matx33d& cam2joint_rmat,
                                     const cv::Vec3d& cam2joint_tvec)
{
  cv::Vec3d pnp_data_rvec, pnp_data_tvec;
  pose_to_opencv(pnp_data, pnp_data_rvec, pnp_data_tvec);
  cv::Matx33d pnp_data_rmat;
  cv::Rodrigues(pnp_data_rvec, pnp_data_rmat);

  //------------------ 获取靶心特征 -------------------
  PoseNode target_pnp_data{};
  cv::Vec3d target_pnp_data_rvec, target_pnp_data_tvec;
  FeatureNodePtr target = nullptr;
  cv::Matx31d target_tmp;

  if (type == RuneType::STRUCK)
  {
    target_tmp = pnp_data_tvec + pnp_data_rmat * rune_target_param.TRANSLATION;
    target_pnp_data_tvec = {target_tmp(0, 0), target_tmp(1, 0), target_tmp(2, 0)};
    target_pnp_data_rvec = rune_target_param.ROTATION * pnp_data_rvec;
    target_pnp_data = opencv_to_pose(target_pnp_data_rvec, target_pnp_data_tvec);
    target = RuneTargetActive::MakeFeature(target_pnp_data, k, d);
  }
  else if (type == RuneType::UNSTRUCK || type == RuneType::PENDING_STRUCK)
  {
    cv::Matx31d tmp = pnp_data_tvec + pnp_data_rmat * rune_target_param.TRANSLATION;
    target_pnp_data_tvec = {tmp(0, 0), tmp(1, 0), tmp(2, 0)};
    target_pnp_data_rvec = rune_target_param.ROTATION * pnp_data_rvec;
    target_pnp_data = opencv_to_pose(target_pnp_data_rvec, target_pnp_data_tvec);
    target = RuneTargetInactive::MakeFeature(target_pnp_data, k, d);
  }
  else
  {
    throw std::runtime_error(
        "The type is not STRUCK or PENDING_STRUCK or PENDING_UNSTRUCK");
  }
  //------------------ 获取神符中心特征 -------------------
  PoseNode center_pnp_data{};
  cv::Vec3d center_pnp_data_rvec, center_pnp_data_tvec;
  cv::Matx31d center_tmp = pnp_data_tvec + pnp_data_rmat * rune_center_param.TRANSLATION;
  center_pnp_data_tvec = {center_tmp(0, 0), center_tmp(1, 0), center_tmp(2, 0)};
  center_pnp_data_rvec = rune_center_param.ROTATION * pnp_data_rvec;
  center_pnp_data = opencv_to_pose(center_pnp_data_rvec, center_pnp_data_tvec);
  FeatureNodePtr center = RuneCenter::MakeFeature(center_pnp_data, k, d);
  //------------------ 获取扇叶特征 -------------------
  PoseNode fan_pnp_data{};
  cv::Vec3d fan_pnp_data_rvec, fan_pnp_data_tvec;
  FeatureNodePtr fan = nullptr;
  cv::Matx31d fan_tmp;
  if (type == RuneType::STRUCK)
  {
    fan_tmp = pnp_data_tvec + pnp_data_rmat * rune_fan_param.ACTIVE_TRANSLATION;
    fan_pnp_data_tvec = {fan_tmp(0, 0), fan_tmp(1, 0), fan_tmp(2, 0)};
    fan_pnp_data_rvec = rune_fan_param.ACTIVE_ROTATION * pnp_data_rvec;
    fan = RuneFan::MakeFeature(fan_pnp_data, true, k, d);
  }
  else if (type == RuneType::PENDING_STRUCK || type == RuneType::UNSTRUCK)
  {
    fan_tmp = pnp_data_tvec + pnp_data_rmat * rune_fan_param.INACTIVE_TRANSLATION;
    fan_pnp_data_tvec = {fan_tmp(0, 0), fan_tmp(1, 0), fan_tmp(2, 0)};
    fan_pnp_data_rvec = rune_fan_param.INACTIVE_ROTATION * pnp_data_rvec;
    fan = RuneFan::MakeFeature(fan_pnp_data, false, k, d);
  }
  else
  {
    throw std::runtime_error(
        "The type is not STRUCK or PENDING_STRUCK or PENDING_UNSTRUCK");
  }

  return make_shared<RuneCombo>(target, center, fan, pnp_data, type, gyro_data, tick, k,
                                d, cam2joint_rmat, cam2joint_tvec);
}

/**
 * @brief 通过三个特征计算神符的角点
 *
 * @param p_target 靶心特征
 * @param p_center 中心特征
 * @param p_fan 扇叶特征
 */
static inline std::vector<cv::Point2f> get_rune_corners(
    const FeatureNodeConstPtr& p_target, const cv::Mat& k, const cv::Mat& d)
{
  // 获取击打区域（靶心）的半径
  float radius = rune_target_param.RADIUS;
  std::vector<cv::Point3f> corners_3d{};
  corners_3d.emplace_back(0, -radius, 0);
  corners_3d.emplace_back(radius, 0, 0);
  corners_3d.emplace_back(0, radius, 0);
  corners_3d.emplace_back(-radius, 0, 0);

  // 进行重投影
  auto p_target_pose = p_target->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA);
  cv::Vec3d p_target_pose_rvec, p_target_pose_tvec;
  pose_to_opencv(p_target_pose, p_target_pose_rvec, p_target_pose_tvec);

  std::vector<cv::Point2f> corners_2d{};
  projectPoints(corners_3d, p_target_pose_rvec, p_target_pose_tvec, k, d, corners_2d);
  return corners_2d;
}

RuneCombo::RuneCombo(const FeatureNodePtr& p_target, const FeatureNodePtr& p_center,
                     const FeatureNodePtr& p_fan, const PoseNode& rune_to_cam,
                     const RuneType& type, const GyroData& gyro_data, int64 tick,
                     const cv::Mat& k, const cv::Mat& d,
                     const cv::Matx33d& cam2joint_rmat, const cv::Vec3d& cam2joint_tvec)
{
  // ----------------获取轮廓的最小外接矩形--------------------
  std::vector<cv::Point> temp_contour{};
  if (p_target != nullptr)
  {
    for (const auto& contour : p_target->getImageCache().getContours())
    {
      temp_contour.insert(temp_contour.end(), contour->Points().begin(),
                          contour->Points().end());
    }
  }
  if (p_center != nullptr)
  {
    for (const auto& contour : p_center->getImageCache().getContours())
    {
      temp_contour.insert(temp_contour.end(), contour->Points().begin(),
                          contour->Points().end());
    }
  }
  if (p_fan != nullptr)
  {
    for (const auto& contour : p_fan->getImageCache().getContours())
    {
      temp_contour.insert(temp_contour.end(), contour->Points().begin(),
                          contour->Points().end());
    }
  }
  cv::RotatedRect rect = minAreaRect(temp_contour);
  auto width = std::max(rect.size.width, rect.size.height);
  auto height = std::min(rect.size.width, rect.size.height);
  auto center = p_target->getImageCache().getCenter();  // 神符组合体的中心就是靶心的中心

  auto& image_info = getImageCache();
  image_info.setWidth(width);
  image_info.setHeight(height);
  image_info.setCenter(center);

  cv::Vec3d cam2joint_rvec;
  cv::Rodrigues(cam2joint_rmat, cam2joint_rvec);

  PoseNode cam_to_joint(opencv_to_pose(cam2joint_rvec, cam2joint_tvec));
  PoseNode rune_to_joint = compose_poses(rune_to_cam, cam_to_joint);
  cv::Matx33f joint_to_gyro_r =
      euler2mat(deg2rad(gyro_data.rotation.yaw), EulerAxis::Y) *
      euler2mat(deg2rad(-1 * gyro_data.rotation.pitch), EulerAxis::X);
  PoseNode joint_to_gyro(opencv_to_pose(cv::Vec3f(cam2joint_rvec), cv::Vec3f{0, 0, 0}));

  PoseNode rune_to_gyro = compose_poses(rune_to_joint, joint_to_gyro);

  auto& pose_info = getPoseCache();
  pose_info.setGyroData(gyro_data);
  pose_info.getPoseNodes()[CoordFrame::CAMERA] = rune_to_cam;   // 相机坐标系
  pose_info.getPoseNodes()[CoordFrame::JOINT] = rune_to_joint;  // 转轴坐标系
  pose_info.getPoseNodes()[CoordFrame::GYRO] = rune_to_gyro;    // 陀螺仪坐标系

  // ------------- 更新神符角度 -------------
  // 获取旋转矩阵
  cv::Matx33d r;
  cv::Vec3d rune_to_gyro_rvec, rune_to_gyro_tvec;
  pose_to_opencv(rune_to_gyro, rune_to_gyro_rvec, rune_to_gyro_tvec);
  cv::Rodrigues(rune_to_gyro_rvec, r);
  float r11 = r(0, 0), r21 = r(1, 0);
  float roll = rad2deg(atan2(r21, r11));

  // --------------设置神符组合体的角点--------------------
  auto corners = get_rune_corners(p_target, k, d);
  image_info.setCorners(corners);
  // ---------- 设置组合体特征指针 ----------
  auto& child_features = getChildFeatures();
  child_features[FeatureNode::ChildFeatureType::rune_target_] = p_target;
  child_features[FeatureNode::ChildFeatureType::rune_center_] = p_center;
  child_features[FeatureNode::ChildFeatureType::rune_fan_] = p_fan;
  // ---------- 更新组合体类型信息 ----------
  setRuneType(type);
  setTick(tick);
}

}  // namespace rune_detector
