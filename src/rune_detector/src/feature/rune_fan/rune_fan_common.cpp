#include "feature/rune_fan/rune_fan.h"
#include "feature/rune_fan/rune_fan_active.h"
#include "feature/rune_fan/rune_fan_inactive.h"
#include "feature/rune_fan/rune_fan_param.h"

namespace rune_detector
{
void RuneFan::FindActiveFans(
    std::vector<FeatureNodePtr>& fans, const std::vector<ContourConstPtr>& contours,
    const std::vector<cv::Vec4i>& hierarchy, const std::unordered_set<size_t>& mask,
    std::unordered_map<FeatureNodeConstPtr, std::unordered_set<size_t>>&
        used_contour_idxs)
{
  RuneFanActive::Find(fans, contours, hierarchy, mask, used_contour_idxs);
}

void RuneFan::FindInactiveFans(
    std::vector<FeatureNodePtr>& fans, const std::vector<ContourConstPtr>& contours,
    const std::vector<cv::Vec4i>& hierarchy, const std::unordered_set<size_t>& mask,
    std::unordered_map<FeatureNodeConstPtr, std::unordered_set<size_t>>&
        used_contour_idxs,
    const std::vector<FeatureNodeConstPtr>& inactive_targets)
{
  RuneFanInactive::Find(fans, contours, hierarchy, mask, used_contour_idxs,
                        inactive_targets);
}

void RuneFan::FindIncompleteActiveFans(
    std::vector<FeatureNodePtr>& fans, const std::vector<ContourConstPtr>& contours,
    const std::vector<cv::Vec4i>& hierarchy, const std::unordered_set<size_t>& mask,
    const cv::Point2f& rotate_center,
    std::unordered_map<FeatureNodeConstPtr, std::unordered_set<size_t>>&
        used_contour_idxs)
{
  RuneFanActive::FindIncomplete(fans, contours, hierarchy, mask, rotate_center,
                                used_contour_idxs);
}

auto RuneFan::GetPnpPoints() const
    -> std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>, std::vector<float>>
{
  // 空实现,不允许被调用
  throw std::runtime_error(
      "getPnpPoints() is not implemented in RuneFan class. Please override this method "
      "in derived classes.");
  std::vector<cv::Point2f> points_2d{};
  std::vector<cv::Point3f> points_3d{};
  std::vector<float> weights{};
  return make_tuple(points_2d, points_3d, weights);
}

auto RuneFan::GetRelativePnpPoints() const
    -> std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>, std::vector<float>>
{
  auto [points_2d, points_3d, weights] = this->GetPnpPoints();
  std::vector<cv::Point3f> relative_points_3d(points_3d.size());
  for (int i = 0; i < points_3d.size(); i++)
  {
    cv::Matx31d points_3d_mat(points_3d[i].x, points_3d[i].y, points_3d[i].z);
    cv::Matx31d relative_points_3d_mat{};
    if (this->getActiveFlag())
    {
      relative_points_3d_mat = rune_fan_param.ACTIVE_ROTATION * points_3d_mat +
                               rune_fan_param.ACTIVE_TRANSLATION;
    }
    else
    {
      relative_points_3d_mat = rune_fan_param.INACTIVE_ROTATION * points_3d_mat +
                               rune_fan_param.INACTIVE_TRANSLATION;
    }
    relative_points_3d[i] = cv::Point3f(static_cast<float>(relative_points_3d_mat(0)),
                                        static_cast<float>(relative_points_3d_mat(1)),
                                        static_cast<float>(relative_points_3d_mat(2)));
  }
  return make_tuple(points_2d, relative_points_3d, weights);
}

static bool check_points(const std::vector<cv::Point2d>& points)
{
  static const float MAX_X = 5000;
  static const float MAX_Y = 5000;

  for (const auto& point : points)
  {
    if (abs(point.x) > MAX_X || abs(point.y) > MAX_Y)
    {
      return false;
    }
  }
  return true;
}

RuneFanPtr RuneFan::MakeFeature(const PoseNode& fan_to_cam, bool is_active,
                                const cv::Mat& k, const cv::Mat& d)
{
  RuneFanPtr result_ptr;

  cv::Vec3d rvec, tvec;
  pose_to_opencv(fan_to_cam, rvec, tvec);

  if (is_active)
  {
    std::vector<cv::Point3d> top_hump_corners{};
    std::vector<cv::Point3d> bottom_center_hump_corners{};
    std::vector<cv::Point3d> side_hump_corners{};
    std::vector<cv::Point3d> bottom_side_hump_corners{};

    top_hump_corners = rune_fan_param.ACTIVE_TOP_3D;
    bottom_center_hump_corners = rune_fan_param.ACTIVE_BOTTOM_CENTER_3D;
    side_hump_corners = rune_fan_param.ACTIVE_SIDE_3D;
    bottom_side_hump_corners = rune_fan_param.ACTIVE_BOTTOM_SIDE_3D;

    std::vector<cv::Point2d> top_hump_corners_2d{};
    std::vector<cv::Point2d> bottom_center_hump_corners_2d{};
    std::vector<cv::Point2d> side_hump_corners_2d{};
    std::vector<cv::Point2d> bottom_side_hump_corners_2d{};

    projectPoints(top_hump_corners, rvec, tvec, k, d, top_hump_corners_2d);
    projectPoints(bottom_center_hump_corners, rvec, tvec, k, d,
                  bottom_center_hump_corners_2d);
    projectPoints(side_hump_corners, rvec, tvec, k, d, side_hump_corners_2d);
    projectPoints(bottom_side_hump_corners, rvec, tvec, k, d,
                  bottom_side_hump_corners_2d);

    if (check_points(top_hump_corners_2d) == false ||
        check_points(bottom_center_hump_corners_2d) == false ||
        check_points(side_hump_corners_2d) == false ||
        check_points(bottom_side_hump_corners_2d) == false)
    {
      return nullptr;
    }

    result_ptr =
        RuneFanActive::MakeFeature(top_hump_corners_2d, bottom_center_hump_corners_2d,
                                   side_hump_corners_2d, bottom_side_hump_corners_2d);
  }
  else
  {
    std::vector<cv::Point3d> corners = rune_fan_param.INACTIVE_3D;
    std::vector<cv::Point2d> corners_2d{};

    projectPoints(corners, rvec, tvec, k, d, corners_2d);

    if (check_points(corners_2d) == false)
    {
      return nullptr;
    }

    result_ptr = RuneFanInactive::MakeFeature(corners_2d[0], corners_2d[1], corners_2d[2],
                                              corners_2d[3]);
  }
  if (result_ptr)
  {
    auto& temp_ptr = result_ptr;
    temp_ptr->getPoseCache().getPoseNodes()[CoordFrame::CAMERA] = PoseNode(fan_to_cam);
  }
  return result_ptr;
}

}  // namespace rune_detector