#include "feature/rune_target/rune_target.h"

namespace rune_detector
{
RuneTarget::RuneTarget(const ContourConstPtr& contour,
                       const std::vector<cv::Point2f>& corners)
{
  if (contour->Points().size() < 3)
  {
    throw std::runtime_error(
        "Contour points are less than 3, cannot construct RuneTarget.");
  }
  auto rotate_rect = contour->MinAreaRect();
  auto width = std::max(rotate_rect.size.width, rotate_rect.size.height);
  auto height = std::min(rotate_rect.size.width, rotate_rect.size.height);
  cv::Point2f center{};
  center =
      contour->Points().size() > 6 ? contour->FittedEllipse().center : contour->Center();
  auto& image_info = getImageCache();
  image_info.setContours(std::vector<ContourConstPtr>{contour});
  image_info.setCorners(corners);
  image_info.setCenter(center);
  image_info.setWidth(width);
  image_info.setHeight(height);
}

void RuneTarget::FindActiveTargets(
    std::vector<FeatureNodePtr>& targets, const std::vector<ContourConstPtr>& contours,
    const std::vector<cv::Vec4i>& hierarchy, const std::unordered_set<size_t>& mask,
    std::unordered_map<FeatureNodeConstPtr, std::unordered_set<size_t>>&
        used_contour_idxs)
{
  RuneTargetActive::Find(targets, contours, hierarchy, mask, used_contour_idxs);
}

void RuneTarget::FindInactiveTargets(
    std::vector<FeatureNodePtr>& targets, const std::vector<ContourConstPtr>& contours,
    const std::vector<cv::Vec4i>& hierarchy, const std::unordered_set<size_t>& mask,
    std::unordered_map<FeatureNodeConstPtr, std::unordered_set<size_t>>&
        used_contour_idxs)
{
  RuneTargetInactive::Find(targets, contours, hierarchy, mask, used_contour_idxs);
}

auto RuneTarget::GetPnpPoints() const
    -> std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>, std::vector<float>>
{
  return make_tuple(std::vector<cv::Point2f>{}, std::vector<cv::Point3f>{},
                    std::vector<float>{});
}

auto RuneTarget::GetRelativePnpPoints() const
    -> std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>, std::vector<float>>
{
  auto [points_2d, points_3d, weights] = this->GetPnpPoints();
  std::vector<cv::Point3f> relative_points_3d(points_3d.size());
  for (size_t i = 0; i < points_3d.size(); i++)
  {
    cv::Matx31d points_3d_mat(points_3d[i].x, points_3d[i].y, points_3d[i].z);
    cv::Matx31d relative_points_3d_mat =
        rune_target_param.ROTATION * points_3d_mat + rune_target_param.TRANSLATION;
    relative_points_3d[i] = cv::Point3f(static_cast<float>(relative_points_3d_mat(0)),
                                        static_cast<float>(relative_points_3d_mat(1)),
                                        static_cast<float>(relative_points_3d_mat(2)));
  }
  return make_tuple(points_2d, relative_points_3d, weights);
}

}  // namespace rune_detector