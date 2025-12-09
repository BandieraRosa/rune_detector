#include "feature/rune_target/rune_target_active.h"

namespace rune_detector
{
RuneTargetActive::RuneTargetActive(const ContourConstPtr& contour,
                                   const std::vector<cv::Point2f>& corners)
    : RuneTarget(contour, corners)
{
  setActiveFlag(true);
}

RuneTargetActive::RuneTargetActive(const cv::Point2f& center,
                                   const std::vector<cv::Point2f>& corners)
{
  std::vector<cv::Point2f> temp_contour = corners;
  float width = rune_target_param.ACTIVE_DEFAULT_SIDE,
        height = rune_target_param.ACTIVE_DEFAULT_SIDE;
  ContourConstPtr contour = nullptr;
  if (temp_contour.size() < 3)
  {
    std::vector<cv::Point> contours_point(temp_contour.begin(), temp_contour.end());
    contour = ContourWrapper<int>::MakeContour(contours_point);
  }
  else
  {
    std::vector<cv::Point2f> hull;
    convexHull(temp_contour, hull);
    std::vector<cv::Point> contours_point(hull.begin(), hull.end());
    contour = ContourWrapper<int>::MakeContour(contours_point);
  }
  setActiveFlag(true);
  auto& image_info = getImageCache();
  image_info.setCenter(center);
  image_info.setWidth(width);
  image_info.setHeight(height);
  image_info.setCorners(corners);
  image_info.setContours(std::vector<ContourConstPtr>{contour});
}

static inline bool is_hierarchy_active_target(const std::vector<cv::Vec4i>& hierarchy,
                                              size_t idx)
{
  return hierarchy[idx][3] == -1 && hierarchy[idx][2] != -1;
}

void RuneTargetActive::Find(
    std::vector<FeatureNodePtr>& targets, const std::vector<ContourConstPtr>& contours,
    const std::vector<cv::Vec4i>& hierarchy, const std::unordered_set<size_t>& mask,
    std::unordered_map<FeatureNodeConstPtr, std::unordered_set<size_t>>&
        used_contour_idxs)
{
  for (size_t i = 0; i < contours.size(); i++)
  {
    if (mask.find(i) != mask.end())
    {
      continue;
    }
    if (is_hierarchy_active_target(hierarchy, i))
    {
      std::unordered_set<size_t> temp_used_contour_idxs{};
      auto p_target =
          RuneTargetActive::MakeFeature(contours, hierarchy, i, temp_used_contour_idxs);
      if (p_target)
      {
        targets.push_back(p_target);
        used_contour_idxs[p_target] = temp_used_contour_idxs;
      }
    }
  }
}

auto RuneTargetActive::GetPnpPoints() const
    -> std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>, std::vector<float>>
{
  return make_tuple(std::vector<cv::Point2f>{getImageCache().getCenter()},
                    std::vector<cv::Point3f>{cv::Point3f(0, 0, 0)},
                    std::vector<float>{1.0});
}

static inline bool check_ellipse(const ContourConstPtr& contour)
{
  if (contour->Points().size() < 6)
  {
    return false;
  }
  float contour_area = static_cast<float>(contour->Area());
  if (contour_area < rune_target_param.ACTIVE_MIN_AREA ||
      contour_area > rune_target_param.ACTIVE_MAX_AREA)
  {
    return false;
  }
  auto fit_ellipse = contour->FittedEllipse();
  float width = std::max(fit_ellipse.size.width, fit_ellipse.size.height);
  float height = std::min(fit_ellipse.size.width, fit_ellipse.size.height);
  float side_ratio = width / height;
  if (side_ratio > rune_target_param.ACTIVE_MAX_SIDE_RATIO ||
      side_ratio < rune_target_param.ACTIVE_MIN_SIDE_RATIO)
  {
    return false;
  }
  float fit_ellipse_area = static_cast<float>(width * height * CV_PI / 4);
  float area_ratio = contour_area / fit_ellipse_area;
  if (area_ratio > rune_target_param.ACTIVE_MAX_AREA_RATIO ||
      area_ratio < rune_target_param.ACTIVE_MIN_AREA_RATIO)
  {
    return false;
  }
  float perimeter = static_cast<float>(contour->Perimeter());
  float fit_ellipse_perimeter =
      CV_PI * (3 * (width + height) - sqrt((3 * width + height) * (width + 3 * height)));
  float perimeter_ratio = perimeter / fit_ellipse_perimeter;
  if (perimeter_ratio > rune_target_param.ACTIVE_MAX_PERI_RATIO ||
      perimeter_ratio < rune_target_param.ACTIVE_MIN_PERI_RATIO)
  {
    return false;
  }
  const auto& hull_contour = contour->ConvexHull();
  float hull_area = static_cast<float>(contourArea(hull_contour));
  float hull_area_ratio = hull_area / contour_area;
  if (hull_area_ratio > rune_target_param.ACTIVE_MAX_CONVEX_AREA_RATIO)
  {
    return false;
  }
  float hull_perimeter = static_cast<float>(arcLength(hull_contour, true));
  float hull_perimeter_ratio = hull_perimeter / perimeter;
  hull_perimeter_ratio =
      hull_perimeter_ratio > 1 ? hull_perimeter_ratio : 1 / hull_perimeter_ratio;
  if (hull_perimeter_ratio > rune_target_param.ACTIVE_MAX_CONVEX_PERI_RATIO)
  {
    return false;
  }
  return true;
}

static inline bool check_concentricity(const std::vector<ContourConstPtr>& contours,
                                       const std::vector<int>& all_sub_idx,
                                       double contour_area)
{
  if (all_sub_idx.empty())
  {
    return false;
  }
  std::unordered_map<size_t, float> area_map;
  for (auto sub_idx : all_sub_idx)
  {
    area_map[sub_idx] = static_cast<float>(contours[sub_idx]->Area());
  }
  auto [max_area_idx, max_sub_area] =
      *max_element(area_map.begin(), area_map.end(), [](const auto& lhs, const auto& rhs)
                   { return lhs.second < rhs.second; });
  if (max_sub_area > contour_area)
  {
    throw std::runtime_error("sub_contour_area > outer_area");
  }
  if (max_sub_area / contour_area < rune_target_param.ACTIVE_MIN_AREA_RATIO_SUB)
  {
    return false;
  }
  return check_ellipse(contours[max_area_idx]);
}

static inline bool check_ten_ring(const std::vector<ContourConstPtr>& contours,
                                  const std::vector<int>& all_sub_idx,
                                  double contour_area)
{
  float total_area = 0;
  for (auto sub_idx : all_sub_idx)
  {
    total_area += static_cast<float>(contours[sub_idx]->Area());
  }
  return total_area / contour_area <=
         rune_target_param.ACTIVE_MAX_AREA_RATIO_SUB_TEN_RING;
}

RuneTargetActivePtr RuneTargetActive::MakeFeature(
    const std::vector<ContourConstPtr>& contours, const std::vector<cv::Vec4i>& hierarchy,
    size_t idx, std::unordered_set<size_t>& used_contour_idxs)
{
  const auto& contour_outer = contours[idx];
  if (contour_outer->Points().size() < 6)
  {
    return nullptr;
  }
  float contour_area = static_cast<float>(contour_outer->Area());
  if (!check_ellipse(contour_outer))
  {
    return nullptr;
  }
  std::vector<int> all_sub_idx;
  get_all_sub_contours_idx(hierarchy, static_cast<int>(idx), all_sub_idx);
  if (contour_area < 100)
  {
    return nullptr;
  }
  if (!check_concentricity(contours, all_sub_idx, contour_area) &&
      !check_ten_ring(contours, all_sub_idx, contour_area))
  {
    return nullptr;
  }
  used_contour_idxs.insert(idx);
  used_contour_idxs.insert(all_sub_idx.begin(), all_sub_idx.end());
  auto fit_ellipse = contour_outer->FittedEllipse();
  std::vector<cv::Point2f> corners{fit_ellipse.center};
  return make_shared<RuneTargetActive>(contour_outer, corners);
}

void RuneTargetActive::DrawFeature(cv::Mat& image,
                                   const DrawConfigConstPtr& /*config*/) const
{
  auto& image_info = getImageCache();
  auto draw_circle = [&]()
  {
    if (!image_info.isSetCorners() || !image_info.isSetCenter())
    {
      return false;
    }
    auto& center = image_info.getCenter();
    auto radius = image_info.isSetHeight() && image_info.isSetWidth()
                      ? std::min(image_info.getHeight(), image_info.getWidth()) / 2.0f
                      : static_cast<float>(rune_target_draw_param.active.point_radius);
    auto& circle_color = rune_target_draw_param.active.color;
    auto circle_thickness = rune_target_draw_param.active.thickness;
    circle(image, center, static_cast<int>(radius), circle_color, circle_thickness);
    if (radius > 30)
    {
      circle(image, center, static_cast<int>(radius * 0.5f), circle_color,
             circle_thickness);
    }
    if (radius > 50)
    {
      circle(image, center, static_cast<int>(radius * 0.25f), circle_color,
             circle_thickness);
    }
    if (radius > 100)
    {
      circle(image, center, static_cast<int>(radius * 0.125f), circle_color,
             circle_thickness);
    }
    return true;
  };
  auto draw_ellipse = [&]()
  {
    if (!image_info.isSetContours() || image_info.getContours().empty() ||
        image_info.getContours().front()->Points().size() < 6)
    {
      return false;
    }
    auto fit_ellipse = image_info.getContours().front()->FittedEllipse();
    auto& circle_color = rune_target_draw_param.active.color;
    auto circle_thickness = rune_target_draw_param.active.thickness;
    ellipse(image, fit_ellipse, circle_color, circle_thickness);
    circle(image, fit_ellipse.center, 2, circle_color, -1);
    return true;
  };
  if (!draw_ellipse())
  {
    draw_circle();
  }
}

RuneTargetActivePtr RuneTargetActive::MakeFeature(const PoseNode& target_to_cam,
                                                  const cv::Mat& k, const cv::Mat& d)
{
  std::vector<cv::Point3f> corners_3d{{0, -rune_target_param.RADIUS, 0},
                                      {rune_target_param.RADIUS, 0, 0},
                                      {0, rune_target_param.RADIUS, 0},
                                      {-rune_target_param.RADIUS, 0, 0}};
  std::vector<cv::Point2f> corners_2d, temp_rune_center;
  cv::Vec3d rvec, tvec;
  pose_to_open_cv(target_to_cam, rvec, tvec);
  projectPoints(corners_3d, rvec, tvec, k, d, corners_2d);
  projectPoints(std::vector<cv::Point3f>{cv::Point3f(0, 0, 0)}, rvec, tvec, k, d,
                temp_rune_center);
  auto result_ptr = make_shared<RuneTargetActive>(temp_rune_center[0], corners_2d);
  if (result_ptr)
  {
    result_ptr->getPoseCache().getPoseNodes()[CoordFrame::CAMERA] = target_to_cam;
  }
  return result_ptr;
}

}  // namespace rune_detector