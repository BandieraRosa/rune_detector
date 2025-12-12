#include "feature/rune_fan/rune_fan_inactive.h"

#include "common/contour_wrapper.hpp"
#include "feature/rune_fan/rune_fan_param.h"

using namespace std;
using namespace cv;

namespace rune_detector
{

static inline bool is_hierarchy_inactive_fan(const vector<Vec4i>& hierarchy, size_t idx)
{
  return hierarchy[idx][3] == -1;
}
static inline double calculate_projection_ratio(const RotatedRect& rect1,
                                                const RotatedRect& rect2)
{
  const RotatedRect& large_rect = rect1.size.area() > rect2.size.area() ? rect1 : rect2;
  const RotatedRect& small_rect = rect1.size.area() > rect2.size.area() ? rect2 : rect1;

  Point2f large_points[4], small_points[4];
  large_rect.points(large_points);
  small_rect.points(small_points);

  struct Edge
  {
    Point2f start, end, dir;
    double len;
  };
  Edge edges[2];
  for (int i = 0; i < 2; ++i)
  {
    edges[i].start = large_points[i];
    edges[i].end = large_points[(i + 1) % 4];
    edges[i].dir = edges[i].end - edges[i].start;
    edges[i].len = norm(edges[i].dir);
    edges[i].dir /= edges[i].len;
  }

  vector<double> proj_x, proj_y;
  for (int i = 0; i < 4; ++i)
  {
    proj_x.push_back((small_points[i] - edges[0].start).dot(edges[0].dir));
    proj_y.push_back((small_points[i] - edges[1].start).dot(edges[1].dir));
  }

  auto range = [](const vector<double>& v)
  {
    return make_pair(*min_element(v.begin(), v.end()), *max_element(v.begin(), v.end()));
  };
  auto [x_min, x_max] = range(proj_x);
  auto [y_min, y_max] = range(proj_y);
  double x_len = edges[0].len, y_len = edges[1].len;
  double x_int = max(0.0, min(x_max, x_len) - max(x_min, 0.0));
  double y_int = max(0.0, min(y_max, y_len) - max(y_min, 0.0));
  return x_int / x_len > y_int / y_len ? (x_max - x_min) / x_len : y_int / y_len;
}

static inline void filter_fan_contours(const std::vector<ContourConstPtr>& in_contours,
                                       std::vector<ContourConstPtr>& out_contours,
                                       vector<FeatureNodePtr>& inactive_targets)
{
  if (in_contours.size() < 2 || inactive_targets.empty())
  {
    return;
  }

  FeatureNodePtr ref_target = nullptr;
  if (inactive_targets.size() > 1)  // 多个靶心，选择最近的那个
  {
    unordered_map<ContourConstPtr, double> contour_weights{};
    double area_sum = 0;
    for (auto& contour : in_contours)
    {
      area_sum += contour->Area();
    }

    if (area_sum == 0)
    {
      return;
    }

    for (auto& contour : in_contours)
    {
      contour_weights[contour] = contour->Area() / area_sum;
    }

    Point2f all_contours_center{0, 0};
    for (auto& contour : in_contours)
    {
      all_contours_center +=
          static_cast<Point2f>(contour->Center()) * contour_weights[contour];
    }

    auto near_target =
        *min_element(inactive_targets.begin(), inactive_targets.end(),
                     [&](const FeatureNodePtr& a, const FeatureNodePtr& b)
                     {
                       return norm(a->getImageCache().getCenter() - all_contours_center) <
                              norm(b->getImageCache().getCenter() - all_contours_center);
                     });
  }
  else
  {
    ref_target = inactive_targets.front();
  }

  auto far_contour = *max_element(in_contours.begin(), in_contours.end(),
                                  [&](const ContourConstPtr& a, const ContourConstPtr& b)
                                  {
                                    return norm(static_cast<Point2f>(a->Center()) -
                                                ref_target->getImageCache().getCenter()) <
                                           norm(static_cast<Point2f>(b->Center()) -
                                                ref_target->getImageCache().getCenter());
                                  });

  out_contours = in_contours;
  out_contours.erase(
      remove_if(out_contours.begin(), out_contours.end(),
                [&](const ContourConstPtr& contour) { return contour == far_contour; }),
      out_contours.end());
}

void RuneFanInactive::Find(
    std::vector<FeatureNodePtr>& fans, const std::vector<ContourConstPtr>& contours,
    const std::vector<cv::Vec4i>& hierarchy, const std::unordered_set<size_t>& mask,
    std::unordered_map<FeatureNodeConstPtr, std::unordered_set<size_t>>&
        used_contour_idxs,
    const std::vector<FeatureNodeConstPtr>& /*inactive_targets*/)
{
  if (contours.empty() || hierarchy.empty())
  {
    throw std::runtime_error(
        "The contours or hierarchy is empty. to find inactive fans.");
  }
  if (contours.size() != hierarchy.size())
  {
    throw std::runtime_error(
        "The contours size is not equal to hierarchy size. to find inactive fans.");
  }
  fans.clear();

  // 获取查找范围集合
  vector<size_t> find_idxs{};
  for (auto i = 0; i < contours.size(); i++)
  {
    if (mask.find(i) != mask.end())
    {
      continue;
    }
    if (is_hierarchy_inactive_fan(hierarchy, i) == false)
    {
      continue;
    }
    find_idxs.push_back(i);
  }

  vector<vector<size_t>> contours_group{};
  contours_group.reserve(find_idxs.size());
  for (const auto& idx : find_idxs)
  {
    contours_group.emplace_back(vector<size_t>(1, idx));  // 初始化每个轮廓为一个组
  }
  for (auto& group : contours_group)
  {
    vector<ContourConstPtr> temp_contours{};
    temp_contours.reserve(group.size());
    for (auto& idx : group)
    {
      temp_contours.push_back(contours[idx]);
    }
    auto p_fan = RuneFanInactive::MakeFeature(temp_contours);
    if (p_fan)
    {
      fans.push_back(p_fan);
      used_contour_idxs.insert({p_fan, {group.begin(), group.end()}});
    }
  }
}

RuneFanInactivePtr RuneFanInactive::MakeFeature(const vector<ContourConstPtr>& contours)
{
  if (contours.empty())
  {
    return nullptr;
  }

  vector<Point> temp_contour{};
  for (auto& contour : contours)
  {
    temp_contour.insert(temp_contour.end(), contour->Points().begin(),
                        contour->Points().end());
  }
  vector<Point> hull_contour_temp{};
  convexHull(temp_contour, hull_contour_temp);
  ContourConstPtr hull_contour = ContourWrapper<int>::MakeContour(hull_contour_temp);
  double hull_area = hull_contour->Area();
  RotatedRect rotated_rect = hull_contour->MinAreaRect();

  double rect_area = rotated_rect.size.area();
  if (hull_area < rune_fan_param.INACTIVE_MIN_AREA)
  {
    return nullptr;
  }
  if (hull_area > rune_fan_param.INACTIVE_MAX_AREA)
  {
    return nullptr;
  }

  double area_ratio = hull_area / rotated_rect.size.area();
  if (area_ratio < rune_fan_param.INACTIVE_MIN_AREA_RATIO)
  {
    return nullptr;
  }

  double width = max(rotated_rect.size.width, rotated_rect.size.height);
  double height = min(rotated_rect.size.width, rotated_rect.size.height);
  double side_ratio = width / height;
  if (side_ratio < rune_fan_param.INACTIVE_MIN_SIDE_RATIO)
  {
    return nullptr;
  }
  if (side_ratio > rune_fan_param.INACTIVE_MAX_SIDE_RATIO)
  {
    return nullptr;
  }

  return make_shared<RuneFanInactive>(hull_contour, contours, rotated_rect);
}

RuneFanInactivePtr RuneFanInactive::MakeFeature(const Point2f& top_left,
                                                const Point2f& top_right,
                                                const Point2f& bottom_right,
                                                const Point2f& bottom_left)
{
  // 若点发生重合，返回空指针
  if (top_left == top_right || top_left == bottom_right || top_left == bottom_left ||
      top_right == bottom_right || top_right == bottom_left ||
      bottom_right == bottom_left)
  {
    return nullptr;
  }

  return make_shared<RuneFanInactive>(top_left, top_right, bottom_right, bottom_left);
}

RuneFanInactive::RuneFanInactive(const ContourConstPtr& hull_contour,
                                 const vector<ContourConstPtr>& arrow_contours,
                                 const RotatedRect& rotated_rect)
{
  auto width = min(rotated_rect.size.width, rotated_rect.size.height);
  auto height = max(rotated_rect.size.width, rotated_rect.size.height);
  auto center = rotated_rect.center;
  vector<Point2f> corners(4);
  rotated_rect.points(corners.data());

  // 取最小外接矩形的长边作为方向
  Point2f direction_temp{};
  if (get_dist(corners[0], corners[1]) > get_dist(corners[0], corners[3]))
  {
    direction_temp = corners[0] - corners[1];
  }
  else
  {
    direction_temp = corners[0] - corners[3];
  }

  setArrowContours(arrow_contours);
  setActiveFlag(false);
  setRotatedRect(rotated_rect);

  auto& image_info = getImageCache();
  image_info.setContours(vector<ContourConstPtr>{hull_contour});
  image_info.setCorners(corners);
  image_info.setWidth(width);
  image_info.setHeight(height);
  image_info.setCenter(center);
  image_info.setDirection(get_unit_vector(direction_temp));  // 扇叶的方向指向神符中心
}

RuneFanInactive::RuneFanInactive(const Point2f& top_left, const Point2f& top_right,
                                 const Point2f& bottom_right, const Point2f& bottom_left)
{
  Point2f top_center = (top_left + top_right) / 2;
  Point2f bottom_center = (bottom_left + bottom_right) / 2;
  Point2f left_center = (top_left + bottom_left) / 2;
  Point2f right_center = (top_right + bottom_right) / 2;

  auto center = (top_center + bottom_center) / 2;
  auto contour = ContourWrapper<int>::MakeContour(
      {static_cast<Point>(top_left), static_cast<Point>(top_right),
       static_cast<Point>(bottom_right), static_cast<Point>(bottom_left)});
  auto corners = {top_left, top_right, bottom_right, bottom_left};
  auto width = get_dist(left_center, right_center);
  auto height = get_dist(top_center, bottom_center);

  setRotatedRect(contour->MinAreaRect());
  setActiveFlag(false);
  auto& image_info = getImageCache();
  image_info.setContours(vector<ContourConstPtr>{contour});
  image_info.setWidth(width);
  image_info.setHeight(height);
  image_info.setCenter(center);
  image_info.setCorners(corners);
  image_info.setDirection(
      get_unit_vector(bottom_center - top_center));  // 扇叶的方向指向神符中心
}

bool RuneFanInactive::CorrectDirection(FeatureNodePtr& fan,
                                       const cv::Point2f& correct_center)
{
  auto rune_fan = RuneFanInactive::Cast(fan);
  if (rune_fan == nullptr)
  {
    return false;
  }
  if (rune_fan->getActiveFlag())
  {
    return false;
  }
  if (rune_fan->getImageCache().getDirection() == cv::Point2f(0, 0))
  {
    return false;
  }
  if (correct_center == cv::Point2f(0, 0))
  {
    return false;
  }
  vector<Point2f> rect_points(4);
  rune_fan->getRotatedRect().points(rect_points.data());
  Point2f direction{};
  if (get_dist(rect_points[0], rect_points[1]) > get_dist(rect_points[0], rect_points[3]))
  {
    direction = rect_points[0] - rect_points[1];
  }
  else
  {
    direction = rect_points[0] - rect_points[3];
  }

  // 参考方向
  Point2f reference_direction =
      get_unit_vector(correct_center - rune_fan->getImageCache().getCenter());
  if (direction.dot(reference_direction) < 0)
  {
    direction = -direction;
  }
  rune_fan->getImageCache().setDirection(get_unit_vector(direction));

  return true;
}

bool RuneFanInactive::CorrectCorners(FeatureNodePtr& fan)
{
  auto rune_fan = RuneFanInactive::Cast(fan);
  if (rune_fan == nullptr)
  {
    return false;
  }
  if (rune_fan->getActiveFlag())
  {
    return false;
  }
  if (rune_fan->getImageCache().getDirection() == cv::Point2f(0, 0))
  {
    return false;
  }

  vector<Point2f> rect_points(4);
  rune_fan->getRotatedRect().points(rect_points.data());
  vector<Point2f> temp_corners(rect_points.begin(), rect_points.end());
  Point2f dirction = rune_fan->getImageCache().getDirection();
  Point2f center = rune_fan->getRotatedRect().center;
  // 按照在方向上的投影排序
  sort(temp_corners.begin(), temp_corners.end(),
       [&](const Point2f& p1, const Point2f& p2)
       {
         Point2f v1 = p1 - center;
         Point2f v2 = p2 - center;
         return v1.dot(dirction) < v2.dot(dirction);
       });
  Point2f v0 = temp_corners[0] - center;
  Point2f v1 = temp_corners[1] - center;
  Point2f v2 = temp_corners[2] - center;
  Point2f v3 = temp_corners[3] - center;
  if (v0.cross(v1) < 0)
  {
    swap(temp_corners[0], temp_corners[1]);
  }
  if (v2.cross(v3) < 0)
  {
    swap(temp_corners[2], temp_corners[3]);
  }

  rune_fan->getImageCache().setCorners(temp_corners);
  return true;
}

ContourConstPtr RuneFanInactive::GetEndArrowContour(
    FeatureNodePtr& inactive_fan, const vector<FeatureNodeConstPtr>& inactive_targets)
{
  // 0. 判空
  if (inactive_targets.empty())
  {
    return nullptr;
  }
  if (inactive_fan == nullptr)
  {
    return nullptr;
  }
  auto fan = RuneFanInactive::Cast(inactive_fan);
  if (fan->getArrowContours().size() < 2)  // 轮廓数量为1时，不进行过滤
  {
    return nullptr;
  }

  FeatureNodeConstPtr ref_target = nullptr;
  vector<ContourConstPtr> arrow_contours = fan->getArrowContours();
  if (inactive_targets.size() == 1)
  {
    ref_target = inactive_targets.front();
  }
  else
  {
    unordered_map<ContourConstPtr, double> contour_weights{};
    double area_sum = 0;
    for (auto& contour : arrow_contours)
    {
      area_sum += contour->Area();
    }
    if (area_sum == 0)
    {
      return nullptr;
    }

    for (auto& contour : arrow_contours)
    {
      contour_weights[contour] = contour->Area() / area_sum;
    }

    Point2f all_contours_center{0, 0};
    for (auto& contour : arrow_contours)
    {
      all_contours_center +=
          static_cast<Point2f>(contour->Center()) * contour_weights[contour];
    }

    auto near_target =
        *min_element(inactive_targets.begin(), inactive_targets.end(),
                     [&](const FeatureNodeConstPtr& a, const FeatureNodeConstPtr& b)
                     {
                       return norm(a->getImageCache().getCenter() - all_contours_center) <
                              norm(b->getImageCache().getCenter() - all_contours_center);
                     });
    ref_target = near_target;
  }

  auto far_contour = *max_element(arrow_contours.begin(), arrow_contours.end(),
                                  [&](const ContourConstPtr& a, const ContourConstPtr& b)
                                  {
                                    return norm(static_cast<Point2f>(a->Center()) -
                                                ref_target->getImageCache().getCenter()) <
                                           norm(static_cast<Point2f>(b->Center()) -
                                                ref_target->getImageCache().getCenter());
                                  });

  vector<ContourConstPtr> rest_contours = arrow_contours;
  rest_contours.erase(
      remove_if(rest_contours.begin(), rest_contours.end(),
                [&](const ContourConstPtr& contour) { return contour == far_contour; }),
      rest_contours.end());
  auto try_make_fan = RuneFanInactive::MakeFeature(rest_contours);
  if (try_make_fan == nullptr)
  {
    return nullptr;
  }
  fan = try_make_fan;
  inactive_fan = static_cast<FeatureNodePtr>(fan);
  return far_contour;
}

auto RuneFanInactive::GetPnpPoints() const
    -> std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>, std::vector<float>>
{
  vector<Point2f> points_2d{};
  vector<Point3f> points_3d{};
  vector<float> weights{};

  const auto& corners = getImageCache().getCorners();
  points_2d.push_back(corners[0]);
  points_2d.push_back(corners[1]);
  points_3d.push_back(rune_fan_param.INACTIVE_3D[0]);
  points_3d.push_back(rune_fan_param.INACTIVE_3D[1]);

  if (points_2d.size() != points_3d.size())
  {
    throw std::runtime_error("The size of points_2d and points_3d must be equal");
  }
  weights.resize(points_2d.size(), 1.0);

  return make_tuple(points_2d, points_3d, weights);
}

void RuneFanInactive::DrawFeature(cv::Mat& image,
                                  const FeatureNode::DrawConfigConstPtr& /*config*/) const
{
  // 绘制角点
  do
  {
    const auto& image_info = getImageCache();
    if (!image_info.isSetCorners())
    {
      break;
    }
    const auto& corners = image_info.getCorners();
    for (int i = 0; i < static_cast<int>(corners.size()); i++)
    {
      auto color = rune_fan_draw_param.inactive.color;
      auto thickness = rune_fan_draw_param.inactive.thickness;
      line(image, corners[i], corners[(i + 1) % corners.size()], color, thickness,
           LINE_AA);
      auto point_radius = rune_fan_draw_param.inactive.point_radius;
      circle(image, corners[i], point_radius, color, thickness, LINE_AA);
    }
    for (int i = 0; i < static_cast<int>(corners.size()); i++)
    {
      auto font_scale = rune_fan_draw_param.inactive.font_scale;
      auto font_thickness = rune_fan_draw_param.inactive.font_thickness;
      auto font_color = rune_fan_draw_param.inactive.font_color;
      putText(image, to_string(i), corners[i], FONT_HERSHEY_SIMPLEX, font_scale,
              font_color, font_thickness, LINE_AA);
    }

  } while (0);

  // 绘制方向
  do
  {
    if (!getImageCache().isSetDirection())
    {
      break;
    }
    auto arrow_thickness = rune_fan_draw_param.inactive.arrow_thickness;
    auto arrow_length = rune_fan_draw_param.inactive.arrow_length;
    auto arrow_color = rune_fan_draw_param.inactive.arrow_color;
    const auto& image_info = getImageCache();
    if (!image_info.isSetCenter())
    {
      break;
    }
    auto center = image_info.getCenter();
    auto direction = getImageCache().getDirection();
    if (direction == Point2f(0, 0))
    {
      break;
    }
    cv::arrowedLine(image, center, center + direction * arrow_length, arrow_color,
                    arrow_thickness, LINE_AA, 0, 0.1);
  } while (0);
}

}  // namespace rune_detector