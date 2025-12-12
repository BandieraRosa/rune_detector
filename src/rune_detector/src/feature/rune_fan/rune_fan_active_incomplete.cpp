
#include "feature/rune_fan/rune_fan_active.h"
#include "feature/rune_fan/rune_fan_param.h"

namespace rune_detector
{

static inline bool get_humps(const std::vector<cv::Point>& contour,
                             std::vector<TopHump>& top_humps)
{
  if (contour.size() < 30)
  {
    return false;
  }
  std::vector<cv::Point> contour_plus(contour.begin(), contour.end());
  contour_plus.insert(contour_plus.end(), contour.begin(),
                      contour.begin() + static_cast<int>(contour.size() / 3));
  cv::Mat angles_mat = RuneFanActive::GetAngles(contour_plus);
  cv::Mat gradient_mat = RuneFanActive::GetGradient(angles_mat);
  std::vector<std::tuple<Line, Line>> line_pairs;
  RuneFanActive::GetLinePairs(contour_plus, angles_mat, gradient_mat, line_pairs);

  std::vector<TopHump> humps;
  for (int i = 0; i < line_pairs.size(); i++)
  {
    auto& [up_line, down_line] = line_pairs[i];
    cv::Point2f direction(cos(deg2rad(up_line.angle)), sin(deg2rad(up_line.angle)));
    cv::Point2f center =
        (contour_plus[up_line.start_idx] + contour_plus[down_line.start_idx]) / 2.0;
    humps.emplace_back(up_line.start_idx, up_line.end_idx, down_line.start_idx,
                       down_line.end_idx, i, direction, center);
  }
  for (auto& h : humps)
  {
    TopHump::SetVertex(h, contour_plus);
  }
  TopHump::Filter(contour_plus, humps);
  top_humps = humps;
  return true;
}

static inline bool is_overlap(const RuneFanActivePtr& fan1, const RuneFanActivePtr& fan2)
{
  return (fan1->getRotatedRect().boundingRect() & fan2->getRotatedRect().boundingRect())
             .area() > 0;
}

static inline bool filter_fan(const std::vector<FeatureNodePtr>& fans,
                              std::vector<FeatureNodePtr>& filtered_fans,
                              const cv::Point2f& rotate_center)
{
  std::unordered_set<FeatureNodePtr> used_fans(fans.begin(), fans.end());
  for (auto& fan : fans)
  {
    if (used_fans.find(fan) == used_fans.end())
    {
      continue;
    }
    auto rune_fan = RuneFanActive::Cast(fan);
    double angle =
        get_vector_min_angle(rune_fan->getImageCache().getDirection(),
                             rotate_center - rune_fan->getImageCache().getCenter(), DEG);
    if (abs(angle) > rune_fan_param.ACTIVE_MAX_DIRECTION_DELTA_INCOMPLETE)
    {
      used_fans.erase(fan);
    }
  }
  if (used_fans.empty())
  {
    return false;
  }
  filtered_fans.insert(filtered_fans.end(), used_fans.begin(), used_fans.end());
  return true;
}

bool RuneFanActive::FindIncomplete(
    std::vector<FeatureNodePtr>& fans, const std::vector<ContourConstPtr>& contours,
    const std::vector<cv::Vec4i>& hierarchy, const std::unordered_set<size_t>& mask,
    const cv::Point2f& rotate_center,
    std::unordered_map<FeatureNodeConstPtr,
                       std::unordered_set<size_t>>& /*used_contour_idxs*/)
{
  std::unordered_set<size_t> pending_idxs;
  for (size_t i = 0; i < contours.size(); i++)
  {
    if (mask.find(i) == mask.end() && contours[i]->Points().size() >= 6 &&
        hierarchy[i][3] == -1)
    {
      pending_idxs.insert(i);
    }
  }

  for (auto it = pending_idxs.begin(); it != pending_idxs.end();)
  {
    if (contours[*it]->Area() < rune_fan_param.ACTIVE_MIN_AREA_INCOMPLETE)
    {
      it = pending_idxs.erase(it);
    }
    else
    {
      ++it;
    }
  }
  if (pending_idxs.empty())
  {
    return false;
  }

  for (auto it = pending_idxs.begin(); it != pending_idxs.end();)
  {
    double ratio = contours[*it]->Area() / pow(contours[*it]->Perimeter(), 2);
    if (ratio > rune_fan_param.ACTIVE_MAX_AREA_PERIMETER_RATIO_INCOMPLETE)
    {
      it = pending_idxs.erase(it);
    }
    else
    {
      ++it;
    }
  }

  static auto get_contour_idx = [](const ContourConstPtr& c,
                                   const std::vector<ContourConstPtr>& contours) -> int
  {
    auto it = std::find(contours.begin(), contours.end(), c);
    if (it != contours.end())
    {
      return static_cast<int>(std::distance(contours.begin(), it));
    }
    throw std::runtime_error("Contour not found");
    return -1;
  };

  std::vector<std::tuple<TopHump, ContourConstPtr>> all_humps;
  for (auto idx : pending_idxs)
  {
    std::vector<TopHump> temp_humps;
    get_humps(contours[idx]->Points(), temp_humps);
    for (auto& h : temp_humps)
    {
      all_humps.push_back({h, contours[idx]});
    }
  }
  if (all_humps.size() < 3)
  {
    return false;
  }

  std::vector<cv::Point2f> hump_centers(all_humps.size());
  for (size_t i = 0; i < all_humps.size(); i++)
  {
    hump_centers[i] = get<0>(all_humps[i]).GetCenter();
  }

  cv::Mat hump_centers_mat(static_cast<int>(hump_centers.size()), 2, CV_32F);
  for (size_t i = 0; i < hump_centers.size(); i++)
  {
    hump_centers_mat.at<float>(static_cast<int>(i), 0) = hump_centers[i].x;
    hump_centers_mat.at<float>(static_cast<int>(i), 1) = hump_centers[i].y;
  }

  cv::flann::Index kd_tree(hump_centers_mat, cv::flann::KDTreeIndexParams(1));

  std::unordered_map<RuneFanActivePtr, std::unordered_set<size_t>> used_contour_idxs_temp;
  for (size_t i = 0; i < all_humps.size(); i++)
  {
    cv::Mat query = (cv::Mat_<float>(1, 2) << hump_centers[i].x, hump_centers[i].y);
    int max_results = 50;
    cv::Mat indices(1, max_results, CV_32S), dists(1, max_results, CV_32F);
    int found = kd_tree.radiusSearch(query, indices, dists, 500 * 500, max_results,
                                     cv::flann::SearchParams(32));

    for (size_t n = 0; n < found; ++n)
    {
      size_t j = indices.at<int>(0, static_cast<int>(n));
      if (j <= i || sqrt(dists.at<float>(0, static_cast<int>(n))) > 300)
      {
        continue;
      }
      auto &h1 = get<0>(all_humps[i]), &h2 = get<0>(all_humps[j]);
      if (get_vector_min_angle(h1.GetDirection(), h2.GetDirection(), DEG) > 30)
      {
        continue;
      }

      for (size_t n2 = 0; n2 < found; ++n2)
      {
        size_t k = indices.at<int>(0, static_cast<int>(n2));
        if (k <= i || k == j || sqrt(dists.at<float>(0, static_cast<int>(n2))) > 300)
        {
          continue;
        }
        auto& h3 = get<0>(all_humps[k]);
        if (get_vector_min_angle(h1.GetDirection(), h3.GetDirection(), DEG) > 30)
        {
          continue;
        }
        if (get_vector_min_angle(h2.GetDirection(), h3.GetDirection(), DEG) > 30)
        {
          continue;
        }

        auto p_fan = MakeFeature(all_humps[i], all_humps[j], all_humps[k]);
        if (p_fan)
        {
          fans.push_back(p_fan);
          used_contour_idxs_temp[p_fan].insert(
              get_contour_idx(get<1>(all_humps[i]), contours));
          used_contour_idxs_temp[p_fan].insert(
              get_contour_idx(get<1>(all_humps[j]), contours));
          used_contour_idxs_temp[p_fan].insert(
              get_contour_idx(get<1>(all_humps[k]), contours));
        }
      }
    }
  }

  std::vector<FeatureNodePtr> filtered_fans;
  filter_fan(fans, filtered_fans, rotate_center);
  return true;
}

RuneFanActive::RuneFanActive(const std::vector<ContourConstPtr>& contours,
                             const std::vector<cv::Point2f>& top_hump_corners,
                             const cv::Point2f& direction)
{
  std::vector<cv::Point> contour_temp;
  for (auto& c : contours)
  {
    contour_temp.insert(contour_temp.end(), c->begin(), c->end());
  }
  std::vector<cv::Point> hull_contour_temp;
  cv::convexHull(contour_temp, hull_contour_temp);
  auto contour = ContourWrapper<int>::MakeContour(hull_contour_temp);
  cv::RotatedRect fit_ellipse = contour->FittedEllipse();
  float width = std::max(fit_ellipse.size.width, fit_ellipse.size.height);
  float height = std::min(fit_ellipse.size.width, fit_ellipse.size.height);
  cv::Point2f center = fit_ellipse.center;

  std::vector<cv::Point2f> corners(top_hump_corners.begin(), top_hump_corners.end());
  const cv::Point2f& top_center = top_hump_corners[1];
  float max_projection = 0;
  for (auto& point : contour->Points())
  {
    float projection =
        get_projection(static_cast<cv::Point2f>(point) - top_center, direction);
    max_projection = std::max(max_projection, projection);
  }
  corners.emplace_back(top_center + max_projection * direction);

  setActiveFlag(true);
  setTopHumpCorners(top_hump_corners);
  setRotatedRect(fit_ellipse);

  auto& image_info = getImageCache();
  image_info.setContours(std::vector<ContourConstPtr>{contour});
  image_info.setWidth(width);
  image_info.setHeight(height);
  image_info.setCenter(center);
  image_info.setCorners(corners);
  image_info.setDirection(direction);
}

static inline cv::Point2f get_hump_center(
    const std::array<std::tuple<TopHump, ContourConstPtr>, 3>& humps)
{
  std::unordered_set<ContourConstPtr> contours;
  for (auto& [h, c] : humps)
  {
    contours.insert(c);
  }
  if (contours.size() == 3)
  {
    cv::Point2f ave_point = (get<0>(humps[0]).GetVertex() + get<0>(humps[1]).GetVertex() +
                             get<0>(humps[2]).GetVertex()) /
                            3.0;
    std::array<float, 3> distances;
    for (int i = 0; i < 3; i++)
    {
      distances[i] = get_dist(ave_point, get<0>(humps[i]).GetVertex());
    }
    return get<1>(
               humps[std::distance(distances.begin(),
                                   std::min_element(distances.begin(), distances.end()))])
        ->Center();
  }
  else if (contours.size() == 2)
  {
    auto hull_contour =
        ContourWrapper<int>::GetConvexHull({contours.begin(), contours.end()});
    return hull_contour->FittedEllipse().center;
  }
  else if (contours.size() == 1)
  {
    return (*contours.begin())->FittedEllipse().center;
  }
  throw std::runtime_error("humps size is not equal to 3");
}

RuneFanActivePtr RuneFanActive::MakeFeature(
    const std::tuple<TopHump, ContourConstPtr>& h1,
    const std::tuple<TopHump, ContourConstPtr>& h2,
    const std::tuple<TopHump, ContourConstPtr>& h3)
{
  auto& [hump_1_obj, contour_1] = h1;
  auto& [hump_2_obj, contour_2] = h2;
  auto& [hump_3_obj, contour_3] = h3;

  cv::Point2f contours_center = get_hump_center({h1, h2, h3});
  TopHumpCombo hump_combo(hump_1_obj, hump_2_obj, hump_3_obj);

  double max_distance =
      std::max({get_dist(hump_1_obj.GetVertex(), hump_2_obj.GetVertex()),
                get_dist(hump_1_obj.GetVertex(), hump_3_obj.GetVertex()),
                get_dist(hump_2_obj.GetVertex(), hump_3_obj.GetVertex())});
  if (max_distance == 0)
  {
    return nullptr;
  }

  double max_side_length = std::max({std::max(contour_1->FittedEllipse().size.width,
                                              contour_1->FittedEllipse().size.height),
                                     std::max(contour_2->FittedEllipse().size.width,
                                              contour_2->FittedEllipse().size.height),
                                     std::max(contour_3->FittedEllipse().size.width,
                                              contour_3->FittedEllipse().size.height)});
  if (max_side_length == 0 || max_distance > 3.0 * max_side_length)
  {
    return nullptr;
  }

  double angle_delta = 1e5;
  if (!TopHump::MakeTopHumps(hump_combo, contours_center, angle_delta))
  {
    return nullptr;
  }
  std::vector<ContourConstPtr> fan_contours{contour_1, contour_2, contour_3};
  auto& top_humps = hump_combo.Humps();
  cv::Point2f direction = -1 * top_humps[1].GetDirection();
  std::vector<cv::Point2f> top_humps_corners;
  top_humps_corners.reserve(top_humps.size());
  for (auto& h : top_humps)
  {
    top_humps_corners.push_back(h.GetVertex());
  }

  auto fan = make_shared<RuneFanActive>(fan_contours, top_humps_corners, direction);
  if (fan && !fan->isSetError())
  {
    fan->setError(angle_delta);
  }
  return fan;
}
}  // namespace rune_detector