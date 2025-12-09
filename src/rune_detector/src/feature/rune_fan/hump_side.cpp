#include <numeric>

#include "common/geom_utils.hpp"
#include "feature/rune_fan/rune_fan_hump.h"
#include "feature/rune_fan/rune_fan_hump_param.h"

namespace rune_detector
{

SideHump::SideHump(const cv::Point2f& _direction, const cv::Point2f& _center)
{
  direction = _direction;
  center = _center;
}

// 获取侧边突起点
std::vector<SideHump> SideHump::getSideHumps(
    const std::vector<cv::Point>& contour_plus, const cv::Point2f& contour_center,
    const std::vector<TopHump>& top_humps,
    const std::vector<BottomCenterHump>& bottom_center_humps,
    std::vector<std::tuple<Line, Line>>& line_pairs)
{
  if (line_pairs.size() < 2) return {};

  std::vector<SideHump> humps;
  if (!getAllHumps(top_humps, line_pairs, humps) || humps.size() < 2) return {};

  for (auto& hump : humps) setVertex(hump, contour_plus);
  filter(humps);
  if (humps.size() < 2) return {};

  std::vector<std::tuple<std::vector<SideHump>, double>> hump_groups;

  // 构造突起点组合
  for (size_t i = 0; i < humps.size() - 1; ++i)
  {
    for (size_t j = i + 1; j < humps.size(); ++j)
    {
      std::vector<SideHump> temp_humps = {humps[i], humps[j]};
      double delta = 0;
      if (make_SideHumps(temp_humps, top_humps, contour_center, delta))
        hump_groups.emplace_back(temp_humps, delta);
    }
  }

  if (hump_groups.empty()) return {};

  return std::get<0>(*min_element(hump_groups.begin(), hump_groups.end(),
                                  [](const auto& a, const auto& b)
                                  { return get<1>(a) < get<1>(b); }));
}

// 获取所有候选侧边突起点
bool SideHump::getAllHumps(const std::vector<TopHump>& top_humps,
                           const std::vector<std::tuple<Line, Line>>& line_pairs,
                           std::vector<SideHump>& humps)
{
  humps.clear();
  if (top_humps.size() != 3)
  {
    std::ostringstream oss;
    oss << "top_humps size != 3 (size=" << top_humps.size() << ")";
    throw std::runtime_error(oss.str());
  }

  std::unordered_set<int> line_pair_idx_set;
  for (int i = 0; i < line_pairs.size(); ++i) line_pair_idx_set.insert(i);
  for (auto& top_hump : top_humps) line_pair_idx_set.erase(top_hump.getLinePairIdx());

  auto isSideHump = [&](const cv::Point2f& top_center, const cv::Point2f& top_dir,
                        const cv::Point2f& center, const cv::Point2f& dir)
  {
    if (get_vector_min_angle(top_dir, dir, DEG) >
        rune_fan_hump_param.SIDE_HUMP_MAX_ANGLE_DELTA)
      return false;
    cv::Point2f v1 = top_dir, v2 = top_center - center, v3 = dir;
    if (v1.cross(v2) * v2.cross(v3) < 0) return false;
    if (get_vector_min_angle(v1, v2, DEG) >
            rune_fan_hump_param.SIDE_HUMP_MAX_ANGLE_DELTA / 2.0 ||
        get_vector_min_angle(v2, v3, DEG) >
            rune_fan_hump_param.SIDE_HUMP_MAX_ANGLE_DELTA / 2.0)
      return false;
    return true;
  };

  cv::Point2f l_center = top_humps[0].getCenter(), l_dir = top_humps[0].getDirection();
  cv::Point2f r_center = top_humps[2].getCenter(), r_dir = top_humps[2].getDirection();

  for (auto& idx : line_pair_idx_set)
  {
    auto& [up_line, down_line] = line_pairs[idx];
    cv::Point2f center = (up_line.center + down_line.center) / 2.0;
    cv::Point2f dir(cos(deg2rad(up_line.angle)), sin(deg2rad(up_line.angle)));

    float to_l = get_dist(center, l_center);
    float to_r = get_dist(center, r_center);

    if (to_l < to_r && isSideHump(l_center, l_dir, center, dir))
    {
      humps.emplace_back(
          dir, get_line_intersection(l_center, l_center + l_dir, center, center + dir));
    }
    else if (to_r <= to_l && isSideHump(r_center, r_dir, center, dir))
    {
      humps.emplace_back(
          dir, get_line_intersection(r_center, r_center + r_dir, center, center + dir));
    }
  }

  return humps.size() >= 2;
}

// 设置突起点的顶点位置
bool SideHump::setVertex(SideHump& hump, const std::vector<cv::Point>& /*contour_plus*/)
{
  hump.vertex = hump.center;
  return true;
}

// 过滤几乎重合的突起点
bool SideHump::filter(std::vector<SideHump>& humps)
{
  for (auto it1 = humps.begin(); it1 != humps.end(); ++it1)
  {
    for (auto it2 = it1 + 1; it2 != humps.end();)
    {
      if (get_dist(it1->getCenter(), it2->getCenter()) < 10)
        it2 = humps.erase(it2);
      else
        ++it2;
    }
  }
  return true;
}

// 确定左右侧突起点
bool SideHump::make_SideHumps(std::vector<SideHump>& humps,
                              const std::vector<TopHump>& top_humps,
                              const cv::Point2f& /*contour_center*/, double& delta)
{
  if (humps.size() != 2)
  {
    std::ostringstream oss;
    oss << "humps size != 2 (size=" << humps.size() << ")";
    throw std::runtime_error(oss.str());
    return false;
  }

  cv::Point2f fan_dir = top_humps[1].getDirection();
  cv::Point2f fan_center = top_humps[1].getCenter();

  double cross0 = (humps[0].getCenter() - fan_center).cross(fan_dir);
  double cross1 = (humps[1].getCenter() - fan_center).cross(fan_dir);

  if (cross0 * cross1 > 0) return false;

  if (cross0 < 0) std::swap(humps[0], humps[1]);
  delta = 0;
  return true;
}

}  // namespace rune_detector