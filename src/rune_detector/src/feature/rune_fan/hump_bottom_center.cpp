#include "feature/rune_fan/rune_fan_hump.h"
#include "feature/rune_fan/rune_fan_hump_param.h"

namespace rune_detector
{

BottomCenterHump::BottomCenterHump(const cv::Point2f& _center,
                                   const cv::Point2f& direction, int _idx)
    : idx_(_idx)
{
  center_ = _center;
  this->direction_ = direction;
}

std::vector<BottomCenterHump> BottomCenterHump::GetBottomCenterHump(
    const std::vector<cv::Point>& contour, const cv::Point2f& /*contour_center*/,
    const std::vector<TopHump>& top_humps)
{
  if (contour.size() < 30)
  {
    return {};
  }
  if (top_humps.size() != 3)
  {
    throw std::runtime_error("Size of the \"top_humps\" are not equal to 3. (size = " +
                             std::to_string(top_humps.size()) + ")");
  }

  std::vector<BottomCenterHump> humps;
  if (!GetAllHumps(contour, top_humps, humps) || !Filter(contour, top_humps, humps) ||
      humps.size() != 1)
  {
    return {};
  }

  humps[0].vertex_ = humps[0].GetCenter();
  return humps;
}

bool BottomCenterHump::GetAllHumps(const std::vector<cv::Point>& contour,
                                   const std::vector<TopHump>& top_humps,
                                   std::vector<BottomCenterHump>& humps)
{
  if (contour.size() < 30)
  {
    return false;
  }
  if (top_humps.size() != 3)
  {
    throw std::runtime_error("Size of the \"top_humps\" are not equal to 3. (size = " +
                             std::to_string(top_humps.size()) + ")");
  }

  cv::Point2f line_center = top_humps[1].GetCenter();
  cv::Point2f line_direction = -top_humps[1].GetDirection();
  std::vector<BottomCenterHump> found_humps;

  cv::Point last_point = contour.back();
  int idx = 0;
  for (auto& point : contour)
  {
    cv::Point2f v1 = static_cast<cv::Point2f>(point) - line_center;
    cv::Point2f v2 = static_cast<cv::Point2f>(last_point) - line_center;
    if ((v1.x * line_direction.y - v1.y * line_direction.x) *
            (v2.x * line_direction.y - v2.y * line_direction.x) <
        0)
    {
      found_humps.emplace_back((v1 + v2) / 2.0f + line_center, line_direction, idx);
    }
    last_point = point;
    idx++;
  }

  if (found_humps.empty())
  {
    return false;
  }
  humps = std::move(found_humps);
  return true;
}

bool BottomCenterHump::Filter(const std::vector<cv::Point>& contour,
                              const std::vector<TopHump>& top_humps,
                              std::vector<BottomCenterHump>& humps)
{
  if (contour.size() < 30 || humps.empty())
  {
    humps = {};
    return false;
  }
  if (top_humps.size() != 3)
  {
    throw std::runtime_error("Size of the \"top_humps\" are not equal to 3. (size = " +
                             std::to_string(top_humps.size()) + ")");
  }

  for (auto it1 = humps.begin(); it1 != humps.end(); it1++)
  {
    for (auto it2 = it1 + 1; it2 != humps.end();)
    {
      if (static_cast<float>(rune_fan_hump_param.BOTTOM_CENTER_HUMP_MIN_INTERVAL) >
          get_dist(it1->GetCenter(), it2->GetCenter()))
      {
        it2 = humps.erase(it2);
      }
      else
      {
        it2++;
      }
    }
  }
  if (humps.empty())
  {
    return false;
  }

  cv::Point2f line_center = top_humps[1].GetCenter();
  cv::Point2f line_direction = -top_humps[1].GetDirection();
  auto bottom_it =
      max_element(humps.begin(), humps.end(),
                  [&](const BottomCenterHump& h1, const BottomCenterHump& h2)
                  {
                    return (h1.GetCenter() - line_center).dot(line_direction) <
                           (h2.GetCenter() - line_center).dot(line_direction);
                  });

  humps = {*bottom_it};
  float delta_angle =
      get_vector_min_angle(line_direction, humps[0].GetCenter() - line_center, DEG);
  if (delta_angle >
      static_cast<float>(rune_fan_hump_param.BOTTOM_CENTER_HUMP_MAX_DELTA_ANGLE))
  {
    humps = {};
    return false;
  }
  return true;
}

}  // namespace rune_detector