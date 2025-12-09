#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "common/geom_utils.hpp"
#include "common/param.hpp"
#include "feature/rune_fan/rune_fan_hump.h"
#include "feature/rune_fan/rune_fan_hump_param.h"

#define RUNE_FAN_DEBUG 1

namespace rune_detector
{

struct TopHumpDebugParam : public Param
{
  bool ENABLE_DEBUG = false;
  void LoadFromNode(rclcpp::Node& node) override
  {
    const std::string PREFIX = "top_hump_debug.";
    auto get_param = [&](const std::string& key, auto& var, const auto& default_val)
    {
      if (!node.has_parameter(PREFIX + key))
      {
        var = node.declare_parameter<std::decay_t<decltype(var)>>(PREFIX + key,
                                                                  default_val);
      }
      else
      {
        node.get_parameter(PREFIX + key, var);
      }
    };
    get_param("enable_debug", ENABLE_DEBUG, false);
  }
};

inline TopHumpDebugParam top_hump_debug_param;

TopHump::TopHump(int _up_start_idx, int _up_end_idx, int _down_start_idx,
                 int _down_end_idx, int _line_pair_idx, const cv::Point2f& _direction,
                 const cv::Point2f& _center)
    : up_start_idx(_up_start_idx),
      up_end_idx(_up_end_idx),
      down_start_idx(_down_start_idx),
      down_end_idx(_down_end_idx),
      line_pair_idx(_line_pair_idx),
      up_itertaion_num(_up_end_idx - _up_start_idx),
      down_itertaion_num(_down_end_idx - _down_start_idx),
      end_iteration_up_idx(_up_end_idx),
      current_state(DOWN)
{
  direction = _direction;
  center = _center;
}

TopHump::TopHump(int up_idx, int down_idx, state find_state,
                 const cv::Point2f& _direction)
    : current_state(find_state),
      up_itertaion_num(0),
      down_itertaion_num(0),
      end_iteration_up_idx(up_idx)
{
  direction = _direction;
  if (find_state == UP)
  {
    up_start_idx = up_idx;
    down_start_idx = down_idx;
    up_end_idx = down_end_idx = 0;
  }
  else if (find_state == DOWN)
  {
    up_end_idx = up_idx;
    down_end_idx = down_idx;
    up_start_idx = down_start_idx = 0;
  }
}

std::vector<TopHump> TopHump::getTopHumps(
    const std::vector<cv::Point>& contour_plus, const cv::Point2f& contour_center,
    const std::vector<std::tuple<Line, Line>>& line_pairs)
{
  if (contour_plus.size() < 20)
  {
    std::cout << "fan active get_top_humps : 轮廓长度小于20,无法进行查找,轮廓数量："
              << contour_plus.size() << '\n';
    return {};
  }
  std::vector<TopHump> humps{};
  TopHump::getAllHumps2(contour_plus, line_pairs, humps);
  if (humps.size() < 3)
  {
    std::cout
        << "fan active get_top_humps : 收集到的突起点数不足3个,无法进行查找,轮廓数量："
        << contour_plus.size() << '\n';
    return {};
  }
  TopHump::filter(contour_plus, humps);
  if (humps.size() < 3)
  {
    std::cout
        << "fan active get_top_humps : 过滤后的突起点数不足3个,无法进行查找,轮廓数量："
        << contour_plus.size() << '\n';
    return {};
  }
  std::vector<std::tuple<TopHumpCombo, double>> hump_combos{};
  for (size_t i = 0; i < humps.size() - 2; ++i)
    for (size_t j = i + 1; j < humps.size() - 1; ++j)
      for (size_t k = j + 1; k < humps.size(); ++k)
      {
        TopHumpCombo hump_combo = {humps[i], humps[j], humps[k]};
        double delta = 0;
        if (TopHump::make_TopHumps(hump_combo, contour_center, delta))
          hump_combos.emplace_back(hump_combo, delta);
      }
  if (hump_combos.empty()) return {};
  auto [best_combo, delta] =
      *min_element(hump_combos.begin(), hump_combos.end(),
                   [](const auto& a, const auto& b) { return get<1>(a) < get<1>(b); });
  return best_combo.getHumpsVector();
}

float gaussian(float x, float sigma)
{
  return (1.0 / (sqrt(2 * M_PI) * sigma)) * exp(-x * x / (2 * sigma * sigma));
}
void normalize(std::vector<float>& kernel)
{
  float sum = 0;
  for (float k : kernel) sum += k;
  for (float& k : kernel) k /= sum;
}
std::vector<float> createGaussianKernel(int size, float sigma)
{
  std::vector<float> kernel(size);
  int center = size / 2;
  for (int i = 0; i < size; ++i) kernel[i] = gaussian(i - center, sigma);
  normalize(kernel);
  return kernel;
}

void TopHump::update(const TopHump& hump)
{
  current_state = hump.current_state;
  if (current_state == UP)
  {
    up_start_idx = std::min(up_start_idx, hump.up_start_idx);
    down_start_idx = std::min(down_start_idx, hump.down_start_idx);
    up_itertaion_num++;
  }
  else if (current_state == DOWN)
  {
    up_end_idx = std::max(up_end_idx, hump.up_end_idx);
    down_end_idx = std::max(down_end_idx, hump.down_end_idx);
    down_itertaion_num++;
  }
  end_iteration_up_idx = hump.end_iteration_up_idx;
  int iteration_num = up_itertaion_num + down_itertaion_num;
  float ratio_old = float(iteration_num - 1) / iteration_num,
        ratio_new = 1.0f / iteration_num;
  direction = direction * ratio_old + hump.direction * ratio_new;
}

inline bool TopHump::getAllHumps2(const std::vector<cv::Point>& contours_plus,
                                  const std::vector<std::tuple<Line, Line>>& line_pairs,
                                  std::vector<TopHump>& humps)
{
  humps.clear();
  for (int i = 0; i < line_pairs.size(); i++)
  {
    auto& [up_line, down_line] = line_pairs[i];
    int up_start_idx = up_line.start_idx, up_end_idx = up_line.end_idx,
        down_start_idx = down_line.start_idx, down_end_idx = down_line.end_idx;
    cv::Point2f direction =
        cv::Point2f(cos(deg2rad(up_line.angle)), sin(deg2rad(up_line.angle)));
    cv::Point2f center =
        (contours_plus[up_start_idx] + contours_plus[down_start_idx]) / 2.0;
    humps.emplace_back(up_start_idx, up_end_idx, down_start_idx, down_end_idx, i,
                       direction, center);
  }
  for (auto& hump : humps) setVertex(hump, contours_plus);
  return humps.size() >= 3;
}

bool TopHump::filter(const std::vector<cv::Point>& contour_plus,
                     std::vector<TopHump>& humps)
{
  if (humps.empty()) return false;
  for (auto it_1 = humps.begin(); it_1 != humps.end(); it_1++)
    for (auto it_2 = it_1 + 1; it_2 != humps.end();)
    {
      if (get_vector_min_angle(it_1->getDirection(), it_2->getDirection(), DEG) <
          rune_fan_hump_param.TOP_HUMP_MAX_ALIGNMENT_DELTA)
      {
        if (get_dist(it_1->getVertex(), it_2->getVertex()) <
            rune_fan_hump_param.TOP_HUMP_MIN_INTERVAL)
          it_2 = humps.erase(it_2);
        else
          it_2++;
      }
      else
      {
        if (get_dist(it_1->getVertex(), it_2->getVertex()) <
            2 * rune_fan_hump_param.TOP_HUMP_MIN_INTERVAL)
          it_2 = humps.erase(it_2);
        else
          it_2++;
      }
    }
  return true;
}

bool TopHump::make_TopHumps(TopHumpCombo& hump_combo, const cv::Point2f& contour_center,
                            double& delta)
{
  auto& humps = hump_combo.humps();
  std::vector<TopHump> humps_to_classify(humps.begin(), humps.end());
  TopHump left_h, right_h, center_h;
  cv::Point2f aveVertex{};
  for (auto& h : humps) aveVertex += h.getVertex();
  aveVertex /= static_cast<float>(humps.size());
  auto center_hump_it = min_element(humps_to_classify.begin(), humps_to_classify.end(),
                                    [&](const TopHump& h1, const TopHump& h2)
                                    {
                                      return get_dist(h1.getVertex(), aveVertex) <
                                             get_dist(h2.getVertex(), aveVertex);
                                    });
  center_h = *center_hump_it;
  humps_to_classify.erase(center_hump_it);
  if (humps_to_classify.size() != 2) return false;
  cv::Point2f c0 = humps_to_classify[0].getVertex() - contour_center,
              c1 = humps_to_classify[1].getVertex() - contour_center;
  if (c0.x * c1.y - c0.y * c1.x > 0)
  {
    left_h = humps_to_classify[0];
    right_h = humps_to_classify[1];
  }
  else
  {
    left_h = humps_to_classify[1];
    right_h = humps_to_classify[0];
  }

  if (HumpDetector::CheckCollinearity(left_h.getVertex(), center_h.getVertex(),
                                      right_h.getVertex()) >
      rune_fan_hump_param.TOP_HUMP_MAX_COLLINEAR_DELTA)
    return false;

  if (get_vector_min_angle(center_h.getDirection(), center_h.getVertex() - contour_center,
                           DEG) > rune_fan_hump_param.TOP_HUMP_MAX_DIRECTION_DELTA)
    return false;

  if (HumpDetector::CheckAlignment(left_h.getDirection(), center_h.getDirection(),
                                   right_h.getDirection()) >
      rune_fan_hump_param.TOP_HUMP_MAX_ALIGNMENT_DELTA)
    return false;

  double distance_ratio = get_dist(left_h.getVertex(), center_h.getVertex()) /
                          get_dist(center_h.getVertex(), right_h.getVertex());
  if (distance_ratio < 1.0) distance_ratio = 1.0 / distance_ratio;
  if (distance_ratio > rune_fan_hump_param.TOP_HUMP_MAX_DISTANCE_RATIO) return false;

  cv::Vec4f line;
  fitLine(std::vector<cv::Point2f>{left_h.getVertex(), center_h.getVertex(),
                                   right_h.getVertex()},
          line, cv::DIST_L2, 0, 0.01, 0.01);
  cv::Point2f ave_direction =
      (left_h.getDirection() + center_h.getDirection() + right_h.getDirection()) / 3.0;
  double line_direction_delta =
      get_vector_min_angle(cv::Point2f(line[0], line[1]), ave_direction, DEG);
  line_direction_delta =
      line_direction_delta > 90 ? 180 - line_direction_delta : line_direction_delta;

  if (line_direction_delta < rune_fan_hump_param.TOP_HUMP_MIN_LINE_DIRECTION_DELTA)
    return false;

  delta = pow(HumpDetector::CheckAlignment(left_h.getDirection(), center_h.getDirection(),
                                           right_h.getDirection()),
              2);
  humps = {left_h, center_h, right_h};
  return true;
}

bool TopHump::setVertex(TopHump& hump, const std::vector<cv::Point>& contour_plus)
{
  if (contour_plus.size() < 20) return false;
  auto farthest_point = max_element(
      contour_plus.begin() + hump.up_start_idx, contour_plus.begin() + hump.down_end_idx,
      [&](const cv::Point& p1, const cv::Point& p2)
      {
        return get_projection(cv::Point2f(p1) - hump.getCenter(), hump.getDirection()) <
               get_projection(cv::Point2f(p2) - hump.getCenter(), hump.getDirection());
      });
  hump.vertex = hump.getCenter() +
                get_projection_vector(cv::Point2f(*farthest_point) - hump.getCenter(),
                                      hump.getDirection());
  return true;
}

}  // namespace rune_detector