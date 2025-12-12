#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

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

static inline TopHumpDebugParam top_hump_debug_param;

TopHump::TopHump(int _up_start_idx, int _up_end_idx, int _down_start_idx,
                 int _down_end_idx, int _line_pair_idx, const cv::Point2f& _direction,
                 const cv::Point2f& _center)
    : up_start_idx_(_up_start_idx),
      up_end_idx_(_up_end_idx),
      down_start_idx_(_down_start_idx),
      down_end_idx_(_down_end_idx),
      line_pair_idx_(_line_pair_idx),
      up_itertaion_num_(_up_end_idx - _up_start_idx),
      down_itertaion_num_(_down_end_idx - _down_start_idx),
      end_iteration_up_idx_(_up_end_idx),
      current_state_(State::DOWN)
{
  direction_ = _direction;
  center_ = _center;
}

TopHump::TopHump(int up_idx, int down_idx, State find_state,
                 const cv::Point2f& _direction)
    : current_state_(find_state), end_iteration_up_idx_(up_idx)
{
  direction_ = _direction;
  if (find_state == State::UP)
  {
    up_start_idx_ = up_idx;
    down_start_idx_ = down_idx;
    up_end_idx_ = down_end_idx_ = 0;
  }
  else if (find_state == State::DOWN)
  {
    up_end_idx_ = up_idx;
    down_end_idx_ = down_idx;
    up_start_idx_ = down_start_idx_ = 0;
  }
}

std::vector<TopHump> TopHump::GetTopHumps(
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
  TopHump::GetAllHumps2(contour_plus, line_pairs, humps);
  if (humps.size() < 3)
  {
    std::cout
        << "fan active get_top_humps : 收集到的突起点数不足3个,无法进行查找,轮廓数量："
        << contour_plus.size() << '\n';
    return {};
  }
  TopHump::Filter(contour_plus, humps);
  if (humps.size() < 3)
  {
    std::cout
        << "fan active get_top_humps : 过滤后的突起点数不足3个,无法进行查找,轮廓数量："
        << contour_plus.size() << '\n';
    return {};
  }
  std::vector<std::tuple<TopHumpCombo, double>> hump_combos{};
  for (size_t i = 0; i < humps.size() - 2; ++i)
  {
    for (size_t j = i + 1; j < humps.size() - 1; ++j)
    {
      for (size_t k = j + 1; k < humps.size(); ++k)
      {
        TopHumpCombo hump_combo = {humps[i], humps[j], humps[k]};
        double delta = 0;
        if (TopHump::MakeTopHumps(hump_combo, contour_center, delta))
        {
          hump_combos.emplace_back(hump_combo, delta);
        }
      }
    }
  }
  if (hump_combos.empty())
  {
    return {};
  }
  auto [best_combo, delta] =
      *min_element(hump_combos.begin(), hump_combos.end(),
                   [](const auto& a, const auto& b) { return get<1>(a) < get<1>(b); });
  return best_combo.GetHumpsVector();
}

static float gaussian(float x, float sigma)
{
  return static_cast<float>(1.0 / (sqrt(2 * M_PI) * sigma)) *
         exp(-x * x / (2 * sigma * sigma));
}
static void normalize(std::vector<float>& kernel)
{
  float sum = 0;
  for (float k : kernel)
  {
    sum += k;
  }
  for (float& k : kernel)
  {
    k /= sum;
  }
}
static std::vector<float> create_gaussian_kernel(int size, float sigma)
{
  std::vector<float> kernel(size);
  int center = size / 2;
  for (int i = 0; i < size; ++i)
  {
    kernel[i] = gaussian(static_cast<float>(i - center), sigma);
  }
  normalize(kernel);
  return kernel;
}

void TopHump::Update(const TopHump& hump)
{
  current_state_ = hump.current_state_;
  if (current_state_ == State::UP)
  {
    up_start_idx_ = std::min(up_start_idx_, hump.up_start_idx_);
    down_start_idx_ = std::min(down_start_idx_, hump.down_start_idx_);
    up_itertaion_num_++;
  }
  else if (current_state_ == State::DOWN)
  {
    up_end_idx_ = std::max(up_end_idx_, hump.up_end_idx_);
    down_end_idx_ = std::max(down_end_idx_, hump.down_end_idx_);
    down_itertaion_num_++;
  }
  end_iteration_up_idx_ = hump.end_iteration_up_idx_;
  int iteration_num = up_itertaion_num_ + down_itertaion_num_;
  float ratio_old =
            static_cast<float>(iteration_num - 1) / static_cast<float>(iteration_num),
        ratio_new = static_cast<float>(1.0f) / static_cast<float>(iteration_num);
  direction_ = direction_ * ratio_old + hump.direction_ * ratio_new;
}

inline bool TopHump::GetAllHumps2(const std::vector<cv::Point>& contours_plus,
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
  for (auto& hump : humps)
  {
    SetVertex(hump, contours_plus);
  }
  return humps.size() >= 3;
}

bool TopHump::Filter(const std::vector<cv::Point>& /*contour_plus*/,
                     std::vector<TopHump>& humps)
{
  if (humps.empty())
  {
    return false;
  }
  for (auto it_1 = humps.begin(); it_1 != humps.end(); it_1++)
  {
    for (auto it_2 = it_1 + 1; it_2 != humps.end();)
    {
      if (get_vector_min_angle(it_1->GetDirection(), it_2->GetDirection(), DEG) <
          rune_fan_hump_param.TOP_HUMP_MAX_ALIGNMENT_DELTA)
      {
        if (get_dist(it_1->GetVertex(), it_2->GetVertex()) <
            static_cast<float>(rune_fan_hump_param.TOP_HUMP_MIN_INTERVAL))
        {
          it_2 = humps.erase(it_2);
        }
        else
        {
          it_2++;
        }
      }
      else
      {
        if (get_dist(it_1->GetVertex(), it_2->GetVertex()) <
            static_cast<float>(2 * rune_fan_hump_param.TOP_HUMP_MIN_INTERVAL))
        {
          it_2 = humps.erase(it_2);
        }
        else
        {
          it_2++;
        }
      }
    }
  }
  return true;
}

bool TopHump::MakeTopHumps(TopHumpCombo& hump_combo, const cv::Point2f& contour_center,
                           double& delta)
{
  auto& humps = hump_combo.Humps();
  std::vector<TopHump> humps_to_classify(humps.begin(), humps.end());
  TopHump left_h, right_h, center_h;
  cv::Point2f ave_vertex{};
  for (auto& h : humps)
  {
    ave_vertex += h.GetVertex();
  }
  ave_vertex /= static_cast<float>(humps.size());
  auto center_hump_it = min_element(humps_to_classify.begin(), humps_to_classify.end(),
                                    [&](const TopHump& h1, const TopHump& h2)
                                    {
                                      return get_dist(h1.GetVertex(), ave_vertex) <
                                             get_dist(h2.GetVertex(), ave_vertex);
                                    });
  center_h = *center_hump_it;
  humps_to_classify.erase(center_hump_it);
  if (humps_to_classify.size() != 2)
  {
    return false;
  }
  cv::Point2f c0 = humps_to_classify[0].GetVertex() - contour_center,
              c1 = humps_to_classify[1].GetVertex() - contour_center;
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

  if (HumpDetector::CheckCollinearity(left_h.GetVertex(), center_h.GetVertex(),
                                      right_h.GetVertex()) >
      rune_fan_hump_param.TOP_HUMP_MAX_COLLINEAR_DELTA)
  {
    return false;
  }

  if (get_vector_min_angle(center_h.GetDirection(), center_h.GetVertex() - contour_center,
                           DEG) > rune_fan_hump_param.TOP_HUMP_MAX_DIRECTION_DELTA)
  {
    return false;
  }

  if (HumpDetector::CheckAlignment(left_h.GetDirection(), center_h.GetDirection(),
                                   right_h.GetDirection()) >
      rune_fan_hump_param.TOP_HUMP_MAX_ALIGNMENT_DELTA)
  {
    return false;
  }

  double distance_ratio = get_dist(left_h.GetVertex(), center_h.GetVertex()) /
                          get_dist(center_h.GetVertex(), right_h.GetVertex());
  if (distance_ratio < 1.0)
  {
    distance_ratio = 1.0 / distance_ratio;
  }
  if (distance_ratio > rune_fan_hump_param.TOP_HUMP_MAX_DISTANCE_RATIO)
  {
    return false;
  }

  cv::Vec4f line;
  fitLine(std::vector<cv::Point2f>{left_h.GetVertex(), center_h.GetVertex(),
                                   right_h.GetVertex()},
          line, cv::DIST_L2, 0, 0.01, 0.01);
  cv::Point2f ave_direction =
      (left_h.GetDirection() + center_h.GetDirection() + right_h.GetDirection()) / 3.0;
  double line_direction_delta =
      get_vector_min_angle(cv::Point2f(line[0], line[1]), ave_direction, DEG);
  line_direction_delta =
      line_direction_delta > 90 ? 180 - line_direction_delta : line_direction_delta;

  if (line_direction_delta < rune_fan_hump_param.TOP_HUMP_MIN_LINE_DIRECTION_DELTA)
  {
    return false;
  }

  delta = pow(HumpDetector::CheckAlignment(left_h.GetDirection(), center_h.GetDirection(),
                                           right_h.GetDirection()),
              2);
  humps = {left_h, center_h, right_h};
  return true;
}

bool TopHump::SetVertex(TopHump& hump, const std::vector<cv::Point>& contour_plus)
{
  if (contour_plus.size() < 20)
  {
    return false;
  }
  auto farthest_point = max_element(
      contour_plus.begin() + hump.up_start_idx_,
      contour_plus.begin() + hump.down_end_idx_,
      [&](const cv::Point& p1, const cv::Point& p2)
      {
        return get_projection(cv::Point2f(p1) - hump.GetCenter(), hump.GetDirection()) <
               get_projection(cv::Point2f(p2) - hump.GetCenter(), hump.GetDirection());
      });
  hump.vertex_ = hump.GetCenter() +
                 get_projection_vector(cv::Point2f(*farthest_point) - hump.GetCenter(),
                                       hump.GetDirection());
  return true;
}

}  // namespace rune_detector