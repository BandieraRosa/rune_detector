#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "feature/rune_fan/rune_fan_hump.h"

namespace rune_detector
{

HumpDetector::HumpDetector(const std::vector<cv::Point>& contour, int _start_idx,
                           int _end_idx)
    : points_(contour.begin() + _start_idx, contour.begin() + _end_idx),
      start_idx_(_start_idx),
      end_idx_(_end_idx)
{
  direction_ = get_points_direction_vector(points_);
  ave_point_ = accumulate(points_.begin(), points_.end(), cv::Point2f(0, 0),
                          [](const cv::Point2f& sum, const cv::Point& p)
                          { return sum + static_cast<cv::Point2f>(p); });
  ave_point_ /= static_cast<float>(points_.size());
}

static inline double calculate_vertex_angle(const cv::Point2f& a, const cv::Point2f& b,
                                            const cv::Point2f& c)
{
  cv::Point2f v1 = b - a, v2 = c - a;
  double n1 = cv::norm(v1), n2 = cv::norm(v2);
  if (n1 < 1e-6 || n2 < 1e-6)
  {
    return 180.0;
  }
  double cos_theta = std::max(-1.0, std::min(1.0, v1.dot(v2) / (n1 * n2)));
  return acos(cos_theta) * 180.0 / CV_PI;
}

double HumpDetector::CheckCollinearity(const cv::Point2f& p1, const cv::Point2f& p2,
                                       const cv::Point2f& p3)
{
  double angle1 = calculate_vertex_angle(p1, p2, p3);
  double angle2 = calculate_vertex_angle(p2, p1, p3);
  double angle3 = calculate_vertex_angle(p3, p1, p2);
  return 180.0 - std::max({angle1, angle2, angle3});
}

double HumpDetector::CheckAlignment(const cv::Point2f& dir1, const cv::Point2f& dir2,
                                    const cv::Point2f& dir3)
{
  double a1 = atan2(dir1.y, dir1.x);
  double a2 = atan2(dir2.y, dir2.x);
  double a3 = atan2(dir3.y, dir3.x);
  double aligned[3] = {0, a2 - a1, a3 - a1};

  for (double& i : aligned)
  {
    while (i < -M_PI)
    {
      i += 2 * M_PI;
    }
  }
  for (double& i : aligned)
  {
    while (i > M_PI)
    {
      i -= 2 * M_PI;
    }
  }

  double mean = (aligned[0] + aligned[1] + aligned[2]) / 3.0;
  double var = 0.0;
  for (double i : aligned)
  {
    var += (i - mean) * (i - mean);
  }
  return sqrt(var / 3.0) * 180.0 / CV_PI;
}

}  // namespace rune_detector