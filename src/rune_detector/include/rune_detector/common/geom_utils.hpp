#pragma once

#include <cmath>
#include <opencv2/core.hpp>
#include <vector>

namespace rune_detector
{
constexpr double PI = CV_PI;
constexpr double DEG2RAD = PI / 180.0;
constexpr double RAD2DEG = 180.0 / PI;

// 角度转换

/**
 * @brief 度转弧度
 */
template <typename T>
constexpr T deg2rad(T deg)
{
  return deg * static_cast<T>(DEG2RAD);
}

/**
 * @brief 弧度转度
 */
template <typename T>
constexpr T rad2deg(T rad)
{
  return rad * static_cast<T>(RAD2DEG);
}

/**
 * @brief 将角度归一化到 [-π, π]
 */
inline float normalize_angle(float angle)
{
  while (angle > static_cast<float>(PI))
  {
    angle -= 2.0f * static_cast<float>(PI);
  }
  while (angle < -static_cast<float>(PI))
  {
    angle += 2.0f * static_cast<float>(PI);
  }
  return angle;
}

/**
 * @brief 计算两角度差（结果在 [-π, π]）
 */
inline float angle_diff(float a, float b) { return normalize_angle(a - b); }

/**
 * @brief 将角度归一化到指定周期（用于神符72°周期Roll角）
 * @param angle 当前角度（弧度）
 * @param last_angle 上一帧角度（弧度）
 * @param period 周期（弧度），默认72°
 */
inline float normalize_angle_to_period(float angle, float last_angle,
                                       float period = deg2rad(72.0f))
{
  float half = period / 2.0f;
  while (angle - last_angle > half)
  {
    angle -= period;
  }
  while (angle - last_angle < -half)
  {
    angle += period;
  }
  return angle;
}

// 2D向量运算

/**
 * @brief 向量角度（弧度，[-π, π]）
 */
inline float vector_angle(const cv::Point2f& v) { return std::atan2(v.y, v.x); }

/**
 * @brief 从p1指向p2的向量角度
 */
inline float vector_angle(const cv::Point2f& p1, const cv::Point2f& p2)
{
  return std::atan2(p2.y - p1.y, p2.x - p1.x);
}

/**
 * @brief 向量长度
 */
inline float norm(const cv::Point2f& v) { return std::sqrt(v.x * v.x + v.y * v.y); }

/**
 * @brief 两点距离
 */
inline float distance(const cv::Point2f& p1, const cv::Point2f& p2)
{
  return norm(p2 - p1);
}

/**
 * @brief 向量归一化
 */
inline cv::Point2f normalize(const cv::Point2f& v)
{
  float len = norm(v);
  return (len > 1e-6f) ? v / len : cv::Point2f(0, 0);
}

/**
 * @brief 2D叉积（返回z分量，用于判断左右侧）
 */
inline float cross(const cv::Point2f& v1, const cv::Point2f& v2)
{
  return v1.x * v2.y - v1.y * v2.x;
}

/**
 * @brief 点到直线的有符号距离
 * @param p 待测点
 * @param l1, l2 直线上两点
 */
inline float point_to_line_distance(const cv::Point2f& p, const cv::Point2f& l1,
                                    const cv::Point2f& l2)
{
  cv::Point2f line = l2 - l1;
  float len = norm(line);
  if (len < 1e-6f)
  {
    return distance(p, l1);
  }
  return cross(line, p - l1) / len;
}

// 轮廓工具

/**
 * @brief 获取轮廓点（支持循环索引）
 */
inline cv::Point2f contour_at(const std::vector<cv::Point>& contour, int idx)
{
  int n = static_cast<int>(contour.size());
  idx = ((idx % n) + n) % n;
  return cv::Point2f(contour[idx]);
}

/**
 * @brief 计算轮廓某点的切线方向角度
 * @param contour 轮廓
 * @param idx 点索引
 * @param step 计算步长（越大越平滑）
 */
inline float contour_tangent_angle(const std::vector<cv::Point>& contour, int idx,
                                   int step = 1)
{
  cv::Point2f prev = contour_at(contour, idx - step);
  cv::Point2f next = contour_at(contour, idx + step);
  return vector_angle(prev, next);
}

/**
 * @brief 计算轮廓所有点的切线角度序列
 */
inline std::vector<float> calc_contour_angles(const std::vector<cv::Point>& contour,
                                              int step = 1)
{
  std::vector<float> angles(contour.size());
  for (size_t i = 0; i < contour.size(); ++i)
  {
    angles[i] = contour_tangent_angle(contour, static_cast<int>(i), step);
  }
  return angles;
}

// 高斯滤波（角点提取用）

/**
 * @brief 创建高斯核
 */
inline std::vector<float> gaussian_kernel(int size, float sigma)
{
  std::vector<float> kernel(size);
  int half = size / 2;
  float sum = 0.0f;
  float s2 = 2.0f * sigma * sigma;

  for (int i = 0; i < size; ++i)
  {
    float x = static_cast<float>(i - half);
    kernel[i] = std::exp(-x * x / s2);
    sum += kernel[i];
  }
  for (auto& k : kernel)
  {
    k /= sum;
  }
  return kernel;
}

/**
 * @brief 一维循环卷积（用于角度序列平滑）
 */
inline std::vector<float> circular_convolve(const std::vector<float>& data,
                                            const std::vector<float>& kernel)
{
  int n = static_cast<int>(data.size());
  int k = static_cast<int>(kernel.size());
  int half = k / 2;
  std::vector<float> result(n);

  for (int i = 0; i < n; ++i)
  {
    float sum = 0.0f;
    for (int j = 0; j < k; ++j)
    {
      int idx = ((i - half + j) % n + n) % n;
      sum += data[idx] * kernel[j];
    }
    result[i] = sum;
  }
  return result;
}

/**
 * @brief 对角度序列进行高斯平滑
 * @param angles 输入角度序列
 * @param kernel_ratio 核大小比例（相对于序列长度）
 * @param sigma 高斯标准差
 */
inline std::vector<float> smooth_angles(const std::vector<float>& angles,
                                        float kernel_ratio = 0.04f, float sigma = 5.0f)
{
  int kernel_size = static_cast<int>(kernel_ratio * static_cast<float>(angles.size()));
  if (kernel_size < 3)
  {
    kernel_size = 3;
  }
  if (kernel_size % 2 == 0)
  {
    kernel_size++;  // 确保奇数
  }

  auto kernel = gaussian_kernel(kernel_size, sigma);
  return circular_convolve(angles, kernel);
}

// 梯度计算（角点提取用）

/**
 * @brief 计算角度序列的梯度（循环差分）
 */
inline std::vector<float> calc_angle_gradient(const std::vector<float>& angles)
{
  int n = static_cast<int>(angles.size());
  std::vector<float> grad(n);

  for (int i = 0; i < n; ++i)
  {
    int next = (i + 1) % n;
    grad[i] = angle_diff(angles[next], angles[i]);
  }
  return grad;
}

// 几何判断

/**
 * @brief 判断三点是否近似共线（角度法）
 * @param p1, p2, p3 三点（p2为中间点）
 * @param max_delta 最大角度偏差（弧度）
 */
inline bool is_collinear(const cv::Point2f& p1, const cv::Point2f& p2,
                         const cv::Point2f& p3, float max_delta = deg2rad(25.0f))
{
  float a1 = vector_angle(p1, p2);
  float a2 = vector_angle(p2, p3);
  return std::abs(angle_diff(a1, a2)) < max_delta;
}

/**
 * @brief 计算中点
 */
inline cv::Point2f midpoint(const cv::Point2f& p1, const cv::Point2f& p2)
{
  return (p1 + p2) * 0.5f;
}

}  // namespace rune_detector
