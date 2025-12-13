#include "feature/rune_center/rune_center.h"

#include <math.h>

#include "common/geom_utils.hpp"
#include "feature/rune_center/rune_center_param.h"

using namespace std;
using namespace cv;

namespace rune_detector
{
RuneCenter::RuneCenter(const Point2f& center_in)
{
  auto contour = ContourWrapper<int>::MakeContour({center_in});
  auto center = center_in;
  auto width = rune_center_param.DEFAULT_SIDE;
  auto height = rune_center_param.DEFAULT_SIDE;
  // 更新角点信息
  vector<Point2f> corners(4);
  corners[0] = center + Point2f(static_cast<float>(-0.5 * width),
                                static_cast<float>(0.5 * height));
  corners[1] = center + Point2f(static_cast<float>(-0.5 * width),
                                static_cast<float>(-0.5 * height));
  corners[2] = center + Point2f(static_cast<float>(0.5 * width),
                                static_cast<float>(-0.5 * height));
  corners[3] =
      center + Point2f(static_cast<float>(0.5 * width), static_cast<float>(0.5 * height));

  // 初始化构造形状信息
  auto& image_info = getImageCache();
  image_info.setCenter(center);
  image_info.setWidth(width);
  image_info.setHeight(height);
  image_info.setCorners(corners);
  image_info.setContours(vector<ContourConstPtr>{contour});
}

RuneCenter::RuneCenter(const ContourConstPtr& contour, RotatedRect& /*rotated_rect*/)
{
  auto min_area_rect = contour->MinAreaRect();
  auto center = contour->Center();
  auto rrect_size = min_area_rect.size;
  auto width = max(rrect_size.width, rrect_size.height);
  auto height = min(rrect_size.width, rrect_size.height);
  // 更新角点信息
  vector<Point2f> corners(4);
  min_area_rect.points(corners.data());

  auto& image_info = getImageCache();
  image_info.setCenter(center);
  image_info.setWidth(width);
  image_info.setHeight(height);
  image_info.setCorners(corners);
  image_info.setContours(vector<ContourConstPtr>{contour});
}
/**
 * @brief 神符中心等级向量判断
 *
 * @param[in] hierarchy 所有的等级向量
 * @param[in] idx 指定的等级向量的下标
 * @return 等级结构是否满足要求
 */
static inline bool is_hierarchy_center(const vector<ContourConstPtr>& contours,
                                       const vector<Vec4i>& hierarchy, size_t idx)
{
  // h[idx] 必须存在若干并列轮廓，并且无父轮廓
  if ((hierarchy[idx][0] == -1 && hierarchy[idx][1] == -1) || hierarchy[idx][3] != -1)
  {
    return false;
  }
  if (hierarchy[idx][2] == -1)
  {
    return true;
  }
  else if (hierarchy[hierarchy[idx][2]][2] == -1)
  {
    RotatedRect outer = contours[idx]->FittedEllipse();
    Point2f outer_center = outer.center;
    Point2f inner_center = contours[hierarchy[idx][2]]->Center();
    auto dis = get_dist(inner_center, outer_center);
    auto size = (outer.size.width + outer.size.height) / 2.;
    // 偏移与最大直径的比值
    if (dis / size > rune_center_param.CENTER_CONCENTRICITY_RATIO)
    {
      return true;
    }
    if (contours[hierarchy[idx][2]]->Points().size() < 10)
    {
      return true;
    }
    return false;
  }
  return false;
}

/**
 * @brief 获取所有子轮廓
 *
 * @param[in] outer_idx 外轮廓的下标
 * @param[in] contours 所有轮廓
 * @param[in] hierarchy 所有等级向量
 * @param[out] sub_contours 子轮廓
 */
static inline void get_sub_contours(size_t outer_idx,
                                    const vector<ContourConstPtr>& contours,
                                    const vector<Vec4i>& hierarchy,
                                    vector<ContourConstPtr>& sub_contours)
{
  // 获取所有子轮廓下标
  vector<size_t> sub_contour_idxs{};
  get_all_sub_contours_idx(hierarchy, static_cast<int>(outer_idx), sub_contour_idxs);

  // 获取所有子轮廓
  for (auto idx : sub_contour_idxs)
  {
    sub_contours.push_back(contours[idx]);
  }
}

void RuneCenter::Find(
    std::vector<FeatureNodePtr>& centers, const std::vector<ContourConstPtr>& contours,
    const std::vector<cv::Vec4i>& hierarchy, const std::unordered_set<size_t>& mask,
    std::unordered_map<FeatureNodeConstPtr, unordered_set<size_t>>& used_contour_idxs)
{
  for (size_t i = 0; i < contours.size(); i++)
  {
    if (mask.find(i) != mask.end())
    {
      continue;
    }
    if (is_hierarchy_center(contours, hierarchy, i))
    {
      // 获取所有子轮廓
      vector<ContourConstPtr> sub_contours{};
      get_sub_contours(i, contours, hierarchy, sub_contours);
      auto p_center = RuneCenter::MakeFeature(contours[i], sub_contours);
      if (p_center != nullptr)
      {
        centers.push_back(p_center);
        used_contour_idxs[p_center] = {i};
      }
    }
  }
}

shared_ptr<RuneCenter> RuneCenter::MakeFeature(const Point2f& center, bool force)
{
  auto result = make_shared<RuneCenter>(center);
  if (force)
  {
  }

  return result;
}

shared_ptr<RuneCenter> RuneCenter::MakeFeature(
    const ContourConstPtr& contour, const std::vector<ContourConstPtr>& sub_contours)
{
  if (contour->Points().size() < 6)
  {
    return nullptr;
  }
  // init
  RotatedRect rotated_rect = contour->FittedEllipse();

  // 1.绝对面积判断
  if (contour->Area() < rune_center_param.MIN_AREA)
  {
    return nullptr;
  }
  if (contour->Area() > rune_center_param.MAX_AREA)
  {
    return nullptr;
  }

  // 2.比例判断
  float width = max(rotated_rect.size.width, rotated_rect.size.height);
  float height = min(rotated_rect.size.width, rotated_rect.size.height);
  float side_ratio = width / height;
  if (side_ratio > rune_center_param.MAX_SIDE_RATIO)
  {
    return nullptr;
  }
  if (side_ratio < rune_center_param.MIN_SIDE_RATIO)
  {
    return nullptr;
  }

  // 3. 圆形度判断
  float area = static_cast<float>(contour->Area());          // 计算轮廓面积
  float len = static_cast<float>(contour->Perimeter(true));  // 计算轮廓周长
  if (len == 0)
  {
    return nullptr;
  }
  float roundness = static_cast<float>(4 * CV_PI * area) / (len * len);  // 圆形度
  if (roundness < rune_center_param.MIN_ROUNDNESS)
  {
    return nullptr;
  }
  if (roundness > rune_center_param.MAX_ROUNDNESS)
  {
    return nullptr;
  }

  // 4. 父轮廓与子轮廓的面积比例判断
  // 获取子轮廓面积之和
  float total_sub_area = 0;
  for (const auto& sub_contour : sub_contours)
  {
    total_sub_area += static_cast<float>(sub_contour->Area());
  }
  float sub_area_ratio = total_sub_area / static_cast<float>(contour->Area());
  if (sub_area_ratio > rune_center_param.MAX_SUB_AREA_RATIO &&
      contour->Area() > rune_center_param.MIN_AREA_FOR_RATIO)
  {
    return nullptr;
  }

  // 5. 与凸包轮廓的面积比例判断
  float convex_area_ratio =
      static_cast<float>(contour->Area()) / static_cast<float>(contour->ConvexArea());
  if (convex_area_ratio < rune_center_param.MIN_CONVEX_AREA_RATIO &&
      contour->Area() > rune_center_param.MIN_AREA_FOR_RATIO)
  {
    return nullptr;
  }

  // 6. 最大凹陷面积判断
  float max_defect_area = 0;
  float max_defect_idx = -1;
  vector<Vec4i> defects;
  const auto& hull = contour->ConvexHullIdx();
  const auto& contour_points = contour->Points();
  if (contour_points.size() < 3 || hull.size() < 3)
  {
    return nullptr;
  }
  float defect_area_ratio = NAN;
  if (isContourConvex(contour_points))
  {
    convexityDefects(contour_points, hull, defects);
    for (size_t i = 0; i < defects.size(); i++)
    {
      Vec4i& d = defects[i];
      Point start = contour->Points()[d[0]];     // 凹陷起点
      Point end = contour->Points()[d[1]];       // 凹陷终点
      Point farthest = contour->Points()[d[2]];  // 凹陷最远点

      // 计算底边长和深度
      float depth = static_cast<float>(d[3]) / static_cast<float>(256.0);  // 深度
      float base_length = static_cast<float>(get_dist(start, end));        // 底边长

      // 近似计算凹陷面积
      double defect_area = base_length * depth / 2.0;
      if (defect_area > max_defect_area)
      {
        max_defect_area = static_cast<float>(defect_area);
        max_defect_idx = static_cast<float>(i);
      }
    }
    defect_area_ratio = max_defect_area / static_cast<float>(contour->Area());
    if (max_defect_idx != -1 &&
        defect_area_ratio > rune_center_param.MAX_DEFECT_AREA_RATIO &&
        contour->Area() > rune_center_param.MIN_AREA_FOR_RATIO)
    {
      return nullptr;
    }
  }
  return make_shared<RuneCenter>(contour, rotated_rect);
}

std::shared_ptr<RuneCenter> RuneCenter::MakeFeature(const PoseNode& center_to_cam,
                                                    const cv::Mat& k, const cv::Mat& d)
{
  vector<Point3f> points_mat_3d{};
  points_mat_3d.push_back(Point3f(0, 0, 0));
  // 重投影
  vector<Point2f> points_2d_reproject{};

  cv::Vec3d rvec, tvec;
  pose_to_open_cv(center_to_cam, rvec, tvec);

  projectPoints(points_mat_3d, rvec, tvec, k, d, points_2d_reproject);

  RuneCenterPtr result_ptr = MakeFeature(points_2d_reproject[0]);
  if (result_ptr)
  {
    auto& pose_info = result_ptr->getPoseCache();
    pose_info.getPoseNodes()[CoordFrame::CAMERA] = center_to_cam;
  }
  return result_ptr;
}

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>, std::vector<float>>
RuneCenter::GetPnpPoints() const
{
  vector<Point2f> points_2d;
  vector<Point3f> points_3d;
  vector<float> weights;
  points_2d.push_back(getImageCache().getCenter());
  points_3d.push_back(Point3f(0, 0, 0));

  if (points_2d.size() != points_3d.size())
  {
    std::ostringstream oss;
    oss << " points_2d and points_3d are not equal. (size = " << points_2d.size() << ", "
        << points_3d.size() << ")";
    throw std::runtime_error(oss.str());
  }
  weights.resize(points_2d.size(), 1.0);

  return make_tuple(points_2d, points_3d, weights);
}

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>, std::vector<float>>
RuneCenter::GetRelativePnpPoints() const
{
  auto [points_2d, points_3d, weights] = this->GetPnpPoints();
  vector<Point3f> relative_points_3d(points_3d.size());
  for (size_t i = 0; i < points_3d.size(); i++)
  {
    Matx31d points_3d_mat(points_3d[i].x, points_3d[i].y, points_3d[i].z);
    Matx31d relative_points_3d_mat{};
    relative_points_3d_mat =
        rune_center_param.ROTATION * points_3d_mat + rune_center_param.TRANSLATION;
    relative_points_3d[i] = Point3f(static_cast<float>(relative_points_3d_mat(0)),
                                    static_cast<float>(relative_points_3d_mat(1)),
                                    static_cast<float>(relative_points_3d_mat(2)));
  }
  return make_tuple(points_2d, relative_points_3d, weights);
}

void RuneCenter::DrawFeature(cv::Mat& image, const DrawConfigConstPtr& /*config*/) const
{
  const auto& image_info = this->getImageCache();
  // 使用默认半径进行绘制
  auto draw_circle = [&]() -> bool
  {
    if (!image_info.isSetCenter())
    {
      return false;
    }

    const auto& center = image_info.getCenter();
    auto radius = rune_center_draw_param.default_radius;
    auto color = rune_center_draw_param.color;
    cv::circle(image, center, static_cast<int>(radius), color, 2);
    return true;
  };

  // 使用轮廓椭圆进行绘制
  auto draw_contour_fit_ellipse = [&]() -> bool
  {
    if (!image_info.isSetContours() || image_info.getContours().empty())
    {
      return false;
    }
    const auto& contour = image_info.getContours().front();
    if (contour->Points().size() < 6)
    {
      return false;
    }
    auto ellipse = contour->FittedEllipse();
    cv::ellipse(image, ellipse, rune_center_draw_param.color, 2);
    return true;
  };

  do
  {
    if (draw_contour_fit_ellipse())
    {
      break;
    }
    else
    {
      draw_circle();
    }
  } while (0);
}

}  // namespace rune_detector