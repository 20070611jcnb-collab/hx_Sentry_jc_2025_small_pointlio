#include "scan_match_localizer/scan_match_localizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>

namespace scan_match_localizer
{

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kBig = 1.0e12;

// Felzenszwalb & Huttenlocher 的一维平方距离变换。
void distanceTransform1D(
  const double * f, int n, double * d, int * v, double * z)
{
  int k = 0;
  v[0] = 0;
  z[0] = -std::numeric_limits<double>::infinity();
  z[1] = std::numeric_limits<double>::infinity();
  for (int q = 1; q < n; ++q) {
    double s = ((f[q] + static_cast<double>(q) * q) -
      (f[v[k]] + static_cast<double>(v[k]) * v[k])) /
      (2.0 * q - 2.0 * v[k]);
    while (s <= z[k]) {
      --k;
      s = ((f[q] + static_cast<double>(q) * q) -
        (f[v[k]] + static_cast<double>(v[k]) * v[k])) /
        (2.0 * q - 2.0 * v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<double>::infinity();
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) {
      ++k;
    }
    const double dq = static_cast<double>(q - v[k]);
    d[q] = dq * dq + f[v[k]];
  }
}

double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

// 按恒定 body twist 外推 pose：yaw 用角速度积分，位置用半步旋转后的 body 速度积分。
Eigen::Isometry3d extrapolateOdomPose3D(const OdomSample & sample, double dt)
{
  Eigen::Isometry3d out = sample.pose3d;
  const Eigen::Vector3d omega = sample.angular_vel;
  const double angle = omega.norm() * dt;

  Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d dR_half = Eigen::Matrix3d::Identity();
  if (angle > 1.0e-9) {
    const Eigen::Vector3d axis = omega.normalized();
    dR = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    dR_half = Eigen::AngleAxisd(0.5 * angle, axis).toRotationMatrix();
  }

  out.linear() = sample.pose3d.linear() * dR;
  out.translation() = sample.pose3d.translation() +
    sample.pose3d.linear() * (dR_half * sample.linear_vel) * dt;
  return out;
}

}  // namespace

ScanMatchLocalizer::ScanMatchLocalizer(const rclcpp::NodeOptions & options)
: rclcpp::Node("scan_match_localizer", options)
{
  map_topic_ = declare_parameter<std::string>("map_topic", "/map");
  scan_topic_ = declare_parameter<std::string>("scan_topic", "/scan");
  odom_topic_ = declare_parameter<std::string>("odom_topic", "/Odometry");
  input_type_ = declare_parameter<std::string>("input_type", "scan");
  cloud_topic_ = declare_parameter<std::string>("cloud_topic", "/cloud_registered_full");
  map_frame_ = declare_parameter<std::string>("map_frame", "map");
  odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
  base_frame_ = declare_parameter<std::string>("base_frame", "base_footprint");
  tf_rate_ = declare_parameter<double>("tf_rate", 50.0);
  max_scan_rate_ = declare_parameter<double>("max_scan_rate", 10.0);
  reloc_rate_ = declare_parameter<double>("reloc_rate", 2.0);
  max_scan_points_ = declare_parameter<int>("max_scan_points", 200);
  min_range_ = declare_parameter<double>("min_range", 0.15);
  max_range_ = declare_parameter<double>("max_range", 10.0);
  min_valid_points_ = declare_parameter<int>("min_valid_points", 30);
  cloud_z_min_ = declare_parameter<double>("cloud_z_min", 0.10);
  cloud_z_max_ = declare_parameter<double>("cloud_z_max", 1.50);
  cloud_body_radius_ = declare_parameter<double>("cloud_body_radius", 0.45);
  occupied_threshold_ = declare_parameter<int>("occupied_threshold", 65);
  max_distance_ = declare_parameter<double>("max_distance", 1.0);
  sigma_hit_ = declare_parameter<double>("sigma_hit", 0.15);
  inlier_distance_ = declare_parameter<double>("inlier_distance", 0.4);
  trim_ratio_ = declare_parameter<double>("trim_ratio", 0.7);
  search_xy_ = declare_parameter<double>("search_xy", 0.4);
  search_yaw_ = declare_parameter<double>("search_yaw", 0.1);
  coarse_xy_step_ = declare_parameter<double>("coarse_xy_step", 0.04);
  coarse_yaw_step_ = declare_parameter<double>("coarse_yaw_step", 0.01);
  init_search_xy_ = declare_parameter<double>("init_search_xy", 1.0);
  init_search_yaw_ = declare_parameter<double>("init_search_yaw", 0.3);
  init_coarse_xy_step_ = declare_parameter<double>("init_coarse_xy_step", 0.08);
  init_coarse_yaw_step_ = declare_parameter<double>("init_coarse_yaw_step", 0.02);
  init_smoothing_alpha_ = declare_parameter<double>("init_smoothing_alpha", 1.0);
  fine_xy_radius_ = declare_parameter<double>("fine_xy_radius", 0.06);
  fine_yaw_radius_ = declare_parameter<double>("fine_yaw_radius", 0.015);
  fine_xy_step_ = declare_parameter<double>("fine_xy_step", 0.01);
  fine_yaw_step_ = declare_parameter<double>("fine_yaw_step", 0.0025);
  prior_xy_weight_ = declare_parameter<double>("prior_xy_weight", 0.05);
  prior_yaw_weight_ = declare_parameter<double>("prior_yaw_weight", 0.05);
  robot_radius_ = declare_parameter<double>("robot_radius", 0.2);
  footprint_penalty_ = declare_parameter<double>("footprint_penalty", 0.25);
  score_threshold_ = declare_parameter<double>("score_threshold", 0.4);
  inlier_ratio_threshold_ = declare_parameter<double>("inlier_ratio_threshold", 0.4);
  max_translation_correction_ = declare_parameter<double>("max_translation_correction", 0.12);
  max_yaw_correction_ = declare_parameter<double>("max_yaw_correction", 0.04);
  smoothing_alpha_ = declare_parameter<double>("smoothing_alpha", 0.25);
  min_correction_xy_ = declare_parameter<double>("min_correction_xy", 0.03);
  min_correction_yaw_ = declare_parameter<double>("min_correction_yaw", 0.01);
  converged_count_ = declare_parameter<int>("converged_count", 3);
  max_xy_spread_ = declare_parameter<double>("max_xy_spread", 0.12);
  max_yaw_spread_ = declare_parameter<double>("max_yaw_spread", 0.03);
  reloc_trigger_timeout_ = declare_parameter<double>("reloc_trigger_timeout", 0.6);
  odom_extrapolate_max_dt_ = declare_parameter<double>("odom_extrapolate_max_dt", 0.5);
  ambiguity_margin_ = declare_parameter<double>("ambiguity_margin", 0.03);
  ambiguity_min_distance_ = declare_parameter<double>("ambiguity_min_distance", 0.4);
  reloc_max_prior_deviation_ = declare_parameter<double>("reloc_max_prior_deviation", 1.2);
  publish_cloud_ = declare_parameter<bool>("publish_cloud", true);
  publish_pose_ = declare_parameter<bool>("publish_pose", true);
  log_period_ = declare_parameter<double>("log_period", 1.0);
  initial_map_to_odom_ = declare_parameter<std::vector<double>>(
    "initial_map_to_odom", {0.0, 0.0, 0.0});

  if (initial_map_to_odom_.size() != 3U) {
    RCLCPP_ERROR(
      get_logger(),
      "[scan_match] initial_map_to_odom must have 3 elements, using [0, 0, 0].");
    initial_map_to_odom_ = {0.0, 0.0, 0.0};
  }
  map_to_odom_.x = initial_map_to_odom_[0];
  map_to_odom_.y = initial_map_to_odom_[1];
  map_to_odom_.yaw = normalizeAngle(initial_map_to_odom_[2]);

  sigma_hit_ = std::max(1.0e-3, sigma_hit_);
  max_distance_ = std::max(0.05, max_distance_);
  inlier_distance_ = std::max(0.01, inlier_distance_);
  trim_ratio_ = std::clamp(trim_ratio_, 0.1, 1.0);
  search_xy_ = std::max(0.0, search_xy_);
  search_yaw_ = std::max(0.0, search_yaw_);
  coarse_xy_step_ = std::max(1.0e-3, coarse_xy_step_);
  coarse_yaw_step_ = std::max(1.0e-4, coarse_yaw_step_);
  init_search_xy_ = std::max(search_xy_, init_search_xy_);
  init_search_yaw_ = std::max(search_yaw_, init_search_yaw_);
  init_coarse_xy_step_ = std::max(1.0e-3, init_coarse_xy_step_);
  init_coarse_yaw_step_ = std::max(1.0e-4, init_coarse_yaw_step_);
  init_smoothing_alpha_ = std::clamp(init_smoothing_alpha_, 0.0, 1.0);
  fine_xy_radius_ = std::max(0.0, fine_xy_radius_);
  fine_yaw_radius_ = std::max(0.0, fine_yaw_radius_);
  fine_xy_step_ = std::max(1.0e-3, fine_xy_step_);
  fine_yaw_step_ = std::max(1.0e-5, fine_yaw_step_);
  prior_xy_weight_ = std::max(0.0, prior_xy_weight_);
  prior_yaw_weight_ = std::max(0.0, prior_yaw_weight_);
  robot_radius_ = std::max(0.0, robot_radius_);
  footprint_penalty_ = std::clamp(footprint_penalty_, 0.0, 1.0);
  max_translation_correction_ = std::max(0.01, max_translation_correction_);
  max_yaw_correction_ = std::max(1.0e-3, max_yaw_correction_);
  smoothing_alpha_ = std::clamp(smoothing_alpha_, 0.0, 1.0);
  min_correction_xy_ = std::max(0.0, min_correction_xy_);
  min_correction_yaw_ = std::max(0.0, min_correction_yaw_);
  if (cloud_z_max_ <= cloud_z_min_) {
    std::swap(cloud_z_min_, cloud_z_max_);
  }
  cloud_body_radius_ = std::max(0.0, cloud_body_radius_);
  converged_count_ = std::max(1, converged_count_);
  max_xy_spread_ = std::max(0.0, max_xy_spread_);
  max_yaw_spread_ = std::max(0.0, max_yaw_spread_);
  reloc_trigger_timeout_ = std::max(0.0, reloc_trigger_timeout_);
  odom_extrapolate_max_dt_ = std::max(0.0, odom_extrapolate_max_dt_);
  ambiguity_margin_ = std::max(0.0, ambiguity_margin_);
  ambiguity_min_distance_ = std::max(0.05, ambiguity_min_distance_);
  reloc_max_prior_deviation_ = std::max(0.0, reloc_max_prior_deviation_);
  tf_rate_ = std::max(1.0, tf_rate_);
  max_scan_rate_ = std::max(0.2, max_scan_rate_);
  reloc_rate_ = std::max(0.2, reloc_rate_);
  max_scan_points_ = std::max(10, max_scan_points_);
  min_valid_points_ = std::max(1, min_valid_points_);
  log_period_ = std::max(0.0, log_period_);

  exp_lut_.resize(static_cast<size_t>(lut_size_));
  for (int i = 0; i < lut_size_; ++i) {
    const double d = max_distance_ * static_cast<double>(i) /
      static_cast<double>(lut_size_ - 1);
    exp_lut_[static_cast<size_t>(i)] = static_cast<float>(
      std::exp(-0.5 * d * d / (sigma_hit_ * sigma_hit_)));
  }

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_buffer_->setUsingDedicatedThread(true);
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this, true);
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  sensor_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  odom_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  map_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  tf_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions sensor_opts;
  sensor_opts.callback_group = sensor_cb_group_;
  rclcpp::SubscriptionOptions odom_opts;
  odom_opts.callback_group = odom_cb_group_;
  rclcpp::SubscriptionOptions map_opts;
  map_opts.callback_group = map_cb_group_;

  auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
  map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
    map_topic_, map_qos,
    std::bind(&ScanMatchLocalizer::mapCallback, this, std::placeholders::_1), map_opts);
  const bool use_cloud =
    input_type_ == "cloud" || input_type_ == "pointcloud" ||
    input_type_ == "cloud_registered";
  if (use_cloud) {
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ScanMatchLocalizer::cloudCallback, this, std::placeholders::_1),
      sensor_opts);
  } else {
    if (input_type_ != "scan" && input_type_ != "laser") {
      RCLCPP_WARN(
        get_logger(), "[scan_match] unknown input_type '%s', using LaserScan.",
        input_type_.c_str());
    }
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ScanMatchLocalizer::scanCallback, this, std::placeholders::_1),
      sensor_opts);
  }
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_, rclcpp::QoS(50),
    std::bind(&ScanMatchLocalizer::odomCallback, this, std::placeholders::_1), odom_opts);

  cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/scan_match/cloud", 10);
  pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/scan_match/pose", 10);

  const auto tf_period = std::chrono::milliseconds(
    static_cast<int>(1000.0 / tf_rate_));
  tf_timer_ = create_wall_timer(
    std::max(tf_period, std::chrono::milliseconds(1)),
    std::bind(&ScanMatchLocalizer::tfTimerCallback, this), tf_cb_group_);

  RCLCPP_INFO(
    get_logger(),
    "[scan_match] map=%s input=%s(%s) odom=%s frames=%s->%s->%s, "
    "search=%.2fm/%.2frad, initial map->odom=(%.2f, %.2f, %.3f)",
    map_topic_.c_str(), input_type_.c_str(),
    use_cloud ? cloud_topic_.c_str() : scan_topic_.c_str(),
    odom_topic_.c_str(),
    map_frame_.c_str(), odom_frame_.c_str(), base_frame_.c_str(),
    search_xy_, search_yaw_, map_to_odom_.x, map_to_odom_.y, map_to_odom_.yaw);
}

void ScanMatchLocalizer::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  const size_t expected = static_cast<size_t>(msg->info.width) *
    static_cast<size_t>(msg->info.height);
  if (msg->info.width == 0U || msg->info.height == 0U ||
    msg->info.resolution <= 0.0 || msg->data.size() != expected)
  {
    RCLCPP_ERROR(
      get_logger(), "[scan_match] invalid map: %ux%u res=%.3f data=%zu",
      msg->info.width, msg->info.height, msg->info.resolution, msg->data.size());
    return;
  }

  std::lock_guard<std::mutex> map_lock(map_mutex_);

  width_ = static_cast<int>(msg->info.width);
  height_ = static_cast<int>(msg->info.height);
  resolution_ = msg->info.resolution;
  origin_x_ = msg->info.origin.position.x;
  origin_y_ = msg->info.origin.position.y;
  origin_yaw_ = normalizeAngle(yawFromQuaternion(msg->info.origin.orientation));
  origin_cos_ = std::cos(origin_yaw_);
  origin_sin_ = std::sin(origin_yaw_);

  if (!msg->header.frame_id.empty() && msg->header.frame_id != map_frame_) {
    RCLCPP_WARN(
      get_logger(), "[scan_match] map frame is '%s', expected '%s'.",
      msg->header.frame_id.c_str(), map_frame_.c_str());
  }

  occupied_.assign(expected, 0U);
  size_t occupied_count = 0;
  for (size_t i = 0; i < expected; ++i) {
    if (msg->data[i] >= occupied_threshold_) {
      occupied_[i] = 1U;
      ++occupied_count;
    }
  }

  const auto t0 = std::chrono::steady_clock::now();
  buildDistanceField();
  const auto t1 = std::chrono::steady_clock::now();
  const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  map_ready_.store(true, std::memory_order_release);
  RCLCPP_INFO(
    get_logger(),
    "[scan_match] map %dx%d res=%.3f m origin=(%.2f, %.2f, %.3f) "
    "occupied=%zu distance_field=%.1f ms",
    width_, height_, resolution_, origin_x_, origin_y_, origin_yaw_,
    occupied_count, dt_ms);
}

void ScanMatchLocalizer::buildDistanceField()
{
  const size_t cells = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  std::vector<double> f(cells, kBig);
  for (size_t i = 0; i < cells; ++i) {
    if (occupied_[i] != 0U) {
      f[i] = 0.0;
    }
  }

  std::vector<double> g(cells, kBig);
  const int max_dim = std::max(width_, height_);
  std::vector<double> line_in(static_cast<size_t>(max_dim), kBig);
  std::vector<double> line_out(static_cast<size_t>(max_dim), kBig);
  std::vector<int> v(static_cast<size_t>(max_dim) + 1U, 0);
  std::vector<double> z(static_cast<size_t>(max_dim) + 1U, 0.0);

  for (int x = 0; x < width_; ++x) {
    for (int y = 0; y < height_; ++y) {
      line_in[static_cast<size_t>(y)] = f[static_cast<size_t>(y) * width_ + x];
    }
    distanceTransform1D(line_in.data(), height_, line_out.data(), v.data(), z.data());
    for (int y = 0; y < height_; ++y) {
      g[static_cast<size_t>(y) * width_ + x] = line_out[static_cast<size_t>(y)];
    }
  }

  distance_field_.assign(cells, static_cast<float>(max_distance_));
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      line_in[static_cast<size_t>(x)] = g[static_cast<size_t>(y) * width_ + x];
    }
    distanceTransform1D(line_in.data(), width_, line_out.data(), v.data(), z.data());
    for (int x = 0; x < width_; ++x) {
      const double d = std::sqrt(line_out[static_cast<size_t>(x)]) * resolution_;
      distance_field_[static_cast<size_t>(y) * width_ + x] =
        static_cast<float>(std::min(d, max_distance_));
    }
  }
}

void ScanMatchLocalizer::worldToGrid(
  double wx, double wy, double & gx, double & gy) const
{
  const double dx = wx - origin_x_;
  const double dy = wy - origin_y_;
  gx = (origin_cos_ * dx + origin_sin_ * dy) / resolution_ - 0.5;
  gy = (-origin_sin_ * dx + origin_cos_ * dy) / resolution_ - 0.5;
}

float ScanMatchLocalizer::sampleDistance(double gx, double gy) const
{
  if (gx < -0.5 || gy < -0.5 ||
    gx > static_cast<double>(width_) - 0.5 ||
    gy > static_cast<double>(height_) - 0.5)
  {
    return static_cast<float>(max_distance_);
  }

  const double cx = std::clamp(gx, 0.0, static_cast<double>(width_ - 1));
  const double cy = std::clamp(gy, 0.0, static_cast<double>(height_ - 1));
  const int x0 = static_cast<int>(std::floor(cx));
  const int y0 = static_cast<int>(std::floor(cy));
  const int x1 = std::min(x0 + 1, width_ - 1);
  const int y1 = std::min(y0 + 1, height_ - 1);
  const double fx = cx - x0;
  const double fy = cy - y0;

  const double d00 = distance_field_[static_cast<size_t>(y0) * width_ + x0];
  const double d10 = distance_field_[static_cast<size_t>(y0) * width_ + x1];
  const double d01 = distance_field_[static_cast<size_t>(y1) * width_ + x0];
  const double d11 = distance_field_[static_cast<size_t>(y1) * width_ + x1];
  const double d = (1.0 - fx) * (1.0 - fy) * d00 + fx * (1.0 - fy) * d10 +
    (1.0 - fx) * fy * d01 + fx * fy * d11;
  return static_cast<float>(d);
}

bool ScanMatchLocalizer::isOccupied(double wx, double wy) const
{
  double gx = 0.0;
  double gy = 0.0;
  worldToGrid(wx, wy, gx, gy);
  const int ix = static_cast<int>(std::floor(gx + 0.5));
  const int iy = static_cast<int>(std::floor(gy + 0.5));
  if (ix < 0 || iy < 0 || ix >= width_ || iy >= height_) {
    return false;
  }
  return occupied_[static_cast<size_t>(iy) * width_ + ix] != 0U;
}

bool ScanMatchLocalizer::footprintOccupied(double wx, double wy) const
{
  if (isOccupied(wx, wy)) {
    return true;
  }
  if (robot_radius_ <= 0.0) {
    return false;
  }
  constexpr int kSamples = 8;
  for (int i = 0; i < kSamples; ++i) {
    const double a = 2.0 * kPi * static_cast<double>(i) / kSamples;
    if (isOccupied(
        wx + robot_radius_ * std::cos(a),
        wy + robot_radius_ * std::sin(a)))
    {
      return true;
    }
  }
  return false;
}

float ScanMatchLocalizer::lookupLikelihood(float distance) const
{
  if (distance >= static_cast<float>(max_distance_)) {
    return 0.0F;
  }
  const float t = distance / static_cast<float>(max_distance_);
  int idx = static_cast<int>(t * static_cast<float>(lut_size_ - 1));
  idx = std::clamp(idx, 0, lut_size_ - 1);
  return exp_lut_[static_cast<size_t>(idx)];
}

bool ScanMatchLocalizer::loadScanInBase(
  const sensor_msgs::msg::LaserScan & scan, std::vector<Eigen::Vector2f> & points)
{
  Eigen::Isometry3d base_scan = Eigen::Isometry3d::Identity();
  if (!scan.header.frame_id.empty() && scan.header.frame_id != base_frame_) {
    auto cached = scan_tf_cache_.find(scan.header.frame_id);
    if (cached == scan_tf_cache_.end()) {
      // 雷达相对底盘的安装是静态的，查一次并缓存，避免每帧都走 TF。
      const tf2::BufferCore & tf_core = *tf_buffer_;
      try {
        const auto tf = tf_core.lookupTransform(
          base_frame_, scan.header.frame_id, tf2::TimePointZero);
        cached = scan_tf_cache_.emplace(
          scan.header.frame_id, tf2::transformToEigen(tf)).first;
        RCLCPP_INFO(
          get_logger(), "[scan_match] cached static transform %s -> %s",
          scan.header.frame_id.c_str(), base_frame_.c_str());
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[scan_match] cannot transform scan '%s' -> '%s': %s",
          scan.header.frame_id.c_str(), base_frame_.c_str(), ex.what());
        return false;
      }
    }
    base_scan = cached->second;
  }

  const double range_min = std::max(static_cast<double>(scan.range_min), min_range_);
  const double range_max = std::min(static_cast<double>(scan.range_max), max_range_);
  points.clear();
  points.reserve(scan.ranges.size());
  for (size_t i = 0; i < scan.ranges.size(); ++i) {
    const double r = scan.ranges[i];
    if (!std::isfinite(r) || r < range_min || r > range_max) {
      continue;
    }
    const double angle = scan.angle_min + static_cast<double>(i) * scan.angle_increment;
    Eigen::Vector3d p(r * std::cos(angle), r * std::sin(angle), 0.0);
    p = base_scan * p;
    points.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()));
  }

  if (static_cast<int>(points.size()) > max_scan_points_) {
    const int stride = (static_cast<int>(points.size()) + max_scan_points_ - 1) /
      max_scan_points_;
    std::vector<Eigen::Vector2f> sampled;
    sampled.reserve(static_cast<size_t>(max_scan_points_));
    for (size_t i = 0; i < points.size();
      i += static_cast<size_t>(stride))
    {
      sampled.push_back(points[i]);
    }
    points.swap(sampled);
  }

  if (static_cast<int>(points.size()) < min_valid_points_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[scan_match] too few valid scan points: %zu (min %d)",
      points.size(), min_valid_points_);
    return false;
  }
  return true;
}

bool ScanMatchLocalizer::loadCloudInBase(
  const sensor_msgs::msg::PointCloud2 & cloud, const Eigen::Isometry3d & odom_base,
  std::vector<Eigen::Vector2f> & points)
{
  if (!cloud.header.frame_id.empty() && cloud.header.frame_id != odom_frame_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[scan_match] cloud frame is '%s', expected '%s'.",
      cloud.header.frame_id.c_str(), odom_frame_.c_str());
    return false;
  }

  const Eigen::Isometry3d base_odom = odom_base.inverse();
  const double min_r = std::max(cloud_body_radius_, min_range_);
  points.clear();
  const size_t count =
    static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height);
  points.reserve(std::min<size_t>(
    count, static_cast<size_t>(max_scan_points_) * 4U));

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");
  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    const float x = *iter_x;
    const float y = *iter_y;
    const float z = *iter_z;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    const Eigen::Vector3d p_base = base_odom * Eigen::Vector3d(x, y, z);
    if (p_base.z() < cloud_z_min_ || p_base.z() > cloud_z_max_) {
      continue;
    }
    const double r = std::hypot(p_base.x(), p_base.y());
    if (r < min_r || r > max_range_) {
      continue;
    }
    points.emplace_back(
      static_cast<float>(p_base.x()), static_cast<float>(p_base.y()));
  }

  if (static_cast<int>(points.size()) > max_scan_points_) {
    const int stride = (static_cast<int>(points.size()) + max_scan_points_ - 1) /
      max_scan_points_;
    std::vector<Eigen::Vector2f> sampled;
    sampled.reserve(static_cast<size_t>(max_scan_points_));
    for (size_t i = 0; i < points.size();
      i += static_cast<size_t>(stride))
    {
      sampled.push_back(points[i]);
    }
    points.swap(sampled);
  }

  if (static_cast<int>(points.size()) < min_valid_points_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[scan_match] too few cloud points after filtering: %zu (min %d)",
      points.size(), min_valid_points_);
    return false;
  }
  return true;
}

bool ScanMatchLocalizer::lookupOdomBase(
  const rclcpp::Time & stamp, Pose2D & odom_base)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (odom_history_.empty()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[scan_match] no odometry on %s yet.", odom_topic_.c_str());
    return false;
  }

  auto it = std::lower_bound(
    odom_history_.begin(), odom_history_.end(), stamp,
    [](const OdomSample & entry, const rclcpp::Time & t) {
      return entry.stamp < t;
    });

  if (it == odom_history_.begin()) {
    if (std::abs((stamp - it->stamp).seconds()) > 0.5) {
      return false;
    }
    odom_base = it->pose;
    return true;
  }
  if (it == odom_history_.end()) {
    const auto & last = odom_history_.back();
    if (std::abs((stamp - last.stamp).seconds()) > 0.5) {
      return false;
    }
    odom_base = last.pose;
    return true;
  }

  const auto & next = *it;
  const auto & prev = *(it - 1);
  const double dt = (next.stamp - prev.stamp).seconds();
  if (dt <= 1.0e-6) {
    odom_base = next.pose;
    return true;
  }
  const double alpha = std::clamp(
    (stamp - prev.stamp).seconds() / dt, 0.0, 1.0);
  odom_base = blend(prev.pose, next.pose, alpha);
  return true;
}

bool ScanMatchLocalizer::lookupOdomPose3D(
  const rclcpp::Time & stamp, Eigen::Isometry3d & odom_base)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (odom_history_.empty()) {
    return false;
  }
  auto it = std::lower_bound(
    odom_history_.begin(), odom_history_.end(), stamp,
    [](const OdomSample & entry, const rclcpp::Time & t) {
      return entry.stamp < t;
    });

  const OdomSample * best = nullptr;
  double best_dt = std::numeric_limits<double>::infinity();
  auto consider = [&](const OdomSample & sample) {
      const double dt = std::abs((sample.stamp - stamp).seconds());
      if (dt < best_dt) {
        best_dt = dt;
        best = &sample;
      }
    };
  if (it != odom_history_.end()) {
    consider(*it);
  }
  if (it != odom_history_.begin()) {
    consider(*(it - 1));
  }
  if (best == nullptr || best_dt > 0.25) {
    // odom 时间戳空洞：用 stamp 之前最近样本的 body twist 前向外推。
    // 云台转速恒定时 yaw 外推足够准，避免整帧被丢弃。
    const OdomSample * anchor = nullptr;
    if (it == odom_history_.end()) {
      anchor = &odom_history_.back();
    } else if (it != odom_history_.begin()) {
      anchor = &(*(it - 1));
    }
    if (anchor != nullptr && stamp >= anchor->stamp) {
      const double gap = (stamp - anchor->stamp).seconds();
      if (gap <= odom_extrapolate_max_dt_) {
        odom_base = extrapolateOdomPose3D(*anchor, gap);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[scan_match] odometry gap %.3f s, extrapolate from last sample.", gap);
        return true;
      }
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[scan_match] no odometry close to cloud stamp (dt=%.3f s).", best_dt);
    return false;
  }
  odom_base = best->pose3d;
  return true;
}

void ScanMatchLocalizer::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  if (!msg->header.frame_id.empty() && msg->header.frame_id != odom_frame_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "[scan_match] odometry frame is '%s', expected '%s'.",
      msg->header.frame_id.c_str(), odom_frame_.c_str());
  }
  rclcpp::Time stamp(msg->header.stamp);
  if (stamp.nanoseconds() == 0) {
    stamp = now();
  }

  Pose2D pose;
  pose.x = msg->pose.pose.position.x;
  pose.y = msg->pose.pose.position.y;
  pose.yaw = normalizeAngle(yawFromQuaternion(msg->pose.pose.orientation));

  OdomSample sample;
  sample.stamp = stamp;
  sample.pose = pose;
  sample.pose3d.translation() = Eigen::Vector3d(
    msg->pose.pose.position.x,
    msg->pose.pose.position.y,
    msg->pose.pose.position.z);
  Eigen::Quaterniond q(
    msg->pose.pose.orientation.w,
    msg->pose.pose.orientation.x,
    msg->pose.pose.orientation.y,
    msg->pose.pose.orientation.z);
  if (q.norm() > 1.0e-6) {
    q.normalize();
    sample.pose3d.linear() = q.toRotationMatrix();
  }
  sample.linear_vel = Eigen::Vector3d(
    msg->twist.twist.linear.x,
    msg->twist.twist.linear.y,
    msg->twist.twist.linear.z);
  sample.angular_vel = Eigen::Vector3d(
    msg->twist.twist.angular.x,
    msg->twist.twist.angular.y,
    msg->twist.twist.angular.z);

  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!odom_history_.empty() && stamp < odom_history_.back().stamp) {
    odom_history_.clear();
  }
  odom_history_.push_back(std::move(sample));
  while (odom_history_.size() > 200U) {
    odom_history_.pop_front();
  }
}

void ScanMatchLocalizer::collectDistances(
  const Pose2D & pose, const std::vector<Eigen::Vector2f> & points,
  std::vector<float> & distances) const
{
  const double c = std::cos(pose.yaw);
  const double s = std::sin(pose.yaw);
  distances.clear();
  distances.reserve(points.size());
  for (const auto & p : points) {
    const double wx = pose.x + c * p.x() - s * p.y();
    const double wy = pose.y + s * p.x() + c * p.y();
    double gx = 0.0;
    double gy = 0.0;
    worldToGrid(wx, wy, gx, gy);
    distances.push_back(sampleDistance(gx, gy));
  }
}

// 候选位姿搜索是纯只读计算，按 yaw 切片并行；每线程保留若干互不相邻的候选峰再归并。
// second_score/second_pose 用于地形歧义判定（并排狗洞等相似结构）。
void ScanMatchLocalizer::searchWindow(
  const std::vector<Eigen::Vector2f> & points, const Pose2D & center,
  double xy_radius, double xy_step, double yaw_radius, double yaw_step,
  const Pose2D & prior, Pose2D & best, double & best_score,
  double & second_score, Pose2D & second_pose) const
{
  const int n_xy = static_cast<int>(std::floor(xy_radius / xy_step + 0.5));
  const int n_yaw = static_cast<int>(std::floor(yaw_radius / yaw_step + 0.5));
  const double inv_points = 1.0 / static_cast<double>(points.size());

  struct Peak
  {
    Pose2D pose;
    double score{-std::numeric_limits<double>::infinity()};
  };
  constexpr int kMaxPeaks = 4;
  const double min_sep = ambiguity_min_distance_;

  // 非极大抑制：相距小于 min_sep 的候选视为同一个峰，只保留更高分。
  auto insert_peak = [min_sep](Peak * peaks, int & count, const Peak & cand) {
      for (int i = 0; i < count; ++i) {
        if (std::hypot(cand.pose.x - peaks[i].pose.x, cand.pose.y - peaks[i].pose.y) < min_sep) {
          if (cand.score > peaks[i].score) {
            peaks[i] = cand;
          }
          return;
        }
      }
      if (count < kMaxPeaks) {
        peaks[count++] = cand;
        return;
      }
      int lowest = 0;
      for (int i = 1; i < count; ++i) {
        if (peaks[i].score < peaks[lowest].score) {
          lowest = i;
        }
      }
      if (cand.score > peaks[lowest].score) {
        peaks[lowest] = cand;
      }
    };

  Peak global_peaks[kMaxPeaks];
  int global_count = 0;

#pragma omp parallel
  {
    std::vector<Eigen::Vector2f> rotated(points.size());
    Peak local_peaks[kMaxPeaks];
    int local_count = 0;

#pragma omp for schedule(static)
    for (int iyaw = -n_yaw; iyaw <= n_yaw; ++iyaw) {
      const double yaw = center.yaw + static_cast<double>(iyaw) * yaw_step;
      const double c = std::cos(yaw);
      const double s = std::sin(yaw);
      for (size_t k = 0; k < points.size(); ++k) {
        rotated[k] = Eigen::Vector2f(
          static_cast<float>(c * points[k].x() - s * points[k].y()),
          static_cast<float>(s * points[k].x() + c * points[k].y()));
      }

      for (int ix = -n_xy; ix <= n_xy; ++ix) {
        const double x = center.x + static_cast<double>(ix) * xy_step;
        for (int iy = -n_xy; iy <= n_xy; ++iy) {
          const double y = center.y + static_cast<double>(iy) * xy_step;

          double sum = 0.0;
          for (size_t k = 0; k < points.size(); ++k) {
            double gx = 0.0;
            double gy = 0.0;
            worldToGrid(x + rotated[k].x(), y + rotated[k].y(), gx, gy);
            sum += lookupLikelihood(sampleDistance(gx, gy));
          }
          double score = sum * inv_points;
          if (footprintOccupied(x, y)) {
            score *= (1.0 - footprint_penalty_);
          }

          const double dx = x - prior.x;
          const double dy = y - prior.y;
          const double dyaw = normalizeAngle(yaw - prior.yaw);
          score -= prior_xy_weight_ * (dx * dx + dy * dy) +
            prior_yaw_weight_ * dyaw * dyaw;

          if (local_count == kMaxPeaks) {
            double lowest = local_peaks[0].score;
            for (int i = 1; i < kMaxPeaks; ++i) {
              lowest = std::min(lowest, local_peaks[i].score);
            }
            if (score <= lowest) {
              continue;
            }
          }
          Peak cand;
          cand.pose.x = x;
          cand.pose.y = y;
          cand.pose.yaw = yaw;
          cand.score = score;
          insert_peak(local_peaks, local_count, cand);
        }
      }
    }

#pragma omp critical
    {
      for (int i = 0; i < local_count; ++i) {
        insert_peak(global_peaks, global_count, local_peaks[i]);
      }
    }
  }

  second_score = -std::numeric_limits<double>::infinity();
  second_pose = prior;
  if (global_count == 0) {
    return;
  }

  int best_idx = 0;
  for (int i = 1; i < global_count; ++i) {
    if (global_peaks[i].score > global_peaks[best_idx].score) {
      best_idx = i;
    }
  }
  for (int i = 0; i < global_count; ++i) {
    if (i == best_idx) {
      continue;
    }
    if (global_peaks[i].score > second_score) {
      second_score = global_peaks[i].score;
      second_pose = global_peaks[i].pose;
    }
  }
  if (global_peaks[best_idx].score > best_score) {
    best_score = global_peaks[best_idx].score;
    best = global_peaks[best_idx].pose;
  }
}

void ScanMatchLocalizer::scanCallback(
  const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  if (!map_ready_.load(std::memory_order_acquire)) {
    return;
  }

  rclcpp::Time stamp(msg->header.stamp);
  if (stamp.nanoseconds() == 0) {
    stamp = now();
  }
  if (has_last_scan_) {
    const double dt = (stamp - last_scan_stamp_).seconds();
    const double rate = converged_ ? max_scan_rate_ : reloc_rate_;
    if (dt >= 0.0 && dt < 1.0 / rate) {
      return;
    }
  }
  last_scan_stamp_ = stamp;
  has_last_scan_ = true;

  std::vector<Eigen::Vector2f> points;
  if (!loadScanInBase(*msg, points)) {
    return;
  }

  Pose2D odom_base;
  if (!lookupOdomBase(stamp, odom_base)) {
    return;
  }

  processPoints(stamp, points, odom_base);
}

void ScanMatchLocalizer::cloudCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  if (!map_ready_.load(std::memory_order_acquire)) {
    return;
  }

  rclcpp::Time stamp(msg->header.stamp);
  if (stamp.nanoseconds() == 0) {
    stamp = now();
  }
  if (has_last_scan_) {
    const double dt = (stamp - last_scan_stamp_).seconds();
    const double rate = converged_ ? max_scan_rate_ : reloc_rate_;
    if (dt >= 0.0 && dt < 1.0 / rate) {
      return;
    }
  }
  last_scan_stamp_ = stamp;
  has_last_scan_ = true;

  Eigen::Isometry3d odom_base3d;
  if (!lookupOdomPose3D(stamp, odom_base3d)) {
    return;
  }

  std::vector<Eigen::Vector2f> points;
  if (!loadCloudInBase(*msg, odom_base3d, points)) {
    return;
  }

  Pose2D odom_base;
  odom_base.x = odom_base3d.translation().x();
  odom_base.y = odom_base3d.translation().y();
  odom_base.yaw = normalizeAngle(
    std::atan2(odom_base3d(1, 0), odom_base3d(0, 0)));
  processPoints(stamp, points, odom_base);
}

void ScanMatchLocalizer::processPoints(
  const rclcpp::Time & stamp, const std::vector<Eigen::Vector2f> & points,
  const Pose2D & odom_base)
{
  Pose2D current_map_to_odom;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_map_to_odom = map_to_odom_;
  }
  const Pose2D prior = compose(current_map_to_odom, odom_base);
  Pose2D best = prior;
  double best_score = -std::numeric_limits<double>::infinity();

  // 收敛后长时间没有成功匹配（map->odom 停止按正常频率更新），说明真实误差
  // 已超出精细匹配的搜索/修正上限，继续用精细窗口只会一直拒绝。
  // 此时主动降级为大窗口重定位；重新稳定收敛后由 updateConvergence 自动切回精细。
  // 要求至少有一次拒帧，避免低帧率下把"帧间隔本身较长"误判为丢失定位。
  if (converged_ && has_last_accept_ && reject_streak_ > 0) {
    const double since_accept = (stamp - last_accept_stamp_).seconds();
    if (since_accept > reloc_trigger_timeout_) {
      converged_ = false;
      relocalizing_ = true;
      reject_streak_ = 0;
      history_.clear();
      has_last_scan_ = false;  // 下一帧不再受跟踪频率限制，加快重捕获
      RCLCPP_WARN(
        get_logger(),
        "[scan_match] no accepted match for %.2f s, switch to relocalization.",
        since_accept);
    }
  }

  // 未收敛时用大窗口 + 大步长快速捕获初始错位；收敛后切回小窗口跟踪。
  const bool initializing = !converged_;
  const double xy_radius = initializing ? init_search_xy_ : search_xy_;
  const double yaw_radius = initializing ? init_search_yaw_ : search_yaw_;
  const double xy_step = initializing ? init_coarse_xy_step_ : coarse_xy_step_;
  const double yaw_step = initializing ? init_coarse_yaw_step_ : coarse_yaw_step_;
  const double max_corr_xy = initializing ?
    std::max(init_search_xy_, max_translation_correction_) :
    max_translation_correction_;
  const double max_corr_yaw = initializing ?
    std::max(init_search_yaw_, max_yaw_correction_) :
    max_yaw_correction_;
  const double alpha = initializing ? init_smoothing_alpha_ : smoothing_alpha_;

  double score = 0.0;
  double mean_distance = 0.0;
  double inlier_ratio = 0.0;
  double coarse_best_score = -std::numeric_limits<double>::infinity();
  double second_score = -std::numeric_limits<double>::infinity();
  Pose2D second_pose = prior;
  {
    // 距离场/栅格只读；map 回调在重建时拿同一把锁。
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    searchWindow(
      points, prior, xy_radius, xy_step, yaw_radius, yaw_step,
      prior, best, best_score, second_score, second_pose);
    coarse_best_score = best_score;
    double fine_second_score = -std::numeric_limits<double>::infinity();
    Pose2D fine_second_pose = prior;
    searchWindow(
      points, best, fine_xy_radius_, fine_xy_step_, fine_yaw_radius_, fine_yaw_step_,
      prior, best, best_score, fine_second_score, fine_second_pose);

    std::vector<float> distances;
    collectDistances(best, points, distances);
    if (distances.empty()) {
      return;
    }
    std::sort(distances.begin(), distances.end());
    const size_t keep = std::max<size_t>(
      1U, static_cast<size_t>(static_cast<double>(distances.size()) * trim_ratio_));
    double sum_likelihood = 0.0;
    double sum_distance = 0.0;
    for (size_t i = 0; i < keep; ++i) {
      sum_likelihood += lookupLikelihood(distances[i]);
      sum_distance += distances[i];
    }
    score = sum_likelihood / static_cast<double>(keep);
    mean_distance = sum_distance / static_cast<double>(keep);
    size_t inliers = 0;
    for (const float d : distances) {
      if (d <= inlier_distance_) {
        ++inliers;
      }
    }
    inlier_ratio =
      static_cast<double>(inliers) / static_cast<double>(distances.size());
  }

  const double corr_x = best.x - prior.x;
  const double corr_y = best.y - prior.y;
  const double corr_yaw = normalizeAngle(best.yaw - prior.yaw);
  const bool score_ok =
    score >= score_threshold_ && inlier_ratio >= inlier_ratio_threshold_;
  const bool correction_ok =
    std::hypot(corr_x, corr_y) <= max_corr_xy &&
    std::abs(corr_yaw) <= max_corr_yaw;
  // 地形相似（并排狗洞等）：最优与次优得分过于接近，本帧不可信。
  const bool ambiguous =
    std::isfinite(second_score) &&
    (coarse_best_score - second_score) < ambiguity_margin_;
  // 重定位阶段限制单次跳变量，防止直接跳到相似地形上。
  const bool reloc_jump_too_far =
    relocalizing_ && std::hypot(corr_x, corr_y) > reloc_max_prior_deviation_;

  Pose2D map_base_output = prior;
  Pose2D updated_map_to_odom = current_map_to_odom;
  if (score_ok && correction_ok && !reloc_jump_too_far) {
    last_accept_stamp_ = stamp;
    has_last_accept_ = true;
    reject_streak_ = 0;
    if (ambiguous) {
      // 保持上一帧 map->odom，用 odom 先验顶住，等歧义消失，避免锁到错洞。
      if (converged_) {
        updateConvergence(updated_map_to_odom);
      }
      map_base_output = compose(updated_map_to_odom, odom_base);
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), static_cast<int>(log_period_ * 1000.0),
        "[scan_match] ambiguous best=%.3f second=%.3f margin=%.3f "
        "second_pose=(%.2f, %.2f), hold map->odom",
        coarse_best_score, second_score, coarse_best_score - second_score,
        second_pose.x, second_pose.y);
    } else if (std::hypot(corr_x, corr_y) < min_correction_xy_ &&
      std::abs(corr_yaw) < min_correction_yaw_)
    {
      // 修正量小于栅格量化噪声，保持上一位姿，避免 map->odom 抖动。
      updateConvergence(updated_map_to_odom);
      map_base_output = compose(updated_map_to_odom, odom_base);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), static_cast<int>(log_period_ * 1000.0),
        "[scan_match] ok score=%.3f inlier=%.2f dmean=%.3f corr=(%.3f, %.3f, %.4f) "
        "map->odom=(%.3f, %.3f, %.4f) pts=%zu init=%d converged=%d",
        score, inlier_ratio, mean_distance, corr_x, corr_y, corr_yaw,
        updated_map_to_odom.x, updated_map_to_odom.y, updated_map_to_odom.yaw,
        points.size(), initializing ? 1 : 0, converged_ ? 1 : 0);
    } else {
      const Pose2D new_map_odom = compose(best, inverse(odom_base));
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        map_to_odom_ = blend(map_to_odom_, new_map_odom, alpha);
        updated_map_to_odom = map_to_odom_;
      }
      updateConvergence(updated_map_to_odom);
      map_base_output = compose(updated_map_to_odom, odom_base);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), static_cast<int>(log_period_ * 1000.0),
        "[scan_match] ok score=%.3f inlier=%.2f dmean=%.3f corr=(%.3f, %.3f, %.4f) "
        "map->odom=(%.3f, %.3f, %.4f) pts=%zu init=%d converged=%d",
        score, inlier_ratio, mean_distance, corr_x, corr_y, corr_yaw,
        updated_map_to_odom.x, updated_map_to_odom.y, updated_map_to_odom.yaw,
        points.size(), initializing ? 1 : 0, converged_ ? 1 : 0);
    }
  } else {
    ++reject_streak_;
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), static_cast<int>(log_period_ * 1000.0),
      "[scan_match] rejected score=%.3f inlier=%.2f corr=(%.3f, %.3f, %.4f) "
      "reason=%s%s%s",
      score, inlier_ratio, corr_x, corr_y, corr_yaw,
      score_ok ? "" : "score ",
      correction_ok ? "" : "correction ",
      reloc_jump_too_far ? "reloc-jump" : "");
  }

  publishOutputs(stamp, map_base_output, points);
}

void ScanMatchLocalizer::publishOutputs(
  const rclcpp::Time & stamp, const Pose2D & map_base,
  const std::vector<Eigen::Vector2f> & points)
{
  if (publish_pose_ && pose_pub_) {
    geometry_msgs::msg::PoseWithCovarianceStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = map_frame_;
    msg.pose.pose.position.x = map_base.x;
    msg.pose.pose.position.y = map_base.y;
    msg.pose.pose.position.z = 0.0;
    const double half_yaw = 0.5 * map_base.yaw;
    msg.pose.pose.orientation.x = 0.0;
    msg.pose.pose.orientation.y = 0.0;
    msg.pose.pose.orientation.z = std::sin(half_yaw);
    msg.pose.pose.orientation.w = std::cos(half_yaw);
    msg.pose.covariance.fill(0.0);
    msg.pose.covariance[0] = 0.0025;
    msg.pose.covariance[7] = 0.0025;
    msg.pose.covariance[35] = 0.0004;
    pose_pub_->publish(msg);
  }

  if (publish_cloud_ && cloud_pub_) {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    cloud.header.frame_id = map_frame_;
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(points.size());
    cloud.is_bigendian = false;
    cloud.point_step = 12;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.is_dense = true;
    cloud.fields.resize(3);
    const char * field_names[3] = {"x", "y", "z"};
    for (size_t i = 0; i < 3; ++i) {
      cloud.fields[i].name = field_names[i];
      cloud.fields[i].offset = static_cast<uint32_t>(i * 4U);
      cloud.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
      cloud.fields[i].count = 1;
    }
    cloud.data.resize(cloud.row_step);

    const double c = std::cos(map_base.yaw);
    const double s = std::sin(map_base.yaw);
    for (size_t i = 0; i < points.size(); ++i) {
      const float wx = static_cast<float>(
        map_base.x + c * points[i].x() - s * points[i].y());
      const float wy = static_cast<float>(
        map_base.y + s * points[i].x() + c * points[i].y());
      const float wz = 0.0F;
      std::memcpy(&cloud.data[i * 12U + 0U], &wx, sizeof(float));
      std::memcpy(&cloud.data[i * 12U + 4U], &wy, sizeof(float));
      std::memcpy(&cloud.data[i * 12U + 8U], &wz, sizeof(float));
    }
    cloud_pub_->publish(cloud);
  }
}

void ScanMatchLocalizer::publishTf(const rclcpp::Time & stamp)
{
  if (!tf_broadcaster_ || map_frame_ == odom_frame_) {
    return;
  }
  Pose2D map_to_odom;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    map_to_odom = map_to_odom_;
  }
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = stamp;
  tf.header.frame_id = map_frame_;
  tf.child_frame_id = odom_frame_;
  tf.transform.translation.x = map_to_odom.x;
  tf.transform.translation.y = map_to_odom.y;
  tf.transform.translation.z = 0.0;
  const double half_yaw = 0.5 * map_to_odom.yaw;
  tf.transform.rotation.x = 0.0;
  tf.transform.rotation.y = 0.0;
  tf.transform.rotation.z = std::sin(half_yaw);
  tf.transform.rotation.w = std::cos(half_yaw);
  tf_broadcaster_->sendTransform(tf);
}

void ScanMatchLocalizer::tfTimerCallback()
{
  publishTf(now());
}

void ScanMatchLocalizer::updateConvergence(const Pose2D & map_odom)
{
  history_.push_back(map_odom);
  while (static_cast<int>(history_.size()) > converged_count_) {
    history_.pop_front();
  }

  double max_xy = 0.0;
  double max_yaw = 0.0;
  for (size_t i = 0; i < history_.size(); ++i) {
    for (size_t j = i + 1; j < history_.size(); ++j) {
      max_xy = std::max(
        max_xy,
        std::hypot(history_[i].x - history_[j].x, history_[i].y - history_[j].y));
      max_yaw = std::max(
        max_yaw,
        std::abs(normalizeAngle(history_[i].yaw - history_[j].yaw)));
    }
  }

  const bool was_converged = converged_;
  converged_ = static_cast<int>(history_.size()) >= converged_count_ &&
    max_xy <= max_xy_spread_ && max_yaw <= max_yaw_spread_;
  if (converged_) {
    relocalizing_ = false;
  }
  if (converged_ && !was_converged) {
    RCLCPP_INFO(
      get_logger(), "[scan_match] converged spread_xy=%.3f spread_yaw=%.4f",
      max_xy, max_yaw);
  } else if (!converged_ && was_converged) {
    RCLCPP_WARN(
      get_logger(), "[scan_match] lost convergence spread_xy=%.3f spread_yaw=%.4f",
      max_xy, max_yaw);
  }
}

Pose2D ScanMatchLocalizer::compose(const Pose2D & a, const Pose2D & b)
{
  Pose2D out;
  out.x = a.x + std::cos(a.yaw) * b.x - std::sin(a.yaw) * b.y;
  out.y = a.y + std::sin(a.yaw) * b.x + std::cos(a.yaw) * b.y;
  out.yaw = normalizeAngle(a.yaw + b.yaw);
  return out;
}

Pose2D ScanMatchLocalizer::inverse(const Pose2D & a)
{
  Pose2D out;
  out.x = -std::cos(a.yaw) * a.x - std::sin(a.yaw) * a.y;
  out.y = std::sin(a.yaw) * a.x - std::cos(a.yaw) * a.y;
  out.yaw = normalizeAngle(-a.yaw);
  return out;
}

Pose2D ScanMatchLocalizer::blend(
  const Pose2D & from, const Pose2D & to, double alpha)
{
  Pose2D out;
  out.x = from.x + alpha * (to.x - from.x);
  out.y = from.y + alpha * (to.y - from.y);
  out.yaw = normalizeAngle(
    from.yaw + alpha * normalizeAngle(to.yaw - from.yaw));
  return out;
}

double ScanMatchLocalizer::normalizeAngle(double angle)
{
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

}  // namespace scan_match_localizer

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<scan_match_localizer::ScanMatchLocalizer>();
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 3U);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
