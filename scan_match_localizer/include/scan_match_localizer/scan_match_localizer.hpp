#ifndef SCAN_MATCH_LOCALIZER__SCAN_MATCH_LOCALIZER_HPP_
#define SCAN_MATCH_LOCALIZER__SCAN_MATCH_LOCALIZER_HPP_

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

namespace scan_match_localizer
{

struct Pose2D
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct OdomSample
{
  rclcpp::Time stamp;
  Pose2D pose;
  Eigen::Isometry3d pose3d{Eigen::Isometry3d::Identity()};
  // body 系速度，用于 odom 时间戳空洞时外推
  Eigen::Vector3d linear_vel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_vel{Eigen::Vector3d::Zero()};
};

/**
 * 固定先验栅格地图 + 实时 LaserScan 的局部似然场匹配。
 *
 * 匹配结果不修改地图，而是换算成 map -> odom 的 TF 发布：
 *   T_map_odom = T_map_base * T_odom_base^-1
 * 其中 T_odom_base 来自 point_lio 的 TF，T_map_base 由似然场搜索得到。
 */
class ScanMatchLocalizer : public rclcpp::Node
{
public:
  explicit ScanMatchLocalizer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void tfTimerCallback();

  // ----- 地图与距离场 -----
  void buildDistanceField();
  void worldToGrid(double wx, double wy, double & gx, double & gy) const;
  float sampleDistance(double gx, double gy) const;
  bool isOccupied(double wx, double wy) const;
  bool footprintOccupied(double wx, double wy) const;
  float lookupLikelihood(float distance) const;

  // ----- 匹配 -----
  bool loadScanInBase(
    const sensor_msgs::msg::LaserScan & scan, std::vector<Eigen::Vector2f> & points);
  bool loadCloudInBase(
    const sensor_msgs::msg::PointCloud2 & cloud, const Eigen::Isometry3d & odom_base,
    std::vector<Eigen::Vector2f> & points);
  bool lookupOdomBase(const rclcpp::Time & stamp, Pose2D & odom_base);
  bool lookupOdomPose3D(const rclcpp::Time & stamp, Eigen::Isometry3d & odom_base);
  void processPoints(
    const rclcpp::Time & stamp, const std::vector<Eigen::Vector2f> & points,
    const Pose2D & odom_base);
  void collectDistances(
    const Pose2D & pose, const std::vector<Eigen::Vector2f> & points,
    std::vector<float> & distances) const;
  void searchWindow(
    const std::vector<Eigen::Vector2f> & points, const Pose2D & center,
    double xy_radius, double xy_step, double yaw_radius, double yaw_step,
    const Pose2D & prior, Pose2D & best, double & best_score,
    double & second_score, Pose2D & second_pose) const;

  // ----- 输出 -----
  void publishTf(const rclcpp::Time & stamp);
  void publishOutputs(
    const rclcpp::Time & stamp, const Pose2D & map_base,
    const std::vector<Eigen::Vector2f> & points);
  void updateConvergence(const Pose2D & map_odom);

  static Pose2D compose(const Pose2D & a, const Pose2D & b);
  static Pose2D inverse(const Pose2D & a);
  static Pose2D blend(const Pose2D & from, const Pose2D & to, double alpha);
  static double normalizeAngle(double angle);

  // ----- 参数 -----
  std::string map_topic_;
  std::string scan_topic_;
  std::string odom_topic_;
  std::string input_type_;
  std::string cloud_topic_;
  std::string map_frame_;
  std::string odom_frame_;
  std::string base_frame_;
  double tf_rate_{50.0};
  double max_scan_rate_{10.0};
  double reloc_rate_{2.0};
  int max_scan_points_{200};
  double min_range_{0.15};
  double max_range_{10.0};
  int min_valid_points_{30};
  double cloud_z_min_{0.10};
  double cloud_z_max_{1.50};
  double cloud_body_radius_{0.45};
  int occupied_threshold_{65};
  double max_distance_{1.0};
  double sigma_hit_{0.15};
  double inlier_distance_{0.4};
  double trim_ratio_{0.7};
  double search_xy_{0.4};
  double search_yaw_{0.1};
  double coarse_xy_step_{0.04};
  double coarse_yaw_step_{0.01};
  double init_search_xy_{1.0};
  double init_search_yaw_{0.3};
  double init_coarse_xy_step_{0.08};
  double init_coarse_yaw_step_{0.02};
  double init_smoothing_alpha_{1.0};
  double fine_xy_radius_{0.06};
  double fine_yaw_radius_{0.015};
  double fine_xy_step_{0.01};
  double fine_yaw_step_{0.0025};
  double prior_xy_weight_{0.05};
  double prior_yaw_weight_{0.05};
  double robot_radius_{0.2};
  double footprint_penalty_{0.25};
  double score_threshold_{0.4};
  double inlier_ratio_threshold_{0.4};
  double max_translation_correction_{0.12};
  double max_yaw_correction_{0.04};
  double smoothing_alpha_{0.25};
  double min_correction_xy_{0.03};
  double min_correction_yaw_{0.01};
  int converged_count_{3};
  double max_xy_spread_{0.12};
  double max_yaw_spread_{0.03};
  // 收敛后超过该时长没有成功匹配，就降级回大窗口重定位
  double reloc_trigger_timeout_{0.6};
  // odom 时间戳空洞时，用最近样本 twist 外推的最大时长
  double odom_extrapolate_max_dt_{0.5};
  // 地形歧义：最优与次优（相距 > ambiguity_min_distance）得分差小于该值时，
  // 不更新 map->odom，保持 odom 先验，避免锁到相似地形（如并排狗洞）。
  double ambiguity_margin_{0.03};
  double ambiguity_min_distance_{0.4};
  // 重定位阶段允许的最大先验偏差，防止跳到相似地形
  double reloc_max_prior_deviation_{0.6};
  bool publish_cloud_{true};
  bool publish_pose_{true};
  double log_period_{1.0};
  std::vector<double> initial_map_to_odom_;

  // ----- 地图状态 -----
  std::atomic<bool> map_ready_{false};
  int width_{0};
  int height_{0};
  double resolution_{0.0};
  double origin_x_{0.0};
  double origin_y_{0.0};
  double origin_yaw_{0.0};
  double origin_cos_{1.0};
  double origin_sin_{0.0};
  std::vector<uint8_t> occupied_;
  std::vector<float> distance_field_;
  std::vector<float> exp_lut_;
  int lut_size_{1024};

  // ----- 估计状态 -----
  Pose2D map_to_odom_;
  std::deque<OdomSample> odom_history_;
  std::unordered_map<std::string, Eigen::Isometry3d> scan_tf_cache_;
  std::deque<Pose2D> history_;
  bool converged_{false};
  // 由“拒帧超时”触发的大窗口重定位（区别于启动时的初始捕获）
  bool relocalizing_{false};
  bool has_last_accept_{false};
  int reject_streak_{0};
  rclcpp::Time last_accept_stamp_;
  bool has_last_scan_{false};
  rclcpp::Time last_scan_stamp_;
  // map_to_odom_/odom_history_ 会被 sensor 线程和 odom/TF 线程同时访问
  mutable std::mutex state_mutex_;
  // occupied_/distance_field_/地图元数据由 map 回调和匹配线程共享
  mutable std::mutex map_mutex_;

  // ----- ROS 接口 -----
  rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;
  rclcpp::CallbackGroup::SharedPtr odom_cb_group_;
  rclcpp::CallbackGroup::SharedPtr map_cb_group_;
  rclcpp::CallbackGroup::SharedPtr tf_cb_group_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr tf_timer_;
};

}  // namespace scan_match_localizer

#endif  // SCAN_MATCH_LOCALIZER__SCAN_MATCH_LOCALIZER_HPP_
