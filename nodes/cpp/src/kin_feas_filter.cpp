#include <ros/ros.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <cl_tsgrasp/Grasps.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Header.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <rosparam_shortcuts/rosparam_shortcuts.h>
#include "Eigen/Core"
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

#include <omp.h>

#include <chrono>
using namespace std::chrono;
#include <iostream>

namespace
{
bool isGraspStateValid(const planning_scene_monitor::LockedPlanningSceneRO planning_scene,
                       robot_state::RobotState* robot_state, const robot_state::JointModelGroup* group,
                       const double* ik_solution)
{
  robot_state->setJointGroupPositions(group, ik_solution);
  robot_state->update();
  if (!robot_state->satisfiesBounds(group))
  {
    ROS_DEBUG_STREAM_NAMED("is_grasp_state_valid", "Ik solution invalid");
    return false;
  }

  if (!planning_scene)
  {
    ROS_ERROR_STREAM_NAMED("is_grasp_state_valid", "No planning scene provided");
    return false;
  }

  return !planning_scene->isStateColliding(*robot_state, "", true); // has optional "verbose" arg
}
}  // namespace


class GraspFilter
{
  public:
    GraspFilter(planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor, 
      const robot_model::JointModelGroup* arm_jmg, double timeout, tf2_ros::Buffer* tf_buffer, 
      int num_threads, double orbital_radius, bool visual_debug);

    std::vector<bool> filter_grasps(std::vector<geometry_msgs::Pose> poses);

    std::string ik_frame_;
    std::string model_frame_;
    tf2_ros::Buffer* tf_buffer_;

    double orbital_radius_;

    bool visual_debug_;
    moveit_visual_tools::MoveItVisualToolsPtr visual_tools_;
    std::map<std::string, std::vector<kinematics::KinematicsBaseConstPtr>> kin_solvers;
    const robot_model::JointModelGroup* arm_jmg_;

    const planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;

  protected:

    size_t num_threads_;
    double timeout_;
    
};

GraspFilter::GraspFilter(planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor, 
  const robot_model::JointModelGroup* arm_jmg, double timeout, tf2_ros::Buffer* tf_buffer, 
  int num_threads, double orbital_radius, bool visual_debug)
  : tf_buffer_(tf_buffer)
  , orbital_radius_(orbital_radius)
  , arm_jmg_(arm_jmg)
  , planning_scene_monitor_(planning_scene_monitor)
  , timeout_(timeout)
  , visual_debug_(visual_debug)
{

  if (visual_debug_) 
  {
    visual_tools_.reset(new moveit_visual_tools::MoveItVisualTools("world", "/moveit_visual_markers", planning_scene_monitor_));
    ros::Duration(5.0).sleep(); // wait for rviz
    visual_tools_->enableBatchPublishing(false);
  }

  if (num_threads > omp_get_max_threads() || num_threads < 1)
  {
    num_threads_ = omp_get_max_threads();
  } else
  {
    num_threads_ = num_threads;
  }
  ROS_INFO_STREAM("Set num_threads_ to: " << num_threads_);

  // Create an ik solver for every thread
  for (std::size_t i = 0; i < num_threads_; ++i)
  {
    kin_solvers[arm_jmg_->getName()].push_back(arm_jmg_->getSolverInstance());

    if (!kin_solvers[arm_jmg_->getName()][i])
    {
      ROS_ERROR_STREAM("No kinematic solver found");
    }
  }

  // Print error if kinematic solver frame different from robot model
  ik_frame_ = kin_solvers[arm_jmg_->getName()][0]->getBaseFrame();
  robot_model::RobotState state = planning_scene_monitor_->getPlanningScene()->getCurrentState();
  model_frame_ = state.getRobotModel()->getModelFrame();
  if (!moveit::core::Transforms::sameFrame(ik_frame_, model_frame_))
  {
      ROS_ERROR_STREAM("Robot model has different frame (" << model_frame_ << ") from kinematic solver frame (" << ik_frame_ << ")");
  }
}

std::vector<bool> GraspFilter::filter_grasps(std::vector<geometry_msgs::Pose> poses)
{
  // Use current state as the IK seed state
  robot_model::RobotStatePtr state = planning_scene_monitor_->getStateMonitor()->getCurrentState();
  std::vector<double> ik_seed_state;
  state->copyJointGroupPositions(arm_jmg_, ik_seed_state);

  // Create constraint_fn
  moveit::core::GroupStateValidityCallbackFn constraint_fn =
      boost::bind(&isGraspStateValid, 
      planning_scene_monitor::LockedPlanningSceneRO(planning_scene_monitor_), _1, _2, _3);

  // Loop through poses and find those that are kinematically feasible
  std::vector<bool> feasible(poses.size(), false);
  boost::mutex vector_lock;

  // omp_set_num_threads(num_threads_);
  // #pragma omp parallel for schedule(dynamic)
  for (std::size_t grasp_id = 0; grasp_id < poses.size(); ++grasp_id)
  {

    // perform IK
    // std::size_t thread_id = omp_get_thread_num();

    std::vector<double> solution;
    moveit_msgs::MoveItErrorCodes error_code;

    // bool isValid = kin_solvers[arm_jmg_->getName()][thread_id]->getPositionIK(
    //   poses[grasp_id], ik_seed_state, solution, error_code); // does not check collision
    bool isValid = state->setFromIK(arm_jmg_, poses[grasp_id], timeout_, constraint_fn);

    if ( visual_debug_ && isValid ){
      trajectory_msgs::JointTrajectoryPoint point;
      state->copyJointGroupPositions(arm_jmg_, point.positions);
      // std::vector<trajectory_msgs::JointTrajectoryPoint> ik_solutions; // save each grasps ik solution for visualization
      // ik_solutions.push_back(point);
      // visual_tools_->publishIKSolutions(ik_solutions, arm_jmg_, 4.0);
      
      ROS_INFO("Publishing IK solutions.");

      visual_tools_->publishRobotState(point.positions, arm_jmg_);
      ros::Duration(0.5).sleep();
      visual_tools_->trigger();
      ros::Duration(0.5).sleep();
      visual_tools_->trigger();

      ROS_INFO_STREAM("Published IK solutions.");

    }
    
    {
      boost::mutex::scoped_lock slock(vector_lock);
      feasible[grasp_id] = isValid;
    }

  }

  return feasible;

}

/* Find the z-offset orbital poses from this list of poses. */
void find_orbital_poses(const std::vector<geometry_msgs::Pose>& grasp_poses,
  std::vector<geometry_msgs::Pose>& orbital_poses, double orbital_radius)
{
  for (geometry_msgs::Pose p : grasp_poses)
  {
    geometry_msgs::Quaternion orn = p.orientation;

    Eigen::Quaterniond q(orn.w, orn.x, orn.y, orn.z);
    Eigen::Matrix3d rot_mat = q.toRotationMatrix();

    Eigen::Vector3d z_hat = rot_mat.col(2);
    Eigen::Vector3d pos(p.position.x, p.position.y, p.position.z);

    pos = pos - z_hat * orbital_radius;

    geometry_msgs::Point o_pos;
    o_pos.x = pos(0);
    o_pos.y = pos(1);
    o_pos.z = pos(2);

    geometry_msgs::Pose o_pose;
    o_pose.position = o_pos;
    o_pose.orientation = orn;

    orbital_poses.push_back(o_pose);
  }
}

void grasps_cb(GraspFilter& gf, ros::Publisher& pub, const cl_tsgrasp::Grasps& msg)
{
  auto start = high_resolution_clock::now();

  // transform all grasp poses into the IK solver frame if not already
  std::vector<geometry_msgs::Pose> poses;
  // for (size_t i = 0; i < msg.poses.size(); ++i)
  // {
    
  //   robot_model::RobotState state = gf.planning_scene_monitor_->getPlanningScene()->getCurrentState();
  //   Eigen::Isometry3d pose;
  //   tf2::fromMsg(msg.poses[i], pose);
  //   kinematics::KinematicsBaseConstPtr p = gf.kin_solvers[gf.arm_jmg_->getName()][0];
  //   state.setToIKSolverFrame(pose, p);
  // }

  geometry_msgs::TransformStamped tf = gf.tf_buffer_->lookupTransform(
        gf.model_frame_, msg.header.frame_id, ros::Time(0), ros::Duration(1.0));

  if (moveit::core::Transforms::sameFrame(gf.model_frame_, msg.header.frame_id))
  {
      poses = msg.poses;
  } else
  {
    for (geometry_msgs::Pose p : msg.poses) 
    {
      geometry_msgs::Pose ik_pose;
      tf2::doTransform(p, ik_pose, tf);
      poses.push_back(ik_pose);
    }
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  ROS_INFO_STREAM("Transforming Grasps took: " << duration.count() << " us");

  start = high_resolution_clock::now();

  // filter grasps by kinematic feasibility
  std::vector<bool> feasible = gf.filter_grasps(poses);

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  ROS_INFO_STREAM("Initial KF check took: " << duration.count() << " us");

  std::vector<geometry_msgs::Pose> filtered_poses;
  std::vector<float> filtered_confs;
  for (size_t i = 0; i < feasible.size(); ++i)
  {
    if (feasible[i])
    {
      filtered_poses.push_back(poses[i]);
      filtered_confs.push_back(msg.confs[i]);
    }
  }

  start = high_resolution_clock::now();

  // filter orbital poses by kinematic feasibility
  std::vector<geometry_msgs::Pose> o_poses;
  find_orbital_poses(filtered_poses, o_poses, gf.orbital_radius_);

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  ROS_INFO_STREAM("Finding orbital poses took: " << duration.count() << " us");

  start = high_resolution_clock::now();
  std::vector<bool> o_pose_feasible = gf.filter_grasps(o_poses);

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  ROS_INFO_STREAM("Filtering orbital poses took: " << duration.count() << " us");

  start = high_resolution_clock::now();

  std::vector<geometry_msgs::Pose> twice_filtered_poses;
  std::vector<geometry_msgs::Pose> orbital_twice_filtered_poses;
  std::vector<float> twice_filtered_confs;
  for (size_t i = 0; i < o_pose_feasible.size(); ++i)
  {
    if (o_pose_feasible[i])
    {
      twice_filtered_poses.push_back(filtered_poses[i]);
      twice_filtered_confs.push_back(filtered_confs[i]);

      // orbital_twice_filtered_poses holds the kinematically feasible
      // orbital poses corresponding to kinematically feasible grasp poses.
      orbital_twice_filtered_poses.push_back(o_poses[i]);
    }
  }

  // transform poses and orbital poses back into camera frame
  // create an inverted transform
  tf2::Transform transform;
  tf2::fromMsg(tf.transform, transform);
  tf2::Transform inv_tf = transform.inverse();
  geometry_msgs::Transform inv_tf_msg = tf2::toMsg(inv_tf);
  geometry_msgs::TransformStamped inv_tf_msg_stamped;
  inv_tf_msg_stamped.transform = inv_tf_msg;
  inv_tf_msg_stamped.header = tf.header;
  inv_tf_msg_stamped.header.frame_id = tf.child_frame_id;
  inv_tf_msg_stamped.child_frame_id = tf.header.frame_id;
  
  std::vector<geometry_msgs::Pose> final_poses_original_frame;
  for (geometry_msgs::Pose p : twice_filtered_poses) 
  {
    geometry_msgs::Pose pose;
    tf2::doTransform(p, pose, inv_tf_msg_stamped);
    final_poses_original_frame.push_back(pose);
  }

  std::vector<geometry_msgs::Pose> orbital_final_poses_original_frame;
  for (geometry_msgs::Pose p : orbital_twice_filtered_poses) 
  {
    geometry_msgs::Pose pose;
    tf2::doTransform(p, pose, inv_tf_msg_stamped);
    orbital_final_poses_original_frame.push_back(pose);
  }

  // publish a new Grasps message
  cl_tsgrasp::Grasps filtered_msg;
  filtered_msg.poses = final_poses_original_frame;
  filtered_msg.orbital_poses = orbital_final_poses_original_frame; // corresponding o_poses
  filtered_msg.confs = twice_filtered_confs;
  filtered_msg.header = msg.header;

  pub.publish(filtered_msg);  

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  ROS_INFO_STREAM("Gathering and publishing took: " << duration.count() << " us");

}

int main(int argc, char **argv)
{
  // Initialize the ROS node
  const std::string parent_name = "kin_feas_filter";
  ros::init(argc, argv, parent_name);
  ros::NodeHandle nh;

  double timeout;
  std::string move_group_name;
  std::string robot_description_name;
  std::string input_grasps_topic;
  std::string output_grasps_topic;
  int num_threads; // setting to a number < 1 or greater than omp_get_max_threads() sets to max
  double orbital_radius;
  bool visual_debug;

  ros::NodeHandle rpnh(nh, parent_name);
  std::size_t error = 0;
  error += !rosparam_shortcuts::get(parent_name, rpnh, "ik_timeout", timeout);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "robot_description_name", robot_description_name);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "move_group_name", move_group_name);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "input_grasps_topic", input_grasps_topic);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "output_grasps_topic", output_grasps_topic);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "num_threads", num_threads);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "orbital_radius", orbital_radius);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "visual_debug", visual_debug);
  rosparam_shortcuts::shutdownIfError(parent_name, error);

  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor = 
    std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(robot_description_name);
  
  // Start the planning scene monitor
  // https://answers.ros.org/question/356183/how-to-use-the-planning-scene-monitor-in-c/
  planning_scene_monitor->startSceneMonitor();
  planning_scene_monitor->startWorldGeometryMonitor(
      planning_scene_monitor::PlanningSceneMonitor::DEFAULT_COLLISION_OBJECT_TOPIC,
      planning_scene_monitor::PlanningSceneMonitor::DEFAULT_PLANNING_SCENE_WORLD_TOPIC,
      false /* skip octomap monitor */);
  planning_scene_monitor->startStateMonitor();
  planning_scene_monitor->requestPlanningSceneState(); // ensure PSM up-to-date

  const robot_model::RobotModelConstPtr robot_model = planning_scene_monitor->getRobotModel();
  const robot_model::JointModelGroup* arm_jmg = robot_model->getJointModelGroup(move_group_name);

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf2_listener(tf_buffer);

  GraspFilter gf(planning_scene_monitor, arm_jmg, timeout, &tf_buffer, num_threads, orbital_radius, visual_debug);

  ros::Publisher pub = nh.advertise<cl_tsgrasp::Grasps>(output_grasps_topic, 1000);

  boost::function<void (const cl_tsgrasp::Grasps&)> cb = boost::bind(&grasps_cb, gf, pub, _1);

  ros::Subscriber sub = nh.subscribe<cl_tsgrasp::Grasps>(input_grasps_topic, 1, cb);
  
  ros::spin();
}