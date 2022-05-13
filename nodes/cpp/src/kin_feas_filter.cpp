#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <cl_tsgrasp/Grasps.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Header.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <rosparam_shortcuts/rosparam_shortcuts.h>
#include "Eigen/Core"

#include <omp.h>

#include <chrono>
using namespace std::chrono;
#include <iostream>

class GraspFilter
{
  public:
    GraspFilter(planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor, const robot_model::JointModelGroup* arm_jmg, double timeout, tf2_ros::Buffer* tf_buffer);

    std::vector<bool> filter_grasps(std::vector<geometry_msgs::Pose> poses, std_msgs::Header header);

    std::string ik_frame_;

    tf2_ros::Buffer* tf_buffer_;

  protected:

    const robot_model::JointModelGroup* arm_jmg_;
    const planning_scene::PlanningScenePtr planning_scene_;
    kinematics::KinematicsBaseConstPtr kin_solver_;
    double timeout_;
    
};


GraspFilter::GraspFilter(planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor, const robot_model::JointModelGroup* arm_jmg, double timeout, tf2_ros::Buffer* tf_buffer)
  : tf_buffer_(tf_buffer)
  , arm_jmg_(arm_jmg)
  , planning_scene_(planning_scene::PlanningScene::clone(planning_scene_monitor->getPlanningScene()))
  , timeout_(timeout)
{
  // Create an ik solver
  kin_solver_ = arm_jmg_->getSolverInstance();
  ik_frame_ = kin_solver_->getBaseFrame();

}

std::vector<bool> GraspFilter::filter_grasps(std::vector<geometry_msgs::Pose> poses, std_msgs::Header header)
{

  // Prepare to transform poses into IK frame
  geometry_msgs::TransformStamped ik_tf = tf_buffer_->lookupTransform(
    ik_frame_, header.frame_id, ros::Time(0), ros::Duration(1.0) 
  );

  // Use current state as the IK seed state
  robot_model::RobotState state = planning_scene_->getCurrentState();
  std::vector<double> ik_seed_state;
  state.copyJointGroupPositions(arm_jmg_, ik_seed_state);

  // Loop through poses and find those that are kinematically feasible
  std::vector<bool> feasible(poses.size(), false);
  boost::mutex lock;
  for (std::size_t grasp_id = 0; grasp_id < poses.size(); ++grasp_id)
  {
    boost::mutex::scoped_lock slock(lock);
    // transform pose into IK frame
    tf2::doTransform(poses[grasp_id], poses[grasp_id], ik_tf);

    std::vector<double> solution;
    moveit_msgs::MoveItErrorCodes error_code;

    // We're using getPositionIK, which finds the joint angles to reach a pose in ~20 uS, without checking for
    // collisions. To incorporate collisions, refer to the moveit_grasps implementation for an example of  using
    // RobotState::setFromIK with a GroupStateValidityCallbackFn. This is much slower.

    // Note: the IKFast plugin seems to work, but it throws a cryptic nonfatal error like "6 is 0.00000".
    kin_solver_->getPositionIK(poses[grasp_id], ik_seed_state, solution, error_code);

    bool isValid = error_code.val == moveit_msgs::MoveItErrorCodes::SUCCESS;

    feasible[grasp_id] = isValid;

  }

  return feasible;

}

/* Find the z-offset orbital poses from this list of poses. */
void find_orbital_poses(const std::vector<geometry_msgs::Pose>& grasp_poses,
  std::vector<geometry_msgs::Pose>& orbital_poses)
{
  for (geometry_msgs::Pose p : grasp_poses)
  {
    geometry_msgs::Quaternion orn = p.orientation;

    Eigen::Quaterniond q(orn.w, orn.x, orn.y, orn.z);
    Eigen::Matrix3d rot_mat = q.toRotationMatrix();

    Eigen::Vector3d z_hat = rot_mat.col(2);
    Eigen::Vector3d pos(p.position.x, p.position.y, p.position.z);

    float ORBIT_RADIUS = 0.1; // TODO fix magic number
    pos = pos - z_hat * ORBIT_RADIUS;

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

ros::Publisher filtered_grasp_pub;
ros::Publisher final_pose_pub;
std::unique_ptr<GraspFilter> gf = nullptr;

void grasps_cb(const cl_tsgrasp::GraspsConstPtr& msg)
{
  // filter grasps by kinematic feasibility
  std::vector<bool> feasible = gf->filter_grasps(msg->poses, msg->header);

  // filter orbital poses by kinematic feasibility
  std::vector<geometry_msgs::Pose> o_poses;
  find_orbital_poses(msg->poses, o_poses);
  std::vector<bool> o_pose_feasible = gf->filter_grasps(o_poses, msg->header);


  std::vector<geometry_msgs::Pose> filtered_poses;
  std::vector<float> filtered_confs;
  for (size_t i = 0; i < feasible.size(); ++i)
  {
    if (feasible[i] && o_pose_feasible[i])
    {
      filtered_poses.push_back(msg->poses[i]);
      filtered_confs.push_back(msg->confs[i]);
      ROS_INFO_STREAM("FEASIBLE POSE FOUND: \nframe: " << msg->header.frame_id <<"\n" << 
        msg->poses[i]
      );
      // Prepare to transform poses into IK frame
      geometry_msgs::TransformStamped tf = gf->tf_buffer_->lookupTransform(
        "world", msg->header.frame_id, ros::Time(0), ros::Duration(1.0) 
      );
      geometry_msgs::Pose res = msg->poses[i];
      tf2::doTransform(res, res, tf);

      ROS_INFO_STREAM("\nframe: " << "world" <<"\n" << 
        res
      );

      
      break;
    }
  }

  // publish a new Grasps message
  cl_tsgrasp::Grasps filtered_msg;
  filtered_msg.poses = filtered_poses;
  filtered_msg.confs = filtered_confs;
  filtered_msg.header = msg->header;

  filtered_grasp_pub.publish(filtered_msg);

  if (filtered_msg.poses.size() > 0)
  {
    // Prepare to transform poses into IK frame
    geometry_msgs::TransformStamped tf = gf->tf_buffer_->lookupTransform(
      "bravo_base_link", msg->header.frame_id, ros::Time(0), ros::Duration(1.0) 
    );
    geometry_msgs::Pose res = filtered_msg.poses[0];
    tf2::doTransform(res, res, tf);

    geometry_msgs::PoseStamped final_goal_pose;
    final_goal_pose.header = msg->header;
    final_goal_pose.header.frame_id = "bravo_base_link";
    final_goal_pose.pose = res;
    final_pose_pub.publish(final_goal_pose);
  }
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

  ros::NodeHandle rpnh(nh, parent_name);
  std::size_t error = 0;
  error += !rosparam_shortcuts::get(parent_name, rpnh, "ik_timeout", timeout);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "robot_description_name", robot_description_name);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "move_group_name", move_group_name);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "input_grasps_topic", input_grasps_topic);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "output_grasps_topic", output_grasps_topic);
  rosparam_shortcuts::shutdownIfError(parent_name, error);

  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(robot_description_name);
  const robot_model::RobotModelConstPtr robot_model = planning_scene_monitor->getRobotModel();
  const robot_model::JointModelGroup* arm_jmg = robot_model->getJointModelGroup(move_group_name);

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf2_listener(tf_buffer);

  gf = std::make_unique<GraspFilter>(planning_scene_monitor, arm_jmg, timeout, &tf_buffer);

  filtered_grasp_pub = nh.advertise<cl_tsgrasp::Grasps>(output_grasps_topic, 1000);
  final_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("tsgrasp/final_goal_pose", 1000);

  ros::Subscriber grasp_sub = nh.subscribe<cl_tsgrasp::Grasps>(input_grasps_topic, 1, &grasps_cb);

  ros::spin();
}