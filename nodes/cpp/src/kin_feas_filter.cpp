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

#include <omp.h>

#include <chrono>
using namespace std::chrono;
#include <iostream>

namespace
{
bool isGraspStateValid(const planning_scene::PlanningScene* planning_scene,
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

  return !planning_scene->isStateColliding(*robot_state, group->getName());
}
}  // namespace


class GraspFilter
{
  public:
    GraspFilter(planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor, const robot_model::JointModelGroup* arm_jmg, double timeout, tf2_ros::Buffer* tf_buffer, int num_threads);

    std::vector<bool> filter_grasps(std::vector<geometry_msgs::Pose> poses, std_msgs::Header header);

    std::string ik_frame;

    tf2_ros::Buffer* tf_buffer_;

  protected:

    const robot_model::JointModelGroup* arm_jmg_;
    const planning_scene::PlanningScenePtr planning_scene;
    std::map<std::string, std::vector<kinematics::KinematicsBaseConstPtr>> kin_solvers;
    size_t num_threads_;
    double timeout_;
    
};


GraspFilter::GraspFilter(planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor, const robot_model::JointModelGroup* arm_jmg, double timeout, tf2_ros::Buffer* tf_buffer, int num_threads)
  : tf_buffer_(tf_buffer)
  , arm_jmg_(arm_jmg)
  , planning_scene(planning_scene::PlanningScene::clone(planning_scene_monitor->getPlanningScene()))
  , timeout_(timeout)
{
  robot_model::RobotState state = planning_scene->getCurrentState();

  // Choose number of threads
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

  // Throw error if kinematic solver frame different from robot model
  ik_frame = kin_solvers[arm_jmg_->getName()][0]->getBaseFrame();
  const std::string& model_frame = state.getRobotModel()->getModelFrame();
  if (!moveit::core::Transforms::sameFrame(ik_frame, model_frame))
  {
      ROS_ERROR_STREAM("Robot model has different frame (" << model_frame << ") from kinematic solver frame (" << ik_frame << ")");
  }
}

std::vector<bool> GraspFilter::filter_grasps(std::vector<geometry_msgs::Pose> poses, std_msgs::Header header)
{
  // Use current state as the IK seed state
  robot_model::RobotState state = planning_scene->getCurrentState();
  std::vector<double> ik_seed_state;
  state.copyJointGroupPositions(arm_jmg_, ik_seed_state);

  // Create constraint_fn
  moveit::core::GroupStateValidityCallbackFn constraint_fn =
      boost::bind(&isGraspStateValid, planning_scene.get(), _1, _2, _3);

  // Loop through poses and find those that are kinematically feasible
  std::vector<bool> feasible(poses.size(), false);
  boost::mutex vector_lock;

  omp_set_num_threads(num_threads_);
  // #pragma omp parallel for schedule(dynamic)
  for (std::size_t grasp_id = 0; grasp_id < poses.size(); ++grasp_id)
  {

    // perform IK
    // std::size_t thread_id = omp_get_thread_num();

    std::vector<double> solution;
    moveit_msgs::MoveItErrorCodes error_code;

    robot_state::RobotState state = planning_scene->getCurrentState();

    bool isValid = state.setFromIK(arm_jmg_, poses[grasp_id], timeout_, constraint_fn);

    {
      boost::mutex::scoped_lock slock(vector_lock);
      feasible[grasp_id] = isValid;
    }

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

void grasps_cb(GraspFilter& gf, ros::Publisher& pub, ros::Publisher& final_pose_pub, const cl_tsgrasp::Grasps& msg)
{
  // transform all grasp poses into the IK solver frame if not already
  std::vector<geometry_msgs::Pose> poses;
  if (moveit::core::Transforms::sameFrame(gf.ik_frame, msg.header.frame_id))
  {
      poses = msg.poses;
  } else
  {
    geometry_msgs::TransformStamped tf = gf.tf_buffer_->lookupTransform(
        gf.ik_frame, msg.header.frame_id, ros::Time(0), ros::Duration(1.0));
    for (geometry_msgs::Pose p : msg.poses) 
    {
      geometry_msgs::Pose ik_pose;
      tf2::doTransform(p, ik_pose, tf);
      poses.push_back(ik_pose);
    }
  }

  // filter grasps by kinematic feasibility
  std::vector<bool> feasible = gf.filter_grasps(poses, msg.header);

  // filter orbital poses by kinematic feasibility
  std::vector<geometry_msgs::Pose> o_poses;
  find_orbital_poses(poses, o_poses);
  std::vector<bool> o_pose_feasible = gf.filter_grasps(o_poses, msg.header);

  std::vector<geometry_msgs::Pose> filtered_poses;
  std::vector<float> filtered_confs;
  for (size_t i = 0; i < feasible.size(); ++i)
  {
    if (feasible[i] && o_pose_feasible[i])
    {
      filtered_poses.push_back(poses[i]);
      filtered_confs.push_back(msg.confs[i]);
    }
  }

  // publish a new Grasps message
  cl_tsgrasp::Grasps filtered_msg;
  filtered_msg.poses = filtered_poses;
  filtered_msg.confs = filtered_confs;
  filtered_msg.header = msg.header;
  filtered_msg.header.frame_id = gf.ik_frame;

  pub.publish(filtered_msg);

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

  ros::NodeHandle rpnh(nh, parent_name);
  std::size_t error = 0;
  error += !rosparam_shortcuts::get(parent_name, rpnh, "ik_timeout", timeout);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "robot_description_name", robot_description_name);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "move_group_name", move_group_name);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "input_grasps_topic", input_grasps_topic);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "output_grasps_topic", output_grasps_topic);
  error += !rosparam_shortcuts::get(parent_name, rpnh, "num_threads", num_threads);
  rosparam_shortcuts::shutdownIfError(parent_name, error);

  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(robot_description_name);
  const robot_model::RobotModelConstPtr robot_model = planning_scene_monitor->getRobotModel();
  const robot_model::JointModelGroup* arm_jmg = robot_model->getJointModelGroup(move_group_name);

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf2_listener(tf_buffer);

  GraspFilter gf(planning_scene_monitor, arm_jmg, timeout, &tf_buffer, num_threads);

  ros::Publisher pub = nh.advertise<cl_tsgrasp::Grasps>(output_grasps_topic, 1000);

  ros::Publisher final_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("tsgrasp/final_goal_pose", 1000);

  boost::function<void (const cl_tsgrasp::Grasps&)> cb = boost::bind(&grasps_cb, gf, pub, final_pose_pub, _1);

  ros::Subscriber sub = nh.subscribe<cl_tsgrasp::Grasps>(input_grasps_topic, 1, cb);
  
  ros::spin();
}