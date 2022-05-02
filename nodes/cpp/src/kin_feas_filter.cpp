#include <ros/ros.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <cl_tsgrasp/Grasps.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Header.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>

#include <omp.h>

#include <chrono>
using namespace std::chrono;

// Things I need for filtering grasps by kinematic feasibility:
// 1) A function that uses MoveIt to quickly attempt an IK solution for a given pose.
// 2) A data structure for representing a given pose.
// 3) A multithreading arrangement to allow IK solutions to be found in parallel.
// I'm simplifying the reference implementation from moveit_grasps.

bool isGraspStateValid(robot_state::RobotState* robot_state, 
  const robot_state::JointModelGroup* joint_group, 
  const double* joint_group_variable_values)
{
  return true; // We set a low bar for grasp state validity.
}
moveit::core::GroupStateValidityCallbackFn constraint_fn = isGraspStateValid;

bool hasIKSolution(robot_state::RobotState state, const robot_model::JointModelGroup* arm_jmg,
  geometry_msgs::Pose ik_pose, double timeout)
{
  return state.setFromIK(
    arm_jmg,
    ik_pose, 
    timeout, 
    constraint_fn
  );
}

class GraspFilter
{
  public:
    GraspFilter(planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor, const robot_model::JointModelGroup* arm_jmg, double timeout, tf2_ros::Buffer* tf_buffer);

    std::vector<bool> filter_grasps(std::vector<geometry_msgs::Pose> poses, std_msgs::Header header);

    std::string ik_frame;

  protected:

    const robot_model::JointModelGroup* arm_jmg_;
    const planning_scene::PlanningScenePtr planning_scene;
    std::map<std::string, std::vector<kinematics::KinematicsBaseConstPtr>> kin_solvers;
    size_t num_threads;
    double timeout_;
    tf2_ros::Buffer* tf_buffer_;
};


GraspFilter::GraspFilter(planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor, const robot_model::JointModelGroup* arm_jmg, double timeout, tf2_ros::Buffer* tf_buffer)
  : arm_jmg_(arm_jmg)
  , planning_scene(planning_scene_monitor->getPlanningScene())
  , timeout_(timeout)
  , tf_buffer_(tf_buffer)
{
  robot_model::RobotState state = planning_scene->getCurrentState();

  // Choose number of threads
  num_threads = omp_get_max_threads();

  // Create an ik solver for every thread
  for (std::size_t i = 0; i < num_threads; ++i)
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

  // Prepare to transform poses into IK frame
  geometry_msgs::TransformStamped ik_tf = tf_buffer_->lookupTransform(
    ik_frame, header.frame_id, ros::Time(0), ros::Duration(1.0) 
  );

  // Use current state as the IK seed state
  robot_model::RobotState state = planning_scene->getCurrentState();
  std::vector<double> ik_seed_state;
  state.copyJointGroupPositions(arm_jmg_, ik_seed_state);

  // Loop through poses and find those that are kinematically feasible
  std::vector<bool> feasible;
  boost::mutex vector_lock;

  omp_set_num_threads(num_threads);
  #pragma omp parallel for schedule(dynamic)
  for (std::size_t grasp_id = 1; grasp_id < poses.size(); ++grasp_id)
  {

    // transform pose into IK frame
    tf2::doTransform(poses[grasp_id], poses[grasp_id], ik_tf);

    // perform IK
    std::size_t thread_id = omp_get_thread_num();

    std::vector<double> solution;
    moveit_msgs::MoveItErrorCodes error_code;

    kin_solvers[arm_jmg_->getName()][thread_id]->searchPositionIK(poses[grasp_id], ik_seed_state, timeout_, solution, error_code);

    bool isValid = error_code.val == moveit_msgs::MoveItErrorCodes::SUCCESS;

    {
      boost::mutex::scoped_lock slock(vector_lock);
      feasible.push_back(isValid);
    }

  }

  return feasible;

}

void grasps_cb(GraspFilter& gf, ros::Publisher& pub, const cl_tsgrasp::Grasps& msg)
{
  // filter grasps by kinematic feasibility
  std::vector<bool> feasible = gf.filter_grasps(msg.poses, msg.header);

  std::vector<geometry_msgs::Pose> filtered_poses;
  std::vector<float> filtered_confs;
  for (size_t i = 0; i < feasible.size(); ++i)
  {
    if (feasible[i])
    {
      filtered_poses.push_back(msg.poses[i]);
      filtered_confs.push_back(msg.confs[i]);
    }
  }

  // publish a new Grasps message
  cl_tsgrasp::Grasps filtered_msg;
  filtered_msg.poses = filtered_poses;
  filtered_msg.confs = filtered_confs;
  filtered_msg.header = msg.header;

  pub.publish(filtered_msg);
}

int main(int argc, char **argv)
{
  // Initialize the ROS node
  ros::init(argc, argv, "kin_feas_filter");
  ros::NodeHandle nh;

  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>("robot_description");
  const robot_model::RobotModelConstPtr robot_model = planning_scene_monitor->getRobotModel();
  const robot_model::JointModelGroup* arm_jmg = robot_model->getJointModelGroup("panda_arm");
  double timeout = 0.05;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf2_listener(tf_buffer);

  GraspFilter gf(planning_scene_monitor, arm_jmg, timeout, &tf_buffer);

  ros::Publisher pub = nh.advertise<cl_tsgrasp::Grasps>("tsgrasp/grasps_filtered", 1000);

  boost::function<void (const cl_tsgrasp::Grasps&)> cb = boost::bind(&grasps_cb, gf, pub, _1);

  ros::Subscriber sub = nh.subscribe<cl_tsgrasp::Grasps>("tsgrasp/grasps", 1, cb);
  
  ros::spin();
}

  // // Make a vector of many poses with random heights
  // std::size_t num_poses = 500;
  // std::vector<geometry_msgs::Pose> poses;
  // poses.resize(num_poses);

  // for (geometry_msgs::Pose &pose : poses) {
  //   pose.position.z = (rand() % 1000) / 1000.0 * 2.0;
  //   pose.orientation.w = 1;
  // }

  // auto start = high_resolution_clock::now();
  // std::vector<bool> feasible = gf.filter_grasps(poses);
  // auto stop = high_resolution_clock::now();
  // auto duration = duration_cast<microseconds>(stop - start);
  
  // size_t sum = 0;
  // for (size_t i = 0; i < feasible.size(); ++i)
  // {
  //   bool inreach = feasible[i];
  //   sum += inreach;
  //   ROS_INFO_STREAM("Grasp " << i << " feasible: " << inreach);
  // }
  // ROS_INFO_STREAM("Proportion feasible: " << (float) sum / feasible.size());
  // ROS_INFO_STREAM("Filtering took: " << duration.count() / 1000000.0 << " seconds.");

// int main(int argc, char **argv)
// {
//   // Initialize the ROS node
//   ros::init(argc, argv, "kin_feas_filter");
//   ros::NodeHandle nh;

//   ros::Subscriber sub = nh.subscribe("tsgrasp/grasps", 1000, grasps_cb);
//   pub = n.advertise<cl_tsgrasp::Grasps>("tsgrasp/grasps_filtered", 1000);

//   planning_scene_monitor = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>("robot_description");

//   robot_model = planning_scene_monitor->getRobotModel();

//   arm_jmg = robot_model->getJointModelGroup("panda_arm");

//   geometry_msgs::PoseStamped pose;
//   pose.pose.position.x = 0;
//   pose.pose.position.y = 0;
//   pose.pose.position.z = 0.5;
//   pose.pose.orientation.x = 0;
//   pose.pose.orientation.y = 0;
//   pose.pose.orientation.z = 0;
//   pose.pose.orientation.w = 1;
//   pose.header.frame_id = "world";

//   robot_model::RobotState state = planning_scene_monitor->getPlanningScene()->getCurrentState();

//   auto start = high_resolution_clock::now();
//   bool solved = hasIKSolution(state, arm_jmg, pose, 1.0);
//   auto stop = high_resolution_clock::now();
//   auto duration = duration_cast<microseconds>(stop - start);

//   ROS_INFO_STREAM("IK solved: " << solved);
//   ROS_INFO_STREAM("    and it took: " << duration.count() << " microseconds.");

//   double timeout = 0.0;

//   // Make a vector of many poses with random heights
//   std::size_t num_poses = 500;
//   std::vector<geometry_msgs::PoseStamped> poses;
//   poses.resize(num_poses);

//   for (geometry_msgs::PoseStamped &pose : poses) {
//     pose.pose.position.z = (rand() % 1000) / 1000.0 * 2.0;
//     pose.pose.orientation.w = 1;
//     pose.header.frame_id = "world";
//   }
  
//   // Choose number of cores
//   std::size_t num_threads = omp_get_max_threads();
//   if (num_threads > poses.size())
//   {
//     num_threads = poses.size();
//   }

//   // Threaded kinematic solvers
//   std::map<std::string, std::vector<kinematics::KinematicsBaseConstPtr>> kin_solvers;

//   // Load kinematic solvers if not already loaded
//   if (kin_solvers[arm_jmg->getName()].size() != num_threads)
//   {
//     kin_solvers[arm_jmg->getName()].clear();

//     // Create an ik solver for every thread
//     for (std::size_t i = 0; i < num_threads; ++i)
//     {
//       ROS_DEBUG_STREAM("Creating ik solver " << i);
//       kin_solvers[arm_jmg->getName()].push_back(arm_jmg->getSolverInstance());

//       // Test to make sure we have a valid kinematics solver
//       if (!kin_solvers[arm_jmg->getName()][i])
//       {
//         ROS_ERROR_STREAM("No kinematic solver found");
//         return 0;
//       }
//     }
//   }

//   std::vector<robot_state::RobotStatePtr> robot_states;

//   // Robot states
//   // Create a robot state for every thread
//   if (robot_states.size() != num_threads)
//   {
//     robot_states.clear();
//     for (std::size_t i = 0; i < num_threads; ++i)
//     {
//       // Copy the previous robot state
//       robot_states.push_back(std::make_shared<robot_state::RobotState>(state));
//     }
//   }
//   else  // update the states
//   {
//     for (std::size_t i = 0; i < num_threads; ++i)
//     {
//       // Copy the previous robot state
//       *(robot_states[i]) = state;
//     }
//   }

//   // Transform poses
//   const std::string& ik_frame = kin_solvers[arm_jmg->getName()][0]->getBaseFrame();
//   const std::string& model_frame = state.getRobotModel()->getModelFrame();
//   if (!moveit::core::Transforms::sameFrame(ik_frame, model_frame))
//   {
//       ROS_ERROR_STREAM("Robot model has different frame (" << model_frame << ") from kinematic solver frame (" << ik_frame << ")");
//   }

//   // Create the seed state vector
//   std::vector<double> ik_seed_state;
//   state.copyJointGroupPositions(arm_jmg, ik_seed_state);

//   // Thread data
//   // Allocate only once to increase performance
//   ik_thread_structs.resize(num_threads);
//   for (std::size_t thread_id = 0; thread_id < num_threads; ++thread_id)
//   {
//     ik_thread_structs[thread_id] =
//         std::make_shared<IkThreadStruct>(arm_jmg, poses, 0, kin_solvers[arm_jmg->getName()][thread_id], robot_states[thread_id], timeout, thread_id);
//     ik_thread_structs[thread_id]->ik_seed_state_ = ik_seed_state;
//   }

//   // Benchmark time
//   auto start2 = high_resolution_clock::now();
//   int valid = 0;

//   // Loop through poses and find those that are kinematically feasible
//   omp_set_num_threads(num_threads);
//   #pragma omp parallel for schedule(dynamic)
//   for (std::size_t grasp_id = 0; grasp_id < poses.size(); ++grasp_id)
//   {
//     std::size_t thread_id = omp_get_thread_num();

//     // Assign grasp to process
//     ik_thread_structs[thread_id]->grasp_id_ = grasp_id;

//     // Process the grasp if it hasn't already been filtered out

//     bool isValid = processCandidateGrasp(ik_thread_structs[thread_id]);

//     if (isValid){++valid;}

//     ROS_INFO_STREAM("Grasp " << grasp_id << " is valid?: " << isValid);
//   }


//   auto stop2 = high_resolution_clock::now();
//   auto duration2 = duration_cast<microseconds>(stop2 - start2);
//   ROS_INFO_STREAM("Completing many IK took: " << duration2.count() / 1000000.0 << " seconds.");

//   ROS_INFO_STREAM("Valid percentage: " << (double) valid / num_poses);

// }