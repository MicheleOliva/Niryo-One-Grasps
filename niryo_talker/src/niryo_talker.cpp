#include "ros/ros.h"
#include <sstream>
#include <cmath>
#include <angles/angles.h>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

// TF
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "tf/transform_listener.h"
#include "tf/transform_datatypes.h"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "tf2_eigen/tf2_eigen.h"

// Messages
#include "sensor_msgs/JointState.h"
#include <std_msgs/Float64.h>
#include <grasp_detection_srv/grasp_detector.h>
#include <grasp_pose_service_msgs/grasp_pose.h>
#include <aruco_camera_extrinsics_srv/aruco_camera_extrinsics.h>

// Implementation of the Niryo One's Manager.
class NiryoManager{
    
    protected:

        moveit::core::RobotStatePtr niryo_realsense_kinematic_state_;
        const moveit::core::JointModelGroup* niryo_realsense_joint_model_group;
        const std::vector<std::string>& niryo_realsense_joint_names;
        ros::NodeHandle nodeHandle;
        ros::Subscriber niryo_joint_state_subscriber_;
        ros::Publisher niryo_realsense_joint_state_publisher_;
        sensor_msgs::JointState niryo_realsense_joint_state_msg;
        std::vector<double> actual_joint_positions;
        ros::ServiceClient grasp_detection_client;
        ros::ServiceClient camera_extrinsics_client;
        ros::ServiceServer grasp_pose_server;
        ros::Subscriber tf_subscriber;
        ros::Subscriber tf_static_subscriber;
        tf2_ros::Buffer tfBuffer;
        XmlRpc::XmlRpcValue grasp_poses_param;
        std::vector<std::vector<double>> grasp_poses;

    public:
        /**
         * Niryo Manager: this class replicates the movements of "Real Niryo One" subscribing to its topic "\joint_states" and republishing its messages on a new topic (same topic, new namespace).
         * To unlock all functionalities its necessary that with the node that uses an instance of this class a robot_state_publisher with some parameters remapped has been set up, as in the launch file 'niryo_grasping.launch' of this package.
         * @param niryo_realsense_kinematic_state: the instance of moveit::core::RobotStatePtr relative to the "simulated" Niryo One.
         * @param niryo_realsense_joint_model_group: the instance of const moveit::core::JointModelGroup* relative to the "simulated" Niryo One.
         * @param niryo_realsense_joint_names: the vector containing the names of the "simulated" Niryo One's joints.
         */
        NiryoManager(   moveit::core::RobotStatePtr niryo_realsense_kinematic_state,
                        const moveit::core::JointModelGroup* niryo_realsense_joint_model_group,
                        const std::vector<std::string>& niryo_realsense_joint_names,
                        ros::NodeHandle nodeHandle):
            niryo_realsense_kinematic_state_(niryo_realsense_kinematic_state),
            niryo_realsense_joint_model_group(niryo_realsense_joint_model_group),
            niryo_realsense_joint_names(niryo_realsense_joint_names),
            nodeHandle(nodeHandle)
            {   
                niryo_joint_state_subscriber_ = nodeHandle.subscribe("/joint_states", 10, &NiryoManager::niryo_jointCallback, this);
                niryo_realsense_joint_state_publisher_ = nodeHandle.advertise<sensor_msgs::JointState>("/niryo_realsense/joint_states", 10, true);
                grasp_detection_client = nodeHandle.serviceClient<grasp_detection_srv::grasp_detector>("compute_grasp_detection");
                camera_extrinsics_client = nodeHandle.serviceClient<aruco_camera_extrinsics_srv::aruco_camera_extrinsics>("get_camera_extrinsics");
                grasp_pose_server = nodeHandle.advertiseService("compute_grasping_pose", &NiryoManager::grasp_pose_serviceCallback, this);
                tf_subscriber = nodeHandle.subscribe("/niryo_realsense/tf", 10, &NiryoManager::tf_non_static_callback, this);
                tf_static_subscriber = nodeHandle.subscribe("/niryo_realsense/tf_static", 10, &NiryoManager::tf_static_callback, this);

                // Loading the set of Grasp Pose Detection poses 

                nodeHandle.getParam("/grasp_detection_position", grasp_poses_param);
                std::vector<double> aux_vector;
                for(int ia = 0; ia < grasp_poses_param.size(); ++ia){
                    for(int ib = 0; ib < grasp_poses_param[ia].size(); ++ib){
                        aux_vector.push_back(double(grasp_poses_param[ia][ib]));
                    }
                    grasp_poses.push_back(aux_vector);
                    aux_vector.clear();
                }
            }

            // From the message published on "/joint_state" construct and publish the new message for "/niryo_realsense/joint_states"
            void niryo_jointCallback(const sensor_msgs::JointState::ConstPtr& msg){

                niryo_realsense_joint_state_msg.name.resize(niryo_realsense_joint_names.size());
                niryo_realsense_joint_state_msg.position.resize(niryo_realsense_joint_names.size());

                if(msg -> velocity.size() > 0){
                    niryo_realsense_joint_state_msg.velocity.resize(niryo_realsense_joint_names.size());
                }
                else{
                    niryo_realsense_joint_state_msg.velocity.resize(msg -> velocity.size());
                }
                if(msg -> effort.size() > 0){
                    niryo_realsense_joint_state_msg.effort.resize(niryo_realsense_joint_names.size());
                }
                else{
                    niryo_realsense_joint_state_msg.velocity.resize(msg -> effort.size());
                }

                size_t j = 0;
                for (size_t i = 0; i < msg -> name.size(); i++) {
                    
                    if (std::count(niryo_realsense_joint_names.begin(), niryo_realsense_joint_names.end(), msg -> name[i]) == 1){

                        niryo_realsense_joint_state_msg.position[j] = msg -> position[i];
                        niryo_realsense_joint_state_msg.name[j] = msg -> name[i];
                        if(msg -> velocity.size() > 0){
                            niryo_realsense_joint_state_msg.velocity[j] = msg -> velocity[i];
                        }
                        if(msg -> effort.size() > 0){
                            niryo_realsense_joint_state_msg.effort[j] = msg -> effort[i];
                        }
                        j++;

                    }
                }
                niryo_realsense_joint_state_msg.header.stamp = ros::Time::now();
                actual_joint_positions = niryo_realsense_joint_state_msg.position;
                niryo_realsense_joint_state_publisher_.publish(niryo_realsense_joint_state_msg);
            }

            int getGraspPoseID() const{
                for(int i=0; i < grasp_poses.size(); i++){
                    bool are_solutions_equal = true;
                    for(int j=0; j < grasp_poses[i].size() && are_solutions_equal; j++){
                        double diff;
                        
                        diff = angles::shortest_angular_distance(grasp_poses[i][j], actual_joint_positions[j]);
                        
                        // If the robot had not been highly inaccurate the threshold would have been different. (1e-3 or less)

                        if(std::fabs(diff) > 1e-1){
                            are_solutions_equal = false;
                        }
                    }
                    if(are_solutions_equal){
                        return i;
                    }
                }   
                return -1;
            }

            // This method handle the Grasp Pose Request from the "Real Niryo One".
            bool grasp_pose_serviceCallback(grasp_pose_service_msgs::grasp_pose::Request &req, grasp_pose_service_msgs::grasp_pose::Response &res){

                // In order to obtain the Grasp Pose is firstly needed that the Grasp Detector returns the variables obtained from the predicted Grasp Rectangle.
                grasp_detection_srv::grasp_detector srv;
                srv.request.number_of_grasps = 2;

                int gdp_id = this->getGraspPoseID();

                if(gdp_id == -1){
                    ROS_ERROR("The robot is not in a known grasp detection pose.");
                }
                else{
                    ROS_INFO("Identified Grasp Pose with ID: %d.", gdp_id);
                }
                
                grasp_detection_client.call(srv);

                // Loading the predicted variables.
                tf2::Vector3 grasp_vector;

                size_t number_of_grasps = srv.response.grasps.size();

                ROS_INFO("Has been identified %zu valid grasps.", number_of_grasps);

                for(size_t i = 0; i < number_of_grasps; i++){

                    // Debug prints.
                    std::ostringstream debug_os;

                    tf2::fromMsg(srv.response.grasps[i].vector, grasp_vector);
                    tf2::Quaternion grasp_quaternion;
                    grasp_quaternion.setRPY(0.0, 0.0, srv.response.grasps[i].angle);
                    
                    // Loading the transform and rotation matrix of the "camera_color_optical_frame" frame with respect to the "base_link" frame.

                    aruco_camera_extrinsics_srv::aruco_camera_extrinsics service;

                    camera_extrinsics_client.call(service);

                    tf2::Transform transform;
                    tf2::fromMsg(service.response.aruco_camera_extrinsics, transform);

                    double yaw, pitch, roll;
                    transform.getBasis().getRPY(roll, pitch, yaw);

                    tf2::Quaternion gripper_p;
                    gripper_p.setEuler(M_PI, 0.0, -yaw);


                    tf2::Vector3 cam_trans;
                    tf2::fromMsg(service.response.aruco_camera_extrinsics.translation, cam_trans);

                    // Obtaining the Grasp Vector of the resulting Grasp Pose.
                    tf2::Vector3 result_vector = transform*grasp_vector;
                    result_vector.setZ(std::max(result_vector.getZ(), 0.05));
                    
                    // Composing the actual rotation of the camera and the needed rotation for grasping, obtaining the orientation of the Grasp Pose.

                    tf2::Quaternion result_quaternion = gripper_p*grasp_quaternion;

                    debug_os << "The Grasp Point is:" << std::endl;
                    debug_os << "X: " << result_vector.getX() << std::endl;
                    debug_os << "Y: " << result_vector.getY() << std::endl;
                    debug_os << "Z: " << result_vector.getZ() << std::endl;

                    debug_os << "The Grasp Quaternion is:" << std::endl;
                    debug_os << "X: " << result_quaternion.getX() << std::endl;
                    debug_os << "Y: " << result_quaternion.getY() << std::endl;
                    debug_os << "Z: " << result_quaternion.getZ() << std::endl;
                    debug_os << "W: " << result_quaternion.getW() << std::endl;
                    
                    ROS_INFO_STREAM(debug_os.str());

                    // Obtaining the Eigen::Isometry3d for the Grasp Pose IK.
                    tf2::Transform result_transform(result_quaternion, result_vector);
                    geometry_msgs::Transform result_transform_msg = tf2::toMsg(result_transform);
                    const Eigen::Isometry3d& end_effector_state = tf2::transformToEigen(result_transform_msg);

                    // Obtaining the Eigen::Isometry3d for the Pre-Grasp Pose IK.

                    tf2::Vector3 pre_grasp_vector_end_effector = tf2::Vector3(0.0, 0.0, -0.05);
                    tf2::Vector3 pre_grasp_vector = result_transform*pre_grasp_vector_end_effector;
                    tf2::Transform pre_result_transform(result_quaternion, pre_grasp_vector);
                    geometry_msgs::Transform pre_result_transform_msg = tf2::toMsg(pre_result_transform);
                    const Eigen::Isometry3d& pre_end_effector_state = tf2::transformToEigen(pre_result_transform_msg);

                    // Pre-Grasp Pose IK.
                    double timeout = 1;
                    bool found_ik = niryo_realsense_kinematic_state_-> setFromIK(niryo_realsense_joint_model_group, pre_end_effector_state, timeout);
                    
                    if(found_ik){
                        std::vector<double> joint_values;
                        niryo_realsense_kinematic_state_ -> copyJointGroupPositions(niryo_realsense_joint_model_group,
                        res.pre_joint_values);

                        // Grasp Pose IK.
                        found_ik = niryo_realsense_kinematic_state_-> setFromIK(niryo_realsense_joint_model_group, end_effector_state, timeout);
                        
                        if(found_ik){

                            ROS_INFO("Inverted the %zu-th pose", i);

                            std::vector<double> joint_values;
                            niryo_realsense_kinematic_state_ -> copyJointGroupPositions(niryo_realsense_joint_model_group,
                            res.joint_values);
                            return true;
                        }
                    }
                }

                ROS_ERROR("NO GRASP POSE IS REACHABLE");
                
                return false;
            }

            // TF callback.
            void tf_non_static_callback(const ros::MessageEvent<tf2_msgs::TFMessage const>& msg_evt){
                tf_transform_callback(msg_evt, false);
            }

            // TF static callback.
            void tf_static_callback(const ros::MessageEvent<tf2_msgs::TFMessage const>& msg_evt){
                tf_transform_callback(msg_evt, true);
            }

            // TF callbacks handler.
            void tf_transform_callback(const ros::MessageEvent<tf2_msgs::TFMessage const>& msg_evt, bool is_static){
                const tf2_msgs::TFMessage& msg_in = *(msg_evt.getConstMessage());
                std::string authority = msg_evt.getPublisherName();
                for (unsigned int i = 0; i < msg_in.transforms.size(); i++){
                    try{
                        tfBuffer.setTransform(msg_in.transforms[i], authority, is_static);
                    }catch (tf2::TransformException& ex){
                        std::string temp = ex.what();
                        ROS_ERROR("Failure to set recieved transform from %s to %s with error: %s\n", msg_in.transforms[i].child_frame_id.c_str(), msg_in.transforms[i].header.frame_id.c_str(), temp.c_str());
                    }
                }
            }
};

int main(int argc, char **argv){

    ros::init(argc, argv, "niryo_manager");
    
    ros::NodeHandle nodeHandle;
    
    /*
        Loading the "Simulated Niryo One", the which one with the camera and the gripper, parameters from the parameter server"
    */
    robot_model_loader::RobotModelLoader niryo_realsense_robot_model_loader("niryo_realsense_description");
    const moveit::core::RobotModelPtr& niryo_realsense_kinematic_model = niryo_realsense_robot_model_loader.getModel();
    const moveit::core::JointModelGroup* niryo_realsense_joint_model_group = niryo_realsense_kinematic_model->getJointModelGroup("arm");
    const std::vector<std::string>& niryo_realsense_joint_names = niryo_realsense_joint_model_group->getVariableNames();
    moveit::core::RobotStatePtr niryo_realsense_kinematic_state(new moveit::core::RobotState(niryo_realsense_kinematic_model));

    NiryoManager niryo_manager( niryo_realsense_kinematic_state,
                                niryo_realsense_joint_model_group,
                                niryo_realsense_joint_names,
                                nodeHandle);
    while(nodeHandle.ok()){
        ros::spinOnce();
    }
    return 0;
}