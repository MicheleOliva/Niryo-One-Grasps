#!/usr/bin/env python3

import rospy
import argparse
import torch
import torch.utils.data
import numpy as np
from geometry_msgs.msg import Vector3
import sys
import os

import matplotlib.pyplot as plt
from cv_bridge import CvBridge

sys.path.append(os.path.dirname(os.path.dirname(__file__)) + "/robotic-grasping")

from utils.data.camera_data import CameraData
from utils.dataset_processing import image
from utils.dataset_processing.grasp import Grasp, detect_grasps
from inference.post_process import post_process_output

from grasp_detection_msg.msg import grasp_detection
from grasp_detection_srv.srv import grasp_detector, grasp_detectorResponse
from rect_center_to_point_srv.srv import rect_center_to_point, rect_center_to_pointRequest
from get_workspace_center_srv.srv import workspace_center, workspace_centerRequest, workspace_centerResponse

from rs_image_bundle_msg.msg import aligned_image_bundle as msg_Image


def parse_args():
    """
    Args parser.
    """
    parser = argparse.ArgumentParser(description='Robotic Grasping Server')
    parser.add_argument('--network', type=str, help='Path to saved network to be used')
    cfgs, _ = parser.parse_known_args()
    return cfgs
 

class GraspServer():
    """
    This class provides the 'compute_grasp_detection' service.
    """


    def __init__(self, network_path, max_gripper_opening):
        """
        GraspServer constructor.
        :param network_path: the path to the pretrainined network to use.
        :param max_gripper_opening: the maximum opening of the robot's gripper.
        """
        
        self.bridge = CvBridge()

        rospy.init_node('grasp_detection_server')

        rospy.logdebug(f"NETWORK: {network_path}")

        rospy.Subscriber("image_bundle", msg_Image, self.handle_image_bundle)

        # Loading device for torch and loading model.
        if torch.cuda.is_available():
            rospy.loginfo("CUDA detected. Running with GPU acceleration.")
            self.device = torch.device("cuda")
            rospy.loginfo("Loading model...")
            self.net = torch.load(network_path)
            rospy.loginfo('Done.')
        else:
            rospy.loginfo("CUDA is *NOT* detected. Running with only CPU.")
            self.device = torch.device("cpu")
            rospy.loginfo("Loading model...")
            self.net = torch.load(network_path, map_location ='cpu')
            rospy.loginfo('Done.')

        self.max_gripper_opening = max_gripper_opening

        rospy.wait_for_service('compute_image2point')
        self.image2point_proxy = rospy.ServiceProxy('compute_image2point', rect_center_to_point)

        rospy.wait_for_service('get_workspace_center')
        self.workspace_coordinates_proxy = rospy.ServiceProxy('get_workspace_center', workspace_center)

        _ = rospy.Service('compute_grasp_detection', grasp_detector, self.handle_grasp_detecting)


    def get_crop_attributes(self, output_size):

        center = self.workspace_coordinates_proxy()

        left = max((center.height - output_size // 2), 0)
        top = max((center.width - output_size // 2), 0)
        right = max((center.height + output_size // 2), output_size)
        bottom = max((center.width + output_size // 2), output_size)

        return (bottom, right), (top, left)


    def handle_image_bundle(self, data):
        self.image_bundle = {
                'rgb' : self.bridge.imgmsg_to_cv2(data.rgb, "rgb8"),
                'aligned_depth' : np.expand_dims(self.bridge.imgmsg_to_cv2(data.depth, "32FC1"), axis=2)
        }


    def handle_grasp_detecting(self, req):

        ng = req.number_of_grasps

        bottom_right, top_left = self.get_crop_attributes(224)

        rospy.loginfo("Received request.")

        rgb = self.image_bundle['rgb']
        depth = self.image_bundle['aligned_depth']

        rgb_img = image.Image(rgb)
        rgb_img.crop(bottom_right=bottom_right, top_left=top_left)

        plt.close('all')

        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
        ax.imshow(rgb_img.img)
        figure.canvas.draw()
        plt.show()

        rgb_img.normalise()
        rgb_img.img = rgb_img.img.transpose((2, 0, 1))

        depth_img = image.Image(depth)
        depth_img.crop(bottom_right=bottom_right, top_left=top_left)
        depth_img.normalise()
        depth_img.img = depth_img.img.transpose((2, 0, 1))

        x = torch.from_numpy(
            np.concatenate(
                (np.expand_dims(depth_img.img, 0),
                 np.expand_dims(rgb_img.img, 0)),
                1
            ).astype(np.float32)
        )

        rospy.loginfo("Images loaded.")

        with torch.no_grad():
            xc = x.to(self.device)
            rospy.loginfo("Inferencing...")
            pred = self.net.predict(xc)

            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
            
            del xc
            torch.cuda.empty_cache()
            
            grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=ng)
            rospy.loginfo("Done.")

        # Plotting.
        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(1, 1, 1)
        ax.imshow(rgb)

        valid_grasps = []

        for j in range(0,len(grasps)):            

            grasp =  Grasp([grasps[j].center[0] + top_left[0], grasps[j].center[1] + top_left[1]], grasps[j].angle, grasps[j].length, grasps[j].width)

            try:

                """
                    Searching for the local minima around the grasp center and using the found point as the depth value.
                """

                fake_grasp = Grasp([grasp.center[0], grasp.center[1]], grasp.angle, grasp.width, grasp.width)
                mask = np.ones(depth.shape)
                rr, cc = fake_grasp.as_gr.polygon_coords(depth.shape)
                mask[rr, cc] = depth[rr, cc]
                mask[mask <=0.28] = 1.0
                
                ind = np.unravel_index(np.argmin(mask, axis=None), mask.shape)

                req = rect_center_to_pointRequest()
                depth_value = depth[ind[0], ind[1]]

                ax.plot(ind[1], ind[0], marker='x', color="white")

                req.depth = depth[ind[0], ind[1]]
                req.coordinates = [grasp.center[0], grasp.center[1]]

                image_center = self.image2point_proxy(req)

            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s"%e)
                exit(1)
            
            rospy.loginfo(f"Grasp point: [{image_center.x}, {image_center.y}, {depth_value[0]}]")
            rospy.loginfo(f"The grasp angle is: {-grasp.angle} rad")
            grasp_point = Vector3()
            grasp_point.x = image_center.x
            grasp_point.y = image_center.y
            grasp_point.z = depth_value[0]

            valid_grasps.append(grasp_detection(grasp_point, - grasp.angle))

            ax.plot(grasp.center[1], grasp.center[0], marker='v', color="white")
            grasp.plot(ax)

        f.canvas.draw()
        plt.show()

        return grasp_detectorResponse(valid_grasps)



if __name__== '__main__':

    args = parse_args()

    while not rospy.has_param("/gripper"):
        rospy.sleep(0.5)

    gripper_info = rospy.get_param("/gripper")

    gs = GraspServer(network_path = args.network,
                     max_gripper_opening = gripper_info['max_opening'])
    
    rospy.spin()