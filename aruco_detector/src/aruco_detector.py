#!/usr/bin/env python3

import numpy as np
import rospy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Transform

from camera_deprojection_srv.srv import camera_deprojection, camera_deprojectionRequest
from get_workspace_center_srv.srv import workspace_center, workspace_centerResponse
from aruco_camera_extrinsics_srv.srv import aruco_camera_extrinsics, aruco_camera_extrinsicsResponse
from rs_image_bundle_msg.msg import aligned_image_bundle as msg_Image

import cv2
# Aruco marker
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

# Initialize the detector parameters using default values
aruco_parameters = cv2.aruco.DetectorParameters_create()

from cv_bridge import CvBridge

### Constant Definition ###
ID_TO_FIND = rospy.get_param("ID_TO_FIND")
MARKER_SIZE = rospy.get_param("MARKER_SIZE")
R_WORLD_TO_ARUCO = np.asarray(rospy.get_param("R_WORLD_TO_ARUCO"))
P_ARUCO_WRT_WORLD = np.asarray(rospy.get_param("P_ARUCO_WRT_WORLD"))
WORKSPACE_CENTER_TO_ARUCO = np.asarray(rospy.get_param("WORKSPACE_CENTER_TO_ARUCO"))


class ArucoDetector():

    def __init__(self):

        rospy.init_node('aruco_detector')

        self.bridge = CvBridge()
        rospy.Subscriber("image_bundle", msg_Image, self.handle_image_bundle)

        rospy.wait_for_service('compute_point2image')
        self.point2image_proxy = rospy.ServiceProxy('compute_point2image', camera_deprojection)

        _ = rospy.Service('get_workspace_center', workspace_center, self.handle_image_coordinates)
        _ = rospy.Service('get_camera_extrinsics', aruco_camera_extrinsics, self.handle_camera_extrinsics)


    def handle_image_bundle(self, data):
        self.rgb_image = self.bridge.imgmsg_to_cv2(data.rgb, "rgb8")


    def handle_camera_extrinsics(self, req):

        # Obtaining R_{Camera}^{Tag} and the position of the camera in tag frame

        rot_tag_camera = self.rot_camera_tag.T
        pos_tag_camera = - np.matmul(rot_tag_camera, self.pos_camera_tag)

        # Camera position and attitude w.r.t. world

        pos_cam_in_world = P_ARUCO_WRT_WORLD + np.matmul(R_WORLD_TO_ARUCO, pos_tag_camera)
        
        rot_cam_in_world = np.matmul(R_WORLD_TO_ARUCO, rot_tag_camera)

        response = aruco_camera_extrinsicsResponse()

        t = Transform()

        # camera position w.r.t. world

        t.translation.x = pos_cam_in_world[0, 0]
        t.translation.y = pos_cam_in_world[1, 0]
        t.translation.z = pos_cam_in_world[2, 0]

        # camera orientation wrt world

        rot_cam_in_world_quat = R.from_matrix(rot_cam_in_world).as_quat()

        t.rotation.x = rot_cam_in_world_quat[0]
        t.rotation.y = rot_cam_in_world_quat[1]
        t.rotation.z = rot_cam_in_world_quat[2]
        t.rotation.w = rot_cam_in_world_quat[3]

        response.aruco_camera_extrinsics = t

        return response


    def handle_image_coordinates(self, req):

        gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)

        markerCorners, _, _ = cv2.aruco.detectMarkers(image=gray,dictionary=aruco_dictionary, parameters=aruco_parameters)

        self.aruco_center = np.array(markerCorners[0][0]).mean(axis=0).astype(np.int)

        intrinsic_matrix = np.asarray(rospy.get_param("camera_intrinsics"))
        distorsion = np.asarray(rospy.get_param("camera_distorsion"))

        ret = cv2.aruco.estimatePoseSingleMarkers(markerCorners, MARKER_SIZE, 
                                                  intrinsic_matrix, 
                                                  distorsion)

        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

        # camera -> tag
        self.rot_camera_tag = np.matrix(cv2.Rodrigues(rvec)[0])
        self.pos_camera_tag = np.asarray([tvec]).T * 0.01

        # Workspace center w.r.t. camera

        wsc = self.pos_camera_tag + np.matmul(self.rot_camera_tag, np.matrix(WORKSPACE_CENTER_TO_ARUCO))

        request = camera_deprojectionRequest()

        request.x = wsc[0, 0]
        request.y = wsc[1, 0]
        request.z = wsc[2, 0]

        point = self.point2image_proxy(request)

        res = workspace_centerResponse()

        res.height = int(point.coordinates[0])
        res.width =  int(point.coordinates[1])

        plt.close('all')
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
        ax.imshow(self.rgb_image)
        ax.plot(res.height, res.width, marker='v', color="red")
        figure.canvas.draw()
        plt.show()

        return res


if __name__== '__main__':

    aruco_detector = ArucoDetector()
    
    rospy.spin()
