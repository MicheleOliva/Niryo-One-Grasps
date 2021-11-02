#!/usr/bin/env python3

import rospy
import argparse
import sys
import os
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import pyrealsense2.pyrealsense2 as rs

from rs_image_bundle_msg.msg import aligned_image_bundle as msg_Image
from rect_center_to_point_srv.srv import rect_center_to_point, rect_center_to_pointResponse
from camera_deprojection_srv.srv import camera_deprojection, camera_deprojectionResponse

sys.path.append(os.path.dirname(os.path.dirname(__file__)) + "/robotic-grasping")

from hardware.camera import RealSenseCamera

def parse_args():
   """
   Args parser.
   """
   parser = argparse.ArgumentParser(description='Image Bundle Publisher Args')
   parser.add_argument('--cam_id', type=str, help='RealSense Camera ID')
   cfgs, _ = parser.parse_known_args()
   return cfgs

class RealSenseHandler():
   """
   RealSense camera handler. Publishes on the topic '\image_bundle' and provides the 'compute_image2point' service.
   """

   def __init__(self, cam_id):
      """
      RealSenseHandler constructor.
      :param cam_id: Intel RealSennse camera identifier.
      """
      self.bridge = CvBridge()

      rospy.init_node('RealSense_Aligned_Images_Publisher')
      rospy.loginfo("Connecting to camera...")
      self.cam = RealSenseCamera(device_id=cam_id, fps=15)
      self.cam.connect()
      rospy.loginfo("Done.")

      rospy.set_param("camera_distorsion", self.cam.intrinsics.coeffs)

      intrinsic_matrix =  [[self.cam.intrinsics.fx, 0,                      self.cam.intrinsics.ppx],
                           [0,                      self.cam.intrinsics.fy, self.cam.intrinsics.ppy],
                           [0,                      0,                      1]]

      rospy.set_param("camera_intrinsics", intrinsic_matrix)

      self.pub = rospy.Publisher('image_bundle', msg_Image, queue_size=10)

      _ = rospy.Service('compute_image2point', rect_center_to_point, self.handle_image2point)
      _ = rospy.Service('compute_point2image', camera_deprojection, self.handle_point2image)


   def handle_point2image(self, data):
      """
      Given the coordinates in the 'camera_color_optical_frame' frame of a point, returns the coordinates of that point in the image plane.
      :param data: the 'camera_deprojection' request message: the coordinates of the point in the 'camera_color_optical_frame' frame.
      :return: the 'camera_deprojection' response message: the coordinates (y,x) of the pointin the image plane.
      """

      y,x = rs.rs2_project_point_to_pixel(self.cam.intrinsics, [data.x, data.y, data.z])

      res = camera_deprojectionResponse()
      res.coordinates = [int(y), int(x)]

      return res

   def handle_image2point(self, data):
      """
      Given the coordinates in the image plane of a point, returns the position that point with respect to the 'camera_color_optical_frame' frame.
      :param data: the 'rect_center_to_point' request message: the coordinates of the point in the image plane (y,x) and the (scaled) depth value.
      :return: the 'rect_center_to_point' response message: the coordinates (x,y,z) of the point with respect to the 'camera_color_optical_frame' frame.
      """

      x,y,z=rs.rs2_deproject_pixel_to_point(self.cam.intrinsics, [data.coordinates[1], data.coordinates[0]], data.depth)

      res = rect_center_to_pointResponse()
      res.x = x
      res.y = y
      res.z = z

      return res

   def publish(self):
      """
      Publishes the camera image bundle necessary for the Grasp Rectangle prediction on the topic '\image_bundle'.
      """
      frames = self.cam.pipeline.wait_for_frames()

      align = rs.align(rs.stream.color)
      aligned_frames = align.process(frames)
      color_frame = aligned_frames.first(rs.stream.color)
      aligned_depth_frame = aligned_frames.get_depth_frame()

      depth_frame = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)

      depth_frame *= self.cam.scale

      rgb_frame = np.asanyarray(color_frame.get_data())

      try:
         rgb_msg = self.bridge.cv2_to_imgmsg(rgb_frame, "rgb8")
         depth_msg = self.bridge.cv2_to_imgmsg(depth_frame, "32FC1")
      except CvBridgeError as e:
         print(e)
      
      msg = msg_Image()
      msg.depth = depth_msg
      msg.rgb = rgb_msg

      self.pub.publish(msg)

if __name__== '__main__':

   args = parse_args()

   r_publisher = RealSenseHandler(cam_id=int(args.cam_id))

   while not rospy.is_shutdown():
      r_publisher.publish()