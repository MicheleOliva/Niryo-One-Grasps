import pyrealsense2.pyrealsense2 as rs
import cv2
import numpy as np
import math


### Constant Definition ###
ID_TO_FIND = 0
MARKER_SIZE = 7  # [cm]
R_WORLD_TO_ARUCO = np.asarray([ [  0,  1, 0 ],
                                [ -1,  0, 0 ],
                                [  0,  0, 1 ]])

P_ARUCO_WRT_WORLD = np.asarray([[0.414],[0.003],[0]])

WORKSPACE_CENTER_TO_ARUCO = np.asarray([[0],[-0.128],[0]])

font = cv2.FONT_HERSHEY_PLAIN


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


if __name__ == '__main__':
    
    # Connect to camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('841512070267')
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    cfg = pipeline.start(config)

    # Determine intrinsics
    rgb_profile = cfg.get_stream(rs.stream.color)
    intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

    intrinsic_matrix = np.asarray([[intrinsics.fx, 0, intrinsics.ppx],
                                   [0, intrinsics.fy, intrinsics.ppy],
                                   [0, 0, 1]])

    distorsion = np.asarray(intrinsics.coeffs)

    # Determine depth scale
    scale = cfg.get_device().first_depth_sensor().get_depth_scale()

    # Aruco marker
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    while True:
        frames = pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= scale
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(image=gray, dictionary=dictionary,
                                                                               parameters=parameters)

        if markerIds is not None and markerIds[0] == ID_TO_FIND:
            ret = cv2.aruco.estimatePoseSingleMarkers(markerCorners, MARKER_SIZE, intrinsic_matrix, distorsion)

            rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

            cv2.aruco.drawDetectedMarkers(color_image, markerCorners)
            cv2.aruco.drawAxis(color_image, intrinsic_matrix, distorsion, rvec, tvec, 10)

            str_position = "Marker Position x=%4.0f y=%4.0f z=%4.0f" % (tvec[0], tvec[1], tvec[2])
            R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
            roll, pitch, yaw = rotationMatrixToEulerAngles(R_ct)
            str_orientation = "Marker Orientation r=%4.0f p=%4.0f y=%4.0f" % (math.degrees(roll),
                                                                              math.degrees(pitch),
                                                                              math.degrees(yaw))
            cv2.putText(color_image, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(color_image, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(color_image, str_orientation, (0, 255), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # camera -> tag
            rot_camera_tag = np.matrix(cv2.Rodrigues(rvec)[0])
            pos_camera_tag = np.asarray([tvec]).T * 0.01

            # Workspace center w.r.t. camera

            wsc = pos_camera_tag + np.matmul(rot_camera_tag, np.matrix(WORKSPACE_CENTER_TO_ARUCO))

            image_coordinates = rs.rs2_project_point_to_pixel(intrinsics, wsc)

            color_image = cv2.circle(color_image, (int(image_coordinates[0]), int(image_coordinates[1])), radius=2, color=(0, 0, 255), thickness=-1)


        cv2.imshow('frame', color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):

            cv2.imwrite(filename="/home/michele/catkin_ws/src/Niryo-One-Grasping/grasp_detector/scripts/pose2.png", img=np.asanyarray(color_frame.get_data()))
            cv2.destroyAllWindows()
            break
