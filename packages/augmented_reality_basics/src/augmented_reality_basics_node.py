#!/usr/bin/env python3

# import external libraries
import rospy
import os
import cv2
import numpy as np
import rospkg
import yaml
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel


# import DTROS-related classes
from duckietown.dtros import \
    DTROS, \
    NodeType, \
    TopicType, \
    DTParam, \
    ParamType

# import messages and services

from std_msgs.msg import Float32
# from duckietown_msgs.msg import \


from sensor_msgs.msg import CompressedImage, CameraInfo

class Augmenter(DTROS):
    def __init__(self, node_name):
        super(Augmenter, self).__init__(
            node_name=node_name,
            node_type=NodeType.VISUALIZATION
        )
        rospack = rospkg.RosPack()
        self.alpha = 0.0
        self.homography = self.load_extrinsics()
        self.H = np.array(self.homography).reshape((3, 3))
        self.Hinv = np.linalg.inv(self.H)




        self.sub_compressed_img = rospy.Subscriber(
            "/robot_name/camera_node/image/compressed",
            CompressedImage,
            self.callback,
            queue_size=1
        )

        self.sub_camera_info = rospy.Subscriber(
            "/robot_name/camera_node/camera_info",
            CameraInfo,
            self.cb_camera_info,
            queue_size=1
        )

        self.pub_map_img = rospy.Publisher(
            "~calibration_pattern/image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
        )

    def callback(self, msg):
        bridge = CvBridge()
        header = msg.header
        cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_img = self.render_segments(cv_img)
        msg_out = bridge.cv2_to_compressed_imgmsg(cv_img, dst_format='jpeg')
        msg_out.header = header

        self.pub_map_img.publish(msg_out)

    def cb_camera_info(self, msg):
        # unsubscribe from camera_info
        self.loginfo('Camera info message received. Unsubscribing from camera_info topic.')
        # noinspection PyBroadException
        try:
            self.sub_camera_info.shutdown()
        except BaseException:
            self.loginfo('BaseException')
            pass
        # ---
        self.H_camera, self.W_camera = msg.height, msg.width
        # create new camera info
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)
        # find optimal rectified pinhole camera

        self.rect_camera_K, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_model.K,
            self.camera_model.D,
            (self.W_camera, self.H_camera),
            self.alpha
            )

        # create rectification map
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.camera_model.K,
            self.camera_model.D,
            None,
            self.rect_camera_K,
            (self.W_camera, self.H_camera),
            cv2.CV_32FC1
            )


    def load_camera_info(self, filename):
        """Loads the camera calibration files.
        Loads the intrinsic camera calibration.
        Args:
            filename (:obj:`str`): filename of calibration files.
        Returns:
            :obj:`CameraInfo`: a CameraInfo message object
        """
        with open(filename, 'r') as stream:
            calib_data = yaml.load(stream)
        cam_info = CameraInfo()
        cam_info.width = calib_data['image_width']
        cam_info.height = calib_data['image_height']
        cam_info.K = calib_data['camera_matrix']['data']
        cam_info.D = calib_data['distortion_coefficients']['data']
        cam_info.R = calib_data['rectification_matrix']['data']
        cam_info.P = calib_data['projection_matrix']['data']
        cam_info.distortion_model = calib_data['distortion_model']
        return cam_info



    def readYamlFile(self,fname):
        """
        Reads the YAML file in the path specified by 'fname'.
        E.G. :
            the calibration file is located in : `/data/config/calibrations/filename/DUCKIEBOT_NAME.yaml`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


    def draw_segment(self, image, pt_x, pt_y, color):
        defined_colors = {
            'red': ['rgb', [1, 0, 0]],
            'green': ['rgb', [0, 1, 0]],
            'blue': ['rgb', [0, 0, 1]],
            'yellow': ['rgb', [1, 1, 0]],
            'magenta': ['rgb', [1, 0, 1]],
            'cyan': ['rgb', [0, 1, 1]],
            'white': ['rgb', [1, 1, 1]],
            'black': ['rgb', [0, 0, 0]]}
        _color_type, [r, g, b] = defined_colors[color]
        cv2.line(image, (pt_x[0], pt_x[1]), (pt_y[0], pt_y[1]), (b * 255, g * 255, r * 255), 5)
        return image



    def process_image(self, img):
        res = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_CUBIC)
        # res = cv2.undistort(img, self.camera_model.K, self.camera_model.D, None, self.rect_camera_K)
        return res



    def ground2pixel(self, ground_point):
        pixel_norm = np.dot(self.Hinv, ground_point)
        return_point = [int(pixel_norm[0]/pixel_norm[2]), int(pixel_norm[1]/pixel_norm[2])]
        return return_point





    def render_segments(self, cv_img):
        cam_info = self.load_camera_info('/data/config/calibrations/camera_intrinsic/' + rospy.get_namespace().strip('/') + '.yaml')
        yaml_file = self.readYamlFile(rospack.get_path('augmented_reality_basics') + '/src/maps/calibration_pattern.yaml')
        segments_ = yaml_file['segments']
        points_ = yaml_file['points']
        img_tmp = self.process_image(cv_img)
        # img_tmp = cv_img
        for i in range(len(segments_)):
            point_a = [points_[segments_[i]['points'][0]][1][0],
                       points_[segments_[i]['points'][0]][1][1],
                       1]
            point_b = [points_[segments_[i]['points'][1]][1][0],
                       points_[segments_[i]['points'][1]][1][1],
                       1]
            pixel_a = self.ground2pixel(np.array(point_a))
            pixel_b = self.ground2pixel(np.array(point_b))
            color = segments_[i]['color']
            img_tmp = self.draw_segment(img_tmp, pixel_a, pixel_b, color)
        return img_tmp

    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.
        Returns:
            :obj:`numpy array`: the loaded homography matrix
        """
        # load intrinsic calibration
        cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log("Can't find calibration file: %s.\n Using default calibration instead."
                     % cali_file, 'warn')
            cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = 'Found no calibration file ... aborting'
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        try:
            with open(cali_file, 'r') as stream:
                calib_data = yaml.load(stream)
        except yaml.YAMLError:
            msg = 'Error in parsing calibration file %s ... aborting' % cali_file
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        return calib_data['homography']




if __name__ == '__main__':
    rospack = rospkg.RosPack()
    augmenter = Augmenter(node_name='augmented_reality_basics_node')


    rospy.spin()
