import sklearn.linear_model._base

import DrawingDebug
from PnPHeadPose import PnPHeadPose
from face_geometry import PCF, get_metric_landmarks
import monitor
import KalmanFilter1D

from scipy.spatial.transform import Rotation
import numpy as np
import cv2
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import normalize

from dataclasses import dataclass, fields
from typing import Tuple
from numbers import Number
from face_geometry import PCF, get_metric_landmarks


class FilteredFloat:
    def __init__(self, value: Number, filter_value: float = None):
        if filter_value is None:
            self.use_filter = False
            self.filter_R = 0.
        else:
            self.use_filter = True
            self.filter_R = filter_value
        self.filter = KalmanFilter1D.Kalman1D(R=self.filter_R ** 2)
        self.value = value

    def set(self, value):
        """
        Adds a new value to be filtered and returns the filtered value
        :param value: New value to be filtered
        """
        if self.use_filter:
            kalman = self.filter.update(value)
            self.value = np.real(kalman)
        else:
            self.value = value
        return self.value

    def get(self):
        return self.value

    def set_filter_value(self, filter_value):
        self.filter_R = filter_value
        self.use_filter = True
        self.filter = KalmanFilter1D.Kalman1D(R=self.filter_R ** 2)


class Filtered2D:
    def __init__(self, value: np.ndarray((2,)), filter_value: float = None):
        if filter_value is None:
            self.use_filter = False
            self.filter_R = 0.
        else:
            self.use_filter = True
            self.filter_R = filter_value
        self.filter = KalmanFilter1D.Kalman1D(R=self.filter_R ** 2)
        self.value = value

    def set(self, value):
        if self.use_filter:
            kalman = self.filter.update(value[0] + 1j * value[1])
            self.value[0], self.value[1] = (np.real(kalman), np.imag(kalman))
        else:
            self.value = value

    def get(self):
        return self.value

    def set_filter_value(self, filter_value):
        if filter_value > 0:
            self.use_filter = True
            self.filter = KalmanFilter1D.Kalman1D(R=filter_value ** 2)
        else:
            self.use_filter = False


@dataclass
class SignalsResult:
    rvec: np.ndarray
    tvec: np.ndarray
    nosetip: np.ndarray
    pitch: FilteredFloat
    yaw: FilteredFloat
    roll: FilteredFloat
    screen_xy: Filtered2D
    jaw_open: FilteredFloat
    mouth_puck: FilteredFloat
    debug1: FilteredFloat
    debug2: FilteredFloat
    debug3: FilteredFloat

    def __init__(self):
        self.rvec = np.zeros((3,))
        self.tvec = np.zeros((3,))
        self.nosetip = np.zeros((3,))
        self.pitch = FilteredFloat(0.)
        self.yaw = FilteredFloat(0.)
        self.roll = FilteredFloat(0.)
        self.jaw_open = FilteredFloat(0.)
        self.mouth_puck = FilteredFloat(0.)
        self.debug1 = FilteredFloat(0.)
        self.debug2 = FilteredFloat(0.)
        self.debug3 = FilteredFloat(0.)
        self.screen_xy = Filtered2D(np.zeros((2,)))


class SignalsCalculater:
    def __init__(self, camera_parameters, frame_size: Tuple[int, int]):
        self.neutral_landmarks = np.zeros((478, 3))
        self.camera_parameters = camera_parameters
        self.head_pose_calculator = PnPHeadPose()
        self.pcf = PCF(1, 10000, 720, 1280)
        self.frame_size = frame_size
        #####
        self.random_ear_indices = np.random.choice(468, (10, 6), replace=False)
        self.ear_indices = np.array([    #ear_indice[:,:6]:vertex index, ear_indices[:,6:9]:normal, ear_indices[:,9] area
            [78, 81, 311, 308, 402, 178, 0.0, -0.42996529795527055, -0.9028454145390757, 1.8274426314614112], # mouth inner
            [61, 39, 269, 291, 405, 181,  0.0, 0.2040910245076216, -0.9789519159363391 , 6.965312397209321],  # mouth outer
            [57, 39, 269, 287, 405, 181, -3.054010372444962e-17, 0.19873127411683472, -0.9800540192703152, 7.9764583345578925], # mouth outer fixed
            [17, 15, 12, 0, 40, 91, 0.3212820652962719, -0.21984308419466744, 0.9211117482969906, 3.115830982338437],  # mouth left
            [17, 15, 12, 0, 270, 291, 0.3711812550939365, 0.3573466573499048, -0.8570459978015998, 3.4556381580062356], # mouth right
            [33, 160, 158, 133, 153, 144, -0.19978355429549854, 0.14676994328387, -0.9687853813830528, 1.1367722762253367],  # left eye
            [362, 385, 387, 263, 373, 380,  0.19999291692562285, 0.14676098146597613, -0.9687435406229681, 1.1367719391562168], # right eye
            [40, 186, 216, 205, 203, 165, -0.47541740761254164, 0.006528390240965514, -0.8797361358156388, 3.960626388572969],  # cheeck left upper
            [91, 43, 202, 211, 194, 182, 0.5194157137308583, -0.09243240323843337, 0.8495078381987012, 2.9545777904087633], # cheeck left lower
            [270, 391, 423, 425, 436, 410, 0.474226383535992, -0.013172185571143313, -0.8803043966070223, 3.960051990701081],  # cheeck right upper
            [321, 273, 422, 431, 418, 406, 0.5194157137308583, 0.09243240323843337, -0.8495078381987012, 2.9545777904087633], # cheeck right lower
            [425, 266, 329, 348, 347, 280, 0.3709208733164779, -0.2003657706562296, -0.9067917421809026, 4.180198480141474], # upper cheeck right
            [205, 50, 118, 119, 100, 36, -0.3697713372049223, -0.19779594392096944, -0.9078248304326647, 4.175765494555663], # upper cheeck left
            [4, 51, 196, 168, 419, 281,  0.0, -0.5519289870388973, -0.8338911159535258, 3.015552169786011], # nose vert
            [218, 220, 275, 438, 274, 237, -0.18661803873499494, 0.1293311129911196, -0.9738825241430211, 1.2001167586923467], # nose hor
            [46, 53, 65, 55, 222, 225, -0.34753463146024194, 0.4684988128464185, -0.8122367526142179, 3.0242659712461393], # left eyebrow
            [276, 283, 295, 285, 442, 444, -0.3297187477116141, -0.45154345458259165, 0.8290923085103998, 2.60853924624981] , # right eyebrow
            [66, 108, 337, 296, 336, 107, -7.00808173621624e-18, -0.2282268394431662, -0.9736079856686588, 7.172903919610362] # between eyebrows
        ])

    def process(self, landmarks, linear_model:MultiOutputRegressor, labels, scaler):
        rvec, tvec = self.pnp_head_pose(landmarks)
        landmarks = landmarks * np.array((self.frame_size[0], self.frame_size[1],
                                          self.frame_size[0]))  # TODO: maybe move denormalization into methods
        landmarks = landmarks[:, :2]

        r = Rotation.from_rotvec(np.squeeze(rvec))

        #TODO: Calculate stuff if head is tilted too much (yaw and pitch), or supress signals. Maybe error is better then unwanted actions

        rotmat, _ = cv2.Rodrigues(rvec)
        angles = r.as_euler("xyz", degrees=True)
        # normalized_landmarks = rotationmat.T@(landmarks-tvec.T)
        jaw_open = self.get_jaw_open(landmarks)
        mouth_puck = self.get_mouth_puck(landmarks)
        l_brow_outer_up = self.cross_ratio_colinear(landmarks, [225, 46, 70, 71])
        r_brow_outer_up = self.cross_ratio_colinear(landmarks, [445, 276, 300, 301])
        brow_inner_up = self.five_point_cross_ratio(landmarks, [9, 69, 299, 65, 295])
        l_smile = self.cross_cross_ratio(landmarks, [216, 207, 214, 212, 206, 92])
        r_smile = self.cross_cross_ratio(landmarks, [436, 427, 434, 432, 426, 322])
        smile = 0.5 * (l_smile + r_smile)
        forehead_length = np.linalg.norm(landmarks[10,:]-landmarks[8,:])
        eye_distance = np.linalg.norm(landmarks[33, :] - landmarks[263, :])
        # TODO better check and logic


        signals = {
            "HeadPitch": angles[0],
            "HeadYaw": angles[1],
            "HeadRoll": angles[2],
            "JawOpen": jaw_open,
            "MouthPuck": mouth_puck,
            "BrowOuterUpLeft": l_brow_outer_up,
            "BrowOuterUpRight": r_brow_outer_up,
            "BrowInnerUp": brow_inner_up,
            "BrowInnerDown": brow_inner_up,
            "MouthSmile": smile
        }

        if len(labels) > 0:
            ear_values = np.array(self.eye_aspect_ratio_batch(landmarks, self.ear_indices)).reshape(1, -1)
            normals = self.ear_indices[:, 6:9]
            rotated_normal = np.matmul(rotmat, normals.T).T
            rotation_factor = np.maximum(abs(rotated_normal[:,2]),0.6)
            area = self.ear_indices[:, 9]
            correction_factor = 1 / rotation_factor
            ear_values = ear_values*correction_factor
            #ear_values = scaler.transform(ear_values)
            reg_result = linear_model.predict(ear_values)
            for i, label in enumerate(labels):
                if label == "neutral":
                    continue
                signals[label] = reg_result[0][i]


        return signals

    def process_ear(self, landmarks):
        rvec, tvec = self.pnp_head_pose(landmarks)

        rotmat, _ = cv2.Rodrigues(rvec)

        landmarks = landmarks * np.array((self.frame_size[0], self.frame_size[1],
                                          self.frame_size[0]))  # TODO: maybe move denormalization into methods
        landmarks = landmarks[:, :2]

        ear_values = self.eye_aspect_ratio_batch(landmarks, indices=self.ear_indices)
        normals = self.ear_indices[:,6:9]
        rotated_normal = np.matmul(rotmat,normals.T).T
        rotation_factor = np.maximum(abs(rotated_normal[:,2]),0.6)
        area = self.ear_indices[:,9]
        correction_factor = 1/rotation_factor
        print(rotation_factor[1])
        #forehead_length = np.linalg.norm(landmarks[10, :] - landmarks[8,:])
        #eye_distance = np.linalg.norm(landmarks[33,:] - landmarks[263,:])
        return ear_values, ear_values*correction_factor

    def process_neutral(self, landmarks):
        pass

    def pnp_head_pose(self, landmarks):
        screen_landmarks = landmarks[:, :2] * np.array(self.frame_size)
        rvec, tvec = self.head_pose_calculator.fit_func(screen_landmarks, self.camera_parameters)
        return rvec, tvec

    def pnp_reference_free(self, landmarks):
        idx = [33, 263, 1, 61, 291, 199]
        screen_landmarks = landmarks[idx, :2] * np.array(self.frame_size)
        landmarks_3d = landmarks[idx, :] * np.array([self.frame_size[0], self.frame_size[1], 1])
        fx, fy, cx, cy = self.camera_parameters

        # Initial fit
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(landmarks_3d, screen_landmarks,
                                                          camera_matrix, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=1)
        # Second fit for higher accuracy
        success, rvec, tvec = cv2.solvePnP(landmarks_3d, screen_landmarks, camera_matrix, None,
                                           rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

        return rvec, tvec

    def geometric_head_pose(self, landmarks):
        nose = [8, 9, 10, 151]
        eyes = [33, 133, 362, 263]
        nose_points = landmarks[nose, :]
        eye_points = landmarks[eyes, :]
        nose_mean = np.mean(nose_points, 0)
        eye_mean = np.mean(eye_points, 0)
        nose_centered = nose_points - nose_mean
        eye_centered = eye_points - eye_mean
        uu_nose, dd_nose, vv_nose = np.linalg.svd(nose_centered)
        up = vv_nose[0]
        uu_eye, dd_eye, vv_eye = np.linalg.svd(eye_centered)
        left = vv_eye[0]
        left = left - np.dot(left, up) * up
        left = left / np.linalg.norm(left)
        front = np.cross(up, left)
        R = [up, left, front]
        r = Rotation.from_matrix(R)
        return r.as_rotvec(), np.zeros((3, 1))

    def procrustes_head_pose(self, landmarks):
        landmarks = landmarks.T
        landmarks = landmarks[:, :468]
        metric_lm, pose_matrix = get_metric_landmarks(landmarks, self.pcf)
        rotatiom_matirx = pose_matrix[:3, :3]
        translation = pose_matrix[3, :3]
        rvec = Rotation.from_matrix(rotatiom_matirx)
        return rvec.as_rotvec(), translation

    def get_jaw_open(self, landmarks):
        mouth_distance = np.linalg.norm(landmarks[14, :] - landmarks[13, :])
        nose_tip = landmarks[1, :]
        chin_moving_landmark = landmarks[18, :]
        head_height = np.linalg.norm(landmarks[10, :] - landmarks[151, :])
        jaw_nose_distance = np.linalg.norm(nose_tip - chin_moving_landmark)
        normalized_distance = jaw_nose_distance / head_height
        return normalized_distance

    def get_mouth_puck(self, landmarks):
        left_distance = np.linalg.norm(landmarks[302] - landmarks[72])
        d = np.linalg.norm(landmarks[151, :] - landmarks[10, :])
        normalized_distance = left_distance / d
        return normalized_distance

    def cross_ratio_colinear(self, landmarks, indices):
        """
        Calculates the cross ratio of 4 "almost" colinear points
        :param landmarks: list of landmarks
        :param indices: indices of 4 landmarks to use
        :return: cross_ratio of the 4 points (is invariant under projective transformations
        """
        assert len(indices) == 4
        p1, p2, p3, p4 = landmarks[indices, :2]
        return (np.linalg.norm(p3 - p1) * np.linalg.norm(p4 - p2)) / (np.linalg.norm(p4 - p1) * np.linalg.norm(p3 - p2))

    def five_point_cross_ratio(self, landmarks, indices):
        """
        Calculates the cross ratio of 5 coplanar points
        :param landmarks: list of landmarks
        :param indices: indices of 5 landmarks to use
        :return: cross_ratio of the 5 points (is invariant under projective transformations
        """
#        assert len(indices) == 5
        p1, p2, p3, p4, p5 = landmarks[indices, :2]
        m124 = np.ones((3, 3))
        m124[:2, 0] = p1
        m124[:2, 1] = p2
        m124[:2, 2] = p4
        m135 = np.ones((3, 3))
        m135[:2, 0] = p1
        m135[:2, 1] = p3
        m135[:2, 2] = p5
        m125 = np.ones((3, 3))
        m125[:2, 0] = p1
        m125[:2, 1] = p2
        m125[:2, 2] = p5
        m134 = np.ones((3, 3))
        m134[:2, 0] = p1
        m134[:2, 1] = p3
        m134[:2, 2] = p4

        return (np.linalg.det(m124) * np.linalg.det(m135)) / (np.linalg.det(m125) * np.linalg.det(m134))

    def cross_cross_ratio(self, landmarks, indices):
        """

        :param landmarks: list of landmarks
        :param indices: indices of 6 landmarks to use
        :return: cross cross ratio
        """
       # assert len(indices) == 6

        return self.five_point_cross_ratio(landmarks, [indices[0]] + indices[2:6]) / self.five_point_cross_ratio(
            landmarks, [indices[1]] + indices[2:6])

    def eye_aspect_ratio(self, landmarks, indices):
        """
        Calculates the eye aspect ratio for the given indices. indices are in the order P1, P2, P3, P4, P5, P6.
        P1, P4 are the eye corners, P2 is opposite to P6 and P3 is opposite to P5.
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

        :param landmarks: landmarks in pixel coordinates
        :param indices: indices of points P1, P2, P3, P4, P5, P6, P1, P2, P3, P4, P5, P6.
        P1, P4 are the eye corners, P2 is opposite to P6 and P3 is opposite to P5
        :return: ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
        """
        assert len(indices) == 6
        p2_p6 = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[5]])
        p3_p5 = np.linalg.norm(landmarks[indices[2]] - landmarks[indices[4]])
        p1_p4 = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    def eye_aspect_ratio_batch(self, landmarks, indices):
        """
        Calculates the eye aspect ratio for the given indices. indices are in the order P1, P2, P3, P4, P5, P6.
        P1, P4 are the eye corners, P2 is opposite to P6 and P3 is opposite to P5.
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

        :param landmarks: landmarks in pixel coordinates
        :param indices: indices of points P1, P2, P3, P4, P5, P6, P1, P2, P3, P4, P5, P6.
        P1, P4 are the eye corners, P2 is opposite to P6 and P3 is opposite to P5
        :return: ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
        """
        assert indices.shape[1] == 10
        p2_p6 = np.linalg.norm(landmarks[indices[:, 1].astype(int)] - landmarks[indices[:, 5].astype(int)], axis=1)
        p3_p5 = np.linalg.norm(landmarks[indices[:, 2].astype(int)] - landmarks[indices[:, 4].astype(int)], axis=1)
        p1_p4 = np.linalg.norm(landmarks[indices[:, 0].astype(int)] - landmarks[indices[:, 3].astype(int)], axis=1)
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

