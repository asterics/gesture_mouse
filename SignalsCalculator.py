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
        #self.random_ear_indices = np.random.choice(468, (10, 6), replace=False)
        # indices = self.ear_indices[:,0:6], d1= self.ear_indices[:,6:9], d2 = self.ear_indices[:,9:12]
        self.ear_indices = np.array([
            [78,81,311,308,402,178,0.0,-0.9378740000000008,0.38806799999999875,-8.612336,0.0,0.0],
            [61,39,269,291,405,181,0.0,-3.2111079999999994,-0.582358000000001,-9.824824,0.0,0.0],
            [57,39,269,287,405,181,0.0,-3.2111079999999994,-0.582358000000001,-12.410568,0.0,0.0],
            [17,15,12,0,40,91,3.753534,0.03198600000000029,-1.2462199999999992,0.0,3.917437999999999,0.8881320000000006],
            [17,15,12,0,270,291,-4.371116,0.5181109999999998,-1.786073,0.0,3.917437999999999,0.8881320000000006],
            [33,160,158,133,153,144,0.0,-1.2135689999999997,-0.18371100000000018,-5.178853999999999,-0.1574920000000004,1.168964],
            [362,385,387,263,373,380,0.0,-1.2135689999999997,-0.18371100000000018,-5.178853999999999,0.1574920000000004,-1.168964],
            [40,186,216,205,203,165,-2.2548919999999995,2.6113569999999995,1.188009,3.836036,4.5316399999999994,-1.7823980000000006],
            [91,43,202,211,194,182,-2.453915,-2.047639,1.2947430000000009,2.4908259999999998,-3.962192,-2.019084000000001],
            [270,391,423,425,436,410,-2.2548919999999995,-2.6113569999999995,-1.188009,-3.836036,4.5316399999999994,-1.7823980000000006],
            [321,273,422,431,418,406,2.453915,-2.047639,1.2947430000000009,-2.4908259999999998,-3.962192,-2.019084000000001],
            [425,266,329,348,347,280,-4.01056,0.1691879999999999,-1.6701860000000002,1.0644939999999998,4.797934,-0.5298939999999996],
            [205,50,118,119,100,36,-4.01056,-0.1691879999999999,1.6701860000000002,-1.0644939999999998,4.797934,-0.5298939999999996],
            [4,51,196,168,419,281,-1.989758,0.0,0.0,0.0,7.468394,-4.701129999999999],
            [218,220,275,438,274,237,-0.08659899999999993,-1.0574679999999999,-0.0053200000000002134,-4.504488,0.0,0.0],
            [46,53,65,55,222,225,0.022348000000000035,-1.6381720000000004,-0.9590930000000006,-8.064732000000001,0.5211660000000009,3.4857520000000006],
            [276,283,295,285,442,444,0.5247960000000003,-1.4374500000000006,-0.5293510000000001,8.064732000000001,0.5211660000000009,3.4857520000000006],
            [66,108,337,296,336,107,0.0,-3.174508000000001,0.7898519999999998,-11.041168,0.0,0.0]
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
            d1 = self.ear_indices[:, 6:9]
            d2 = self.ear_indices[:,9:12]
            cam_normal = np.array([0,0,1])
            rotated_d1 = np.matmul(rotmat, d1.T).T
            rotated_d2 = np.matmul(rotmat, d2.T).T
            projected_d1 = np.linalg.norm(np.cross(np.cross(cam_normal,rotated_d1),cam_normal))
            projected_d2 = np.linalg.norm(np.cross(np.cross(cam_normal,rotated_d2),cam_normal))
            correction_factor = projected_d2/projected_d1
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

        d1 = self.ear_indices[:, 6:9]
        d2 = self.ear_indices[:, 9:12]
        cam_normal = np.array([0, 0, 1])
        rotated_d1 = np.matmul(rotmat, d1.T).T
        rotated_d2 = np.matmul(rotmat, d2.T).T
        projected_d1 = np.linalg.norm(np.cross(np.cross(cam_normal, rotated_d1), cam_normal))
        projected_d2 = np.linalg.norm(np.cross(np.cross(cam_normal, rotated_d2), cam_normal))
        correction_factor = projected_d2 / projected_d1
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
        #assert indices.shape[1] == 10
        p2_p6 = np.linalg.norm(landmarks[indices[:, 1].astype(int)] - landmarks[indices[:, 5].astype(int)], axis=1)
        p3_p5 = np.linalg.norm(landmarks[indices[:, 2].astype(int)] - landmarks[indices[:, 4].astype(int)], axis=1)
        p1_p4 = np.linalg.norm(landmarks[indices[:, 0].astype(int)] - landmarks[indices[:, 3].astype(int)], axis=1)
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

