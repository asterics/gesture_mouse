import dataclasses
import random
import time
import uuid
from threading import Thread
import socket
import json
from typing import Dict
import os
from typing import List
from pathlib import Path
import pickle
import csv

import PIL.Image
import mediapipe as mp
import cv2
import numpy as np
import sklearn
from PySide6.QtCore import QThread
import keyboard
import pynput

from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer, MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, MultiTaskLassoCV, LassoLarsIC, LogisticRegression, RidgeClassifier, LassoLarsCV
from sklearn.svm import SVR, SVC, LinearSVR
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier, RegressorChain
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import normalize
import sklearn.preprocessing as preprocessing
from sklearn.metrics.pairwise import cosine_similarity, chi2_kernel, cosine_distances
from sklearn.pipeline import Pipeline

from scipy.spatial.transform import Rotation

import Mouse
import DrawingDebug
import SignalsCalculator
import monitor
from Signal import Signal
from KalmanFilter1D import Kalman1D
import FPSCounter
import util

from pyLiveLinkFace import PyLiveLinkFace, FaceBlendShape

mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_connections = mp.solutions.face_mesh_connections

colors = [(166,206,227),(31,120,180),(178,223,138),(51,160,44),(151,154,53),(227,26,28),(153,91,111),(255,127,0),(202,178,214),(106,61,154),(255,255,153),(177,89,40), (0,255,0), (0,0,255), (0,255,255), (255,255,255)]

class Demo(Thread):
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.is_tracking = False
        self.mouse_enabled = False
        self.mouse_absolute = True
        self.mouse: Mouse.Mouse = Mouse.Mouse()

        self.frame_width, self.frame_height = (640, 480)
        self.annotated_landmarks = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.int8)
        self.fps_counter = FPSCounter.FPSCounter(20)
        self.fps = 0
        self.cam_cap = None

        self.UDP_PORT = 11111
        self.my_ip = util.get_ip()
        self.socket = None
        self.webcam_dev_nr = 0
        self.vid_source_file=None

        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
               max_num_faces=1,
               refine_landmarks=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5)

        self.camera_parameters = (500, 500, 640 / 2, 480 / 2)
        self.signal_calculator = SignalsCalculator.SignalsCalculater(camera_parameters=self.camera_parameters,
                                                                     frame_size=(self.frame_width, self.frame_height))

        self.use_mediapipe = True
        self.filter_landmarks = True
        self.landmark_kalman = [Kalman1D(R=0.0065 ** 2) for _ in range(468)] #TODO: improve values, maybe move to calculator (mediapipe landmark smoothing calculator)

        # Calibration
        self.calibration_samples = dict()

        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #self.VideoWriter: cv2.VideoWriter = cv2.VideoWriter("dummy.mp4", self.fourcc, 30, (self.frame_height, self.frame_width))
        self.calibrate_neutral: bool = False
        self.neutral_signals = []
        self.pose_signals = []

        self.calibrate_pose: bool = False

        self.onehot_encoder = OneHotEncoder(sparse_output=False, dtype=float)
        self.scaler = StandardScaler()
        #self.linear_model = MultiOutputRegressor(SVR())
        self.linear_model = MLPClassifier()
        #self.linear_model = MultiOutputRegressor(KNeighborsRegressor(metric="cosine"))
        #self.linear_model = MultiOutputRegressor(GradientBoostingRegressor(max_features=6,verbose=1,loss="huber"))
        self.linear_signals: List[str] = []

        # add hotkey
        # TODO: how to handle activate mouse / toggle mouse etc. by global hotkey
        # keyboard.add_hotkey("esc", lambda: self.stop())
        keyboard.add_hotkey("alt + 1", lambda: self.toggle_gesture_mouse())  # TODO: Linux alternative
        keyboard.add_hotkey("m", lambda: self.toggle_mouse_mode())
        keyboard.add_hotkey("c", lambda: self.mouse.centre_mouse())
        #keyboard.on_press_key("r", lambda e: self.disable_gesture_mouse())
        #keyboard.on_release_key("r", lambda e: self.enable_gesture_mouse())
        # add mouse_events
        self.signals: Dict[str, Signal] = {}

        self.disable_gesture_mouse()

        self.write_csv = False
        self.csv_file_name = "log.csv" #TODO: or select
        self.csv_file_fp = None
        self.csv_writer = None

        print(self.csv_file_name)

    def run(self):
        self.is_running = True
        while self.is_running:
            if self.is_tracking:
                if self.use_mediapipe:
                    self.setup_signals("config/mediapipe_default.json") #TODO: change to latest
                    self.__start_camera()
                    self.__run_mediapipe()
                    self.__stop_camera()
                else:
                    self.setup_signals("config/iphone_default.json")
                    self.__start_socket()
                    self.__run_livelinkface()
                    self.__stop_socket()
            time.sleep(0.5)

    def __run_mediapipe(self):
        # TODO: split this up, it's getting crowded
        while self.is_running and self.is_tracking and self.cam_cap.isOpened() and self.use_mediapipe:
            success, image = self.cam_cap.read()
            if not success:
                print("couldn't read frame")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if not results.multi_face_landmarks:
                continue
            landmarks = results.multi_face_landmarks[0]

            np_landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in
                 landmarks.landmark])

            if self.filter_landmarks:
                for i in range(468):
                    kalman_filters_landm_complex = self.landmark_kalman[i].update(
                        np_landmarks[i, 0] + 1j * np_landmarks[i, 1])
                    np_landmarks[i, 0], np_landmarks[i, 1] = np.real(kalman_filters_landm_complex), np.imag(
                        kalman_filters_landm_complex)


            # only check videowriter not none? #
            ear_values, ear_values_corrected = self.signal_calculator.process_ear(np_landmarks)
            if self.calibrate_neutral and success:

                #self.VideoWriter.write(image)
                self.neutral_signals.append(ear_values_corrected)
                #continue

            if self.calibrate_pose and success:
                #self.VideoWriter.write(image)
                self.pose_signals.append(ear_values_corrected)
                #continue
            ########

            result = self.signal_calculator.process(np_landmarks, self.linear_model, self.linear_signals, self.scaler)

            for signal_name in self.signals:
                value = result.get(signal_name)
                if value is None:
                    print(f"Tracker doesn't measure signal {signal_name}")
                    continue

                self.signals[signal_name].set_value(value)

            self.mouse.process_signal(self.signals)

            # Debug
            #black = np.zeros((self.frame_height, self.frame_height, 3)).astype(np.uint8)
            annotated_img = DrawingDebug.draw_landmarks_fast(np_landmarks, image)
            for i, indices in enumerate(self.signal_calculator.ear_indices):
                annotated_img = DrawingDebug.draw_landmarks_fast(np_landmarks, annotated_img, index=indices[:6].astype(int), color=colors[i%len(colors)])
            self.annotated_landmarks = cv2.flip(annotated_img,1)
            if self.write_csv:
                gesture="neutral"
                if self.calibrate_pose or self.calibrate_neutral:
                    gesture = self.calibration_name
                elif keyboard.is_pressed("q"):
                    gesture="JawOpen"
                elif keyboard.is_pressed("w"):
                    gesture="Smile"
                elif keyboard.is_pressed("e"):
                    gesture="Frown"
                elif keyboard.is_pressed("r"):
                    gesture="CheekPuff"
                elif keyboard.is_pressed("t"):
                    gesture="MouthPuck"
                elif keyboard.is_pressed("z"):
                    gesture="BlinkLeft"
                elif keyboard.is_pressed("u"):
                    gesture="BlinkRight"
                elif keyboard.is_pressed("i"):
                    gesture="BrowUp"
                elif keyboard.is_pressed("o"):
                    gesture="BrowDown"
                elif keyboard.is_pressed("p"):
                    gesture="BrowUpLeft"
                elif keyboard.is_pressed("a"):
                    gesture="BrowUpRight"
                elif keyboard.is_pressed("s"):
                    gesture="NoseSneer"

                row = [time.time(),*np_landmarks.astype(np.float32).flatten(), *ear_values.astype(np.float32).flatten(), *ear_values_corrected.astype(np.float32).flatten(), gesture, *result.values()]
                self.csv_writer.writerow(row)
                print(gesture)
                print(row)


            self.fps = self.fps_counter()

    def __run_livelinkface(self):
        while self.is_running and self.is_tracking and not self.use_mediapipe:
            try:
                data, addr = self.socket.recvfrom(1024)
                success, live_link_face = PyLiveLinkFace.decode(data)
            except socket.error:
                success = False

            if success:
                row = [time.time()]
                blendshapes = []
                for blendshape in FaceBlendShape:
                    value = live_link_face.get_blendshape(blendshape)
                    blendshapes.append(value)
                    row.append(value)
                    self.signals[blendshape.name].set_value(value)
                #Calibration
                blendshapes = np.array(blendshapes)

                if len(self.linear_signals) > 0:
                    reg_result = self.linear_model.predict(blendshapes.reshape(1, -1))
                    for i, label in enumerate(self.linear_signals):
                        if label == "neutral":
                            continue
                        self.signals.get(label).set_value(reg_result[0][i])

                if self.calibrate_neutral and success:
                    #TODO: Ignore Head/Eye Pose?
                    # self.VideoWriter.write(image)
                    self.neutral_signals.append(blendshapes)
                    # continue

                if self.calibrate_pose and success:
                    # self.VideoWriter.write(image)
                    self.pose_signals.append(blendshapes)

                if self.write_csv:
                    self.csv_writer.writerow(row)
                if self.mouse_enabled:
                    self.mouse.process_signal(self.signals)

    def __start_camera(self):
        if self.vid_source_file:
            self.cam_cap = cv2.VideoCapture(self.vid_source_file)
        else:
            self.cam_cap = cv2.VideoCapture(self.webcam_dev_nr)
        #self.cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        #self.cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        #self.cam_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P',
        #                                                             'G'))  # From https://forum.opencv.org/t/videoio-v4l2-dev-video0-select-timeout/8822/4 for linux

    def __stop_camera(self):
        if self.cam_cap is not None:
            self.cam_cap.release()

    def __start_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind(("", self.UDP_PORT))

    def __stop_socket(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def stop_tracking(self):
        print("Stopping tracking..")
        self.is_tracking = False

    def start_tracking(self):
        print("Starting tracking..")
        self.is_tracking = True

    def stop(self):
        print("Stopping Demo..")
        self.is_running = False
        if self.csv_file_fp is not None:
            self.csv_file_fp.close()

    def update_webcam_device_selection(self,device_nr):
        print(f"Setting camera with device nr {device_nr}")
        self.webcam_dev_nr=int(device_nr)
        # unset video source file for now.
        # TODO: Use enum to have radio logic between the 3 modes.
        self.vid_source_file=None

    def update_webcam_video_file_selection(self,vid_source_file):
        print(f"Setting camera with video file {vid_source_file}")
        self.vid_source_file=vid_source_file

    def disable_gesture_mouse(self):
        # Disables gesture mouse and enables normal mouse input
        self.mouse_enabled = False
        self.mouse.disable_gesture()

    def enable_gesture_mouse(self):
        # Disables normal mouse and enables gesture mouse
        self.mouse_enabled = True
        self.mouse.enable_gesture()

    def toggle_gesture_mouse(self):
        # Toggles between gesture and normal mouse
        if self.mouse_enabled:
            self.disable_gesture_mouse()
        else:
            self.enable_gesture_mouse()

    def set_filter_value(self, name: str, filter_value: float):
        signal = self.signals.get(name, None)
        if signal is not None:
            signal.set_filter_value(filter_value)

    def set_use_mediapipe(self, selected: bool):
        self.use_mediapipe = selected

    def set_filter_landmarks(self, enabled: bool):
        self.filter_landmarks = enabled

    def start_write_csv(self, file_name:str):
        self.csv_file_name = file_name
        self.csv_file_fp = open(self.csv_file_name, "w+", newline="")
        self.csv_writer = csv.writer(self.csv_file_fp, delimiter=";")
        if self.use_mediapipe:
            row = ["time"]
            for i in range(478):
                row.append(f"landmark_{i}_x")
                row.append(f"landmark_{i}_y")
                row.append(f"landmark_{i}_z")
            for i in range(len(self.signal_calculator.ear_indices)):
                row.append(f"ear_{i}")
            for i in range(len(self.signal_calculator.ear_indices)):
                row.append(f"corrected_ear_{i}")
            row.append("Gesture")
            for signal in self.signals.keys():
                row.append(signal)
            self.csv_writer.writerow(row)
        else:
            row = ["time"]
            for signal in self.signals:
                row.append(signal)
            self.csv_writer.writerow(row)
        self.write_csv = True

    def stop_write_csv(self):
        self.write_csv = False
        if self.csv_file_fp is not None:
            self.csv_file_fp.close()
            self.csv_file_fp = None
            self.csv_writer = None

    def toggle_mouse_mode(self):
        self.mouse.toggle_mode()

    def setup_signals(self, json_path: str):
        """
        Reads a config file and setup ups the available signals.
        :param json_path: Path to json
        """
        parsed_settings = json.load(open(json_path, "r"))
        self.signals = dict()
        parsed_signals = parsed_settings.get("signals")
        for json_signal in parsed_signals:
            # read values
            name = json_signal["name"]
            lower_threshold = json_signal["lower_threshold"]
            higher_threshold = json_signal["higher_threshold"]
            filter_value = json_signal["filter_value"]

            # construct signal
            signal = Signal(name)
            signal.set_filter_value(filter_value)
            signal.set_threshold(lower_threshold, higher_threshold)
            self.signals[name] = signal
        gesture_model = parsed_settings.get("gesture_model")
        if gesture_model is not None:
            gesture_save_location=gesture_model.get("gesture_model_location")
            encoder_save_location = gesture_model.get("encoder_location")
            calibration_samples_location = gesture_model.get("calibration_samples_location")
            if gesture_save_location is not None and os.path.exists(gesture_save_location):
                with open(gesture_save_location, "br") as fp:
                    self.linear_model = pickle.load(fp)
            else:
                f"File not found: {gesture_save_location}"
            if encoder_save_location is not None and os.path.exists(encoder_save_location):
                with open(encoder_save_location, "br") as fp:
                    self.onehot_encoder = pickle.load(fp)
                self.linear_signals = self.onehot_encoder.categories_[0]
            else:
                f"File not found: {encoder_save_location}"
            if calibration_samples_location is not None and os.path.exists(calibration_samples_location):
                with open(calibration_samples_location, "br") as fp:
                    self.calibration_samples = pickle.load(fp)
            else:
                f"File not found: {encoder_save_location}"


    def save_signals(self, save_location):
        # save_location = "path/to/file.json"
        path = Path(save_location)
        folder = path.parent
        profile_name = path.stem
        settings_dict = {}
        signals = []
        for signal_name in self.signals:
            signal = self.signals[signal_name]
            name = signal.name
            lower_threshold = signal.lower_threshold
            higher_threshold = signal.higher_threshold
            filter_value = signal.raw_value.filter_R
            signals.append({
                "name": name,
                "lower_threshold": lower_threshold,
                "higher_threshold": higher_threshold,
                "filter_value": filter_value
            })
        settings_dict["signals"] = signals
        gesture_model_location = folder / profile_name / "gesture_model.pkl"
        encoder_location = folder / profile_name / "encoder.pkl"
        calibration_samples_location = folder / profile_name / "calibration_samples.pkl"
        gesture_model_location.parent.mkdir(exist_ok=True,parents=True)
        encoder_location.parent.mkdir(exist_ok=True,parents=True)
        calibration_samples_location.parent.mkdir(exist_ok=True,parents=True)
        settings_dict["gesture_model"] = {
            "gesture_model_location": str(gesture_model_location.relative_to(Path(".").absolute())),
            "encoder_location": str(encoder_location.relative_to(Path(".").absolute())),
            "calibration_samples_location": str(calibration_samples_location.relative_to(Path(".").absolute()))
        }
        with open(path, "w+") as fp:
            json.dump(settings_dict,fp, indent=2)
        with open(gesture_model_location, "bw+") as fp:
            pickle.dump(self.linear_model, fp)
        with open(encoder_location, "bw+") as fp:
            pickle.dump(self.onehot_encoder, fp)
        with open(calibration_samples_location, "bw+") as fp:
            pickle.dump(self.calibration_samples, fp)


    def calibrate_signal(self, calibration_sample, name):
        neutral_samples = np.array(calibration_sample[name]["neutral"])
        pose_samples = np.array(calibration_sample[name]["pose"])
        neutral_samples = neutral_samples[len(neutral_samples) // 4:3 * len(neutral_samples) // 4]
        pose_samples = pose_samples[len(pose_samples) // 4:3 * len(neutral_samples) // 4]
        signal = self.signals.get(name)
        min_value = max_value = 0
        if signal is not None:
            min_value = np.percentile(neutral_samples, 65)
            max_value = np.percentile(pose_samples, 35)
            signal.set_threshold(min_value, max_value)
        return min_value, max_value

    # Combine these methods?
    def calibrate_neutral_start(self, name):
        self.neutral_signals=[]
        self.pose_signals=[]
        #if not os.path.exists(f"calibration/{name}"):
        #    os.mkdir(f"calibration/{name}")
        #self.VideoWriter.open(f"calibration/{name}/{name}_neutral.mp4", self.fourcc, 30, (self.frame_width,self.frame_height))
        self.calibrate_neutral = True
        self.calibration_name = "calibration_neutral"

    def calibrate_neutral_stop(self, name):
       #self.VideoWriter.release()
        self.calibrate_neutral = False
    def calibrate_pose_start(self, name):
        #if not os.path.exists(f"calibration/{name}"):
        # os.mkdir(f"calibration/{name}")
        #self.VideoWriter.open(f"calibration/{name}/{name}_pose.mp4", self.fourcc, 30, (self.frame_width,self.frame_height))
        self.calibrate_pose = True
        self.calibration_name = "calibration_"+name
    def calibrate_pose_stop(self, name):
        #self.VideoWriter.release()
        self.calibrate_pose = False
        print("Accepting calibration samples")
        self.calibration_samples[name] = {"neutral": self.neutral_signals, "pose": self.pose_signals}
    #####
    def recalibrate(self, name):
        print(f"=== Recalibrating === with f{len(self.calibration_samples)}")
        new_linear_model = sklearn.clone(self.linear_model)
        if len(self.calibration_samples)==0:
            print("Nothing to calibrate")
            return
        data_array = []
        label_array = []
        unique_labels = ["neutral"]
        for pose_name in self.calibration_samples:
            unique_labels.append(pose_name)
            for label, data in self.calibration_samples[pose_name].items():
                data = data[20:len(data)-20]
                data_array.extend(data)
                if label == "neutral":
                    label_array.extend(["neutral"]*len(data))
                else:
                    label_array.extend([pose_name] * len(data))
        data_array = np.array(data_array)
        label_array = np.array(label_array).reshape(-1,1)

        self.onehot_encoder.fit(label_array)
        y = self.onehot_encoder.transform(label_array)

        #self.scaler.fit(data_array)
        #data_array=self.scaler.transform(data_array)

        self.signals[name] = Signal(name)
        self.signals[name].set_higher_threshold(1.)
        self.signals[name].set_lower_threshold(0.)
        self.signals[name].set_filter_value(0.0001)

        new_linear_model.fit(data_array, y)
        self.linear_model = new_linear_model
        self.linear_signals = self.onehot_encoder.categories_[0]
        #print(self.linear_model.classes_)
        #print(self.onehot_encoder.inverse_transform(self.linear_model.classes_))

if __name__ == '__main__':
    demo = Demo()
    demo.run()
    bla = input("Press any key to stop")
