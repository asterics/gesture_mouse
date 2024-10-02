import dataclasses
import queue
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
from queue import Queue

import mediapipe as mp
import cv2
import numpy as np
import sklearn
import keyboard
from case_insensitive_dict import CaseInsensitiveDict

from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer, MinMaxScaler
from sklearn.svm import SVR, SVC, LinearSVR

from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier, RegressorChain


import Mouse
import DrawingDebug
import SignalsCalculator
from Gesture import GestureSignal
from KalmanFilter1D import Kalman1D
import FPSCounter
import util

from pyLiveLinkFace import PyLiveLinkFace, FaceBlendShape

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import face_landmarker

model_path = './data/model/face_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
mp_Blendshapes = face_landmarker.Blendshapes
VisionRunningMode = mp.tasks.vision.RunningMode

mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_connections = mp.solutions.face_mesh_connections

colors = [(166, 206, 227), (31, 120, 180), (178, 223, 138), (51, 160, 44), (151, 154, 53), (227, 26, 28),
          (153, 91, 111), (255, 127, 0), (202, 178, 214), (106, 61, 154), (255, 255, 153), (177, 89, 40), (0, 255, 0),
          (0, 0, 255), (0, 255, 255), (255, 255, 255)]

VID_RES_X=320
VID_RES_Y=240

class Demo(Thread):
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.is_tracking = False
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
        self.vid_source_file = None

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=True,
            #result_callback = self.mp_callback
        )

        self.face_mesh = FaceLandmarker.create_from_options(options)

        self.camera_parameters = (480, 480, 640 / 2, 480 / 2)
        self.signal_calculator = SignalsCalculator.SignalsCalculater(camera_parameters=self.camera_parameters,
                                                                     frame_size=(self.frame_width, self.frame_height))

        self.use_mediapipe = True
        self.filter_landmarks = True
        self.landmark_kalman = [Kalman1D(R=0.0065 ** 2) for _ in range(
            468)]  # TODO: improve values

        # Calibration
        self.calibration_samples = dict()

        self.calibrate_neutral: bool = False
        self.neutral_signals = []
        self.pose_signals = []

        self.calibrate_pose: bool = False

        self.onehot_encoder = OneHotEncoder(sparse_output=False, dtype=float)
        self.scaler = Normalizer()
        self.means = np.ones((18, 1))
        self.linear_model = MultiOutputRegressor(SVR(kernel="rbf"))
        self.linear_signals: List[str] = []

        # add hotkey
        # TODO: how to handle activate mouse / toggle mouse etc. by global hotkey
        # keyboard.add_hotkey("esc", lambda: self.stop())
        # alt + 1: toggle gestures + mouse movement
        # alt + g: toggle gestures (keyboard and mouse)
        # alt + m: toggle mouse movement
        keyboard.add_hotkey("alt + 1", lambda: self.toggle_gesture_mouse())  # TODO: Linux alternative
        keyboard.add_hotkey("alt + g", lambda: self.toggle_gestures())
        keyboard.add_hotkey("alt + m", lambda: self.toggle_mouse_movement())
        keyboard.add_hotkey("alt + t", lambda: self.toggle_tracking())
        keyboard.add_hotkey("m", lambda: self.toggle_mouse_mode())
        keyboard.add_hotkey("c", lambda: self.mouse.centre_mouse())
        keyboard.add_hotkey(".", lambda: self.mouse.switch_monitor())
        keyboard.add_hotkey("t", lambda: self.mouse.toggle_tracking_mode())
        # keyboard.on_press_key("r", lambda e: self.disable_gesture_mouse())
        # keyboard.on_release_key("r", lambda e: self.enable_gesture_mouse())
        # add mouse_events
        self.signals: CaseInsensitiveDict[str, GestureSignal] = {}

        self.disable_gesture_mouse()

        self.write_csv = False
        self.csv_file_name = "log.csv"
        self.csv_file_fp = None
        self.csv_writer = None

        self.recording_mode = False  # TODO: Enum?
        self.iphone_csv_fp = None
        self.iphone_csv_writer = None
        self.mediapipe_csv_fp = None
        self.mediapipe_csv_writer = None

        self.image_q = Queue(3)

    def run(self):
        self.is_running = True
        while self.is_running:
            if self.is_tracking:
                if self.use_mediapipe:
                    self.setup_signals("config/mediapipe_blendshape.json")  # TODO: change to latest
                    self.__start_camera()
                    self.__run_mediapipe()
                    self.__stop_camera()
                elif self.recording_mode:
                    self.__start_camera()
                    self.__start_socket()
                    print("Recording Mode")
                    self.__csv_recording()
                    self.__stop_camera()
                    self.__stop_socket()
                else:
                    self.setup_signals("config/iphone_default.json")
                    self.__start_socket()
                    self.__run_livelinkface()
                    self.__stop_socket()
            time.sleep(0.01)

    def __run_mediapipe(self):
        while self.is_running and self.is_tracking and self.cam_cap.isOpened() and self.use_mediapipe:
            success, image = self.cam_cap.read()
            if not success:
                print("couldn't read frame")
                continue

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            timestamp = int(1000 * time.time())
            # results = self.face_mesh.process(image)
            image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            result = self.face_mesh.detect_for_video(image_mp, timestamp)
            self.mp_callback(result, image_mp, timestamp)

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

                # Pose for Mouse
                self.signals["upDown"].set_value(-live_link_face.get_blendshape(FaceBlendShape.headPitch))
                self.signals["leftRight"].set_value(-live_link_face.get_blendshape(FaceBlendShape.headYaw))
                # Calibration
                blendshapes = np.array(blendshapes)

                if len(self.linear_signals) > 0:
                    reg_result = self.linear_model.predict(blendshapes.reshape(1, -1))
                    for i, label in enumerate(self.linear_signals):
                        if label == "neutral":
                            continue
                        self.signals.get(label).set_value(reg_result[0][i])

                if self.calibrate_neutral and success:
                    # self.VideoWriter.write(image)
                    self.neutral_signals.append(blendshapes)
                    # continue

                if self.calibrate_pose and success:
                    # self.VideoWriter.write(image)
                    self.pose_signals.append(blendshapes)

                if self.write_csv:
                    self.csv_writer.writerow(row)
                if self.mouse.mouse_movement_enabled:
                    self.mouse.process_signal(self.signals)

    def __start_camera(self):
        if self.vid_source_file:
            self.cam_cap = cv2.VideoCapture(self.vid_source_file)
        else:
            start=int(1000*time.time())
            #self.cam_cap = cv2.VideoCapture(self.webcam_dev_nr, cv2.CAP_DSHOW)
            #on Linux there is no DSHOW available, so let opencv decide which API to choose.
            self.cam_cap = cv2.VideoCapture(self.webcam_dev_nr)
            self.cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH,VID_RES_X)
            self.cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VID_RES_Y)
            print(f"Starting camera took {int(1000*time.time())-start}, resolution={self.cam_cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

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

    def __csv_recording(self):
        while self.is_running and self.is_tracking and self.cam_cap.isOpened() and self.recording_mode:
            mp_success, image = self.cam_cap.read()

            try:
                data, addr = self.socket.recvfrom(1024)
                ip_success, live_link_face = PyLiveLinkFace.decode(data)
            except socket.error:
                ip_success = False

            if not mp_success or not ip_success:
                print(f"mediapipe: {mp_success}, iphone: {ip_success}, skipping frame")
                continue

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            timestamp_ms = int(1000 * time.time())
            # results = self.face_mesh.process(image)
            image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            result = self.face_mesh.detect_for_video(image_mp, timestamp_ms)

            if not result.face_landmarks:
                print("No face detected")
                continue
            transformation_matrix = result.facial_transformation_matrixes[0]
            print(transformation_matrix)
            mp_landmarks = result.face_landmarks[0]
            blendshapes = result.face_blendshapes[0]

            np_landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in
                 mp_landmarks])

            ear_values, ear_values_corrected = self.signal_calculator.process_ear(np_landmarks,
                                                                                  facial_transformation_matrix=transformation_matrix)
            mp_blendshape = []
            for blendshape in blendshapes:
                mp_blendshape.append(blendshape.score)

            image = image_mp.numpy_view().copy()
            annotated_img = DrawingDebug.draw_landmarks_fast(np_landmarks, image)

            for i, indices in enumerate(self.signal_calculator.ear_indices):
                annotated_img = DrawingDebug.draw_landmarks_fast(np_landmarks, annotated_img,
                                                                 index=indices[:6].astype(int),
                                                                 color=colors[i % len(colors)])
            self.annotated_landmarks = cv2.flip(annotated_img, 1)

            blendshapes = []
            for blendshape in FaceBlendShape:
                value = live_link_face.get_blendshape(blendshape)
                blendshapes.append(value)

            if self.write_csv:
                mp_row = [timestamp_ms, *np_landmarks.astype(np.float32).flatten(), *transformation_matrix.astype(np.float32).flatten(),
                          *ear_values.astype(np.float32).flatten(),
                          *ear_values_corrected.astype(np.float32).flatten(), *mp_blendshape]
                ip_row = [timestamp_ms, *blendshapes]
                self.mediapipe_csv_writer.writerow(mp_row)
                self.iphone_csv_writer.writerow(ip_row)

    def toggle_tracking(self):
        print("toggle tracking..")
        if self.is_tracking:
            self.stop_tracking()
        else:
            self.start_tracking()
    def stop_tracking(self):
        print("Stopping tracking..")
        self.is_tracking = False

    def start_tracking(self):
        print("Starting tracking..")
        self.is_tracking = True

    def stop(self):
        print("Stopping Demo..")
        self.stop_tracking()
        self.__stop_camera()
        self.__stop_socket()
        self.is_running = False
        self.face_mesh.close()
        if self.csv_file_fp is not None:
            self.csv_file_fp.close()

    def update_webcam_device_selection(self, device_nr):
        print(f"Setting camera with device nr {device_nr}")
        self.webcam_dev_nr = int(device_nr)
        # unset video source file for now.
        # TODO: Use enum to have radio logic between the 3 modes.
        self.vid_source_file = None

    def update_webcam_video_file_selection(self, vid_source_file):
        print(f"Setting camera with video file {vid_source_file}")
        self.vid_source_file = vid_source_file

    def toggle_gestures(self):
        if self.mouse.mouse_gesture_enabled:
            self.disable_gestures()
        else:
            self.enable_gestures()

    """
    Disables alls configured gestures (mouse and keyboard).
    """
    def disable_gestures(self):
        print("disabling gestures")
        # Disables gesture mouse and enables normal mouse input
        for signal in self.signals.values():
            signal.set_actions_active(False)
        self.mouse.disable_gestures()

    """
    Disables all gestures and mouse movement.
    """
    def disable_gesture_mouse(self):
        self.disable_mouse_movement()
        self.disable_gestures()

    """
    Enables all gestures (mouse and keyboard)
    """
    def enable_gestures(self):
        print("enabling gestures")
        for signal in self.signals.values():
            signal.set_actions_active(True)

        self.mouse.enable_gestures()

    """
    Enables all gestures and the mouse.
    """
    def enable_gesture_mouse(self):
        # Enables gesture mouse and enables normal mouse input
        self.enable_mouse_movement()
        self.enable_gestures()

    def enable_mouse_movement(self):
        self.mouse.enable_mouse_movement()
    def disable_mouse_movement(self):
        self.mouse.disable_mouse_movement()

    def toggle_mouse_movement(self):
        self.mouse.toggle_mouse_movement()

    def toggle_gesture_mouse(self):
        print("toggling mouse and gestures")
        # Toggles between gesture and normal mouse
        if self.mouse.mouse_movement_enabled:
            self.disable_gesture_mouse()
        else:
            self.enable_gesture_mouse()

    def set_use_mediapipe(self, selected: bool):
        self.use_mediapipe = selected

    def set_filter_landmarks(self, enabled: bool):
        self.filter_landmarks = enabled

    def start_write_csv(self, file_path: str):
        if self.recording_mode:
            path = Path(file_path)
            file_name = path.name

            iphone_csv_file_name = path.parent / ("iphone_" + file_name)
            mediapipe_csv_file_name = path.parent / ("mediapipe_" + file_name)

            self.iphone_csv_fp = open(iphone_csv_file_name, "w+", newline="")
            self.iphone_csv_writer = csv.writer(self.iphone_csv_fp, delimiter=";")

            self.mediapipe_csv_fp = open(mediapipe_csv_file_name, "w+", newline="")
            self.mediapipe_csv_writer = csv.writer(self.mediapipe_csv_fp, delimiter=";")

            mediapipe_header = ["time"]
            iphone_header = ["time"]

            for i in range(478):
                mediapipe_header.append(f"landmark_{i}_x")
                mediapipe_header.append(f"landmark_{i}_y")
                mediapipe_header.append(f"landmark_{i}_z")
            for i in range(16):
                mediapipe_header.append(f"transformation_matrix_{i}")
            for i in range(len(self.signal_calculator.ear_indices)):
                mediapipe_header.append(f"ear_{i}")
            for i in range(len(self.signal_calculator.ear_indices)):
                mediapipe_header.append(f"corrected_ear_{i}")
            for blendshape in mp_Blendshapes:
                mediapipe_header.append(blendshape.name)
            for blendshape in FaceBlendShape:
                iphone_header.append(blendshape.name)

            self.iphone_csv_writer.writerow(iphone_header)
            self.mediapipe_csv_writer.writerow(mediapipe_header)

        else:
            self.csv_file_name = file_path
            print(file_path)
            self.csv_file_fp = open(self.csv_file_name, "w+", newline="")
            self.csv_writer = csv.writer(self.csv_file_fp, delimiter=";")
            if self.use_mediapipe:
                row = ["time"]
                for i in range(478):
                    row.append(f"landmark_{i}_x")
                    row.append(f"landmark_{i}_y")
                    row.append(f"landmark_{i}_z")
                for i in range(16):
                    row.append(f"transformation_matrix_{i}")
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
        if self.iphone_csv_fp is not None:
            self.iphone_csv_fp.close()
            self.iphone_csv_fp = None
            self.iphone_csv_writer = None
        if self.mediapipe_csv_fp is not None:
            self.mediapipe_csv_fp.close()
            self.mediapipe_csv_fp = None
            self.mediapipe_csv_writer = None

    def toggle_mouse_mode(self):
        self.mouse.toggle_mode()

    def setup_signals(self, json_path: str):
        """
        Reads a config file and setup ups the available signals.
        :param json_path: Path to json
        """
        parsed_settings = json.load(open(json_path, "r"))

        # only reset self.signals if it is None, otherwise there could be configured actions that we don't want to override.
        if self.signals is None: self.signals = CaseInsensitiveDict()

        parsed_signals = parsed_settings.get("signals")
        for json_signal in parsed_signals:
            # read values
            name = json_signal["name"]
            lower_threshold = json_signal["lower_threshold"]
            higher_threshold = json_signal["higher_threshold"]
            filter_value = json_signal["filter_value"]

            # construct signal
            signal = GestureSignal(name)
            signal.set_filter_value(filter_value)
            signal.set_threshold(lower_threshold, higher_threshold)

            # if there are already configured actions, reassign them to the signal[name].actions property
            if name in self.signals:
                signal.actions=self.signals[name].actions

            self.signals[name] = signal
        gesture_model = parsed_settings.get("gesture_model")
        if gesture_model is not None:
            gesture_save_location = gesture_model.get("gesture_model_location")
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
        gesture_model_location.parent.mkdir(exist_ok=True, parents=True)
        encoder_location.parent.mkdir(exist_ok=True, parents=True)
        calibration_samples_location.parent.mkdir(exist_ok=True, parents=True)
        settings_dict["gesture_model"] = {
            "gesture_model_location": str(gesture_model_location.relative_to(Path(".").absolute())),
            "encoder_location": str(encoder_location.relative_to(Path(".").absolute())),
            "calibration_samples_location": str(calibration_samples_location.relative_to(Path(".").absolute()))
        }
        with open(path, "w+") as fp:
            json.dump(settings_dict, fp, indent=2)
        with open(gesture_model_location, "bw+") as fp:
            pickle.dump(self.linear_model, fp)
        with open(encoder_location, "bw+") as fp:
            pickle.dump(self.onehot_encoder, fp)
        with open(calibration_samples_location, "bw+") as fp:
            pickle.dump(self.calibration_samples, fp)

    def delete_signal(self, name:str):
        self.signals.pop(name,None)

        if name in self.linear_signals:
            self.calibration_samples.pop(name)
            self.recalibrate()

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
        self.neutral_signals = []
        self.pose_signals = []
        # if not os.path.exists(f"calibration/{name}"):
        #    os.mkdir(f"calibration/{name}")
        # self.VideoWriter.open(f"calibration/{name}/{name}_neutral.mp4", self.fourcc, 30, (self.frame_width,self.frame_height))
        self.calibrate_neutral = True
        self.calibration_name = "calibration_neutral"

    def calibrate_neutral_stop(self, name):
        # self.VideoWriter.release()
        self.calibrate_neutral = False

    def calibrate_pose_start(self, name):
        # if not os.path.exists(f"calibration/{name}"):
        # os.mkdir(f"calibration/{name}")
        # self.VideoWriter.open(f"calibration/{name}/{name}_pose.mp4", self.fourcc, 30, (self.frame_width,self.frame_height))
        self.calibrate_pose = True
        self.calibration_name = "calibration_" + name

    def calibrate_pose_stop(self, name):
        # self.VideoWriter.release()
        self.calibrate_pose = False
        print("Accepting calibration samples")
        self.calibration_samples[name] = {"neutral": self.neutral_signals, "pose": self.pose_signals}

    #####
    def recalibrate(self) -> bool:
        print(f"=== Recalibrating === with f{len(self.calibration_samples)}")

        # clone linear model so we only write to linear_model when calibration is finished (better for asynchronous)
        new_linear_model = sklearn.clone(self.linear_model)
        if len(self.calibration_samples) == 0:
            print("Nothing to calibrate")
            return False

        # convert data to numpy
        data_array = []
        label_array = []
        unique_labels = ["neutral"]

        for pose_name in self.calibration_samples:
            unique_labels.append(pose_name)
            for label, data in self.calibration_samples[pose_name].items():
                data = data[20:len(data) - 20] # cut start and end put
                data_array.extend(data)
                if label == "neutral":
                    label_array.extend(["neutral"] * len(data))
                else:
                    label_array.extend([pose_name] * len(data))
        data_array = np.array(data_array)
        label_array = np.array(label_array).reshape(-1, 1)

        # Onehot encoding
        self.onehot_encoder.fit(label_array)
        y = self.onehot_encoder.transform(label_array)


        self.means = np.mean(data_array, axis=0)

        # Fit the model
        new_linear_model.fit(data_array, y)
        self.linear_model = new_linear_model
        self.linear_signals = self.onehot_encoder.categories_[0]

        return True

    def add_signal(self, name):
        self.signals[name] = GestureSignal(name)
        self.signals[name].set_higher_threshold(1.)
        self.signals[name].set_lower_threshold(0.)
        self.signals[name].set_filter_value(0.0001)

    def mp_callback(self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        image = output_image.numpy_view().copy()

        # No face detected
        if not result.face_landmarks:
            self.update_video_display(image)
            print("Face not detected")
            return

        # get landmarks, transformation(head pose) and blendshapes

        transformation_matrix = result.facial_transformation_matrixes[0]
        mp_landmarks = result.face_landmarks[0]
        blendshapes = result.face_blendshapes[0]

        # convert to numpy
        np_landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in
             mp_landmarks])

        # Kalman filter landmarks (x,y coordinates)
        if self.filter_landmarks:
            for i in range(468):
                kalman_filters_landm_complex = self.landmark_kalman[i].update(
                    np_landmarks[i, 0] + 1j * np_landmarks[i, 1])
                np_landmarks[i, 0], np_landmarks[i, 1] = np.real(kalman_filters_landm_complex), np.imag(
                    kalman_filters_landm_complex)

        if self.mouse.tracking_mode == Mouse.TrackingMode.PNP:
            rvec, tvec = self.signal_calculator.pnp_head_pose(np_landmarks)
            transformation_matrix = np.ones((4,4))
            rotmat, _ = cv2.Rodrigues(rvec)
            transformation_matrix[:3,:3] = rotmat
            transformation_matrix[3,:3] = tvec.squeeze()

        ear_values, ear_values_corrected = self.signal_calculator.process_ear(np_landmarks,
                                                                              facial_transformation_matrix=transformation_matrix,
                                                                              random_augmentation=(
                                                                                      self.calibrate_pose or self.calibrate_neutral),
                                                                              tracking_mode=self.mouse.tracking_mode)
        # record calibration samples
        if self.calibrate_neutral:
            self.neutral_signals.append(ear_values_corrected)

        if self.calibrate_pose:
            self.pose_signals.append(ear_values_corrected)


        # calculate head pose and custom blendshapes/gestures
        result = self.signal_calculator.process(np_landmarks, self.linear_model, self.linear_signals,
                                                transformation_matrix, self.means,tracking_mode=self.mouse.tracking_mode)

        #read mediapipe blendshapes
        for blendshape in blendshapes:
            result[blendshape.category_name] = blendshape.score

        # Filter result, set value of signal. GestureSignal triggers appropriate action
        for signal_name in self.signals:
            value = result.get(signal_name, 0.)
            if value is None:
                print(f"Tracker doesn't measure signal {signal_name}")
                continue

            self.signals[signal_name].set_value(value)

        # Move mouse and do clicks
        self.mouse.process_signal(self.signals)

        # Debug Image

        # black = np.zeros((self.frame_height, self.frame_height, 3)).astype(np.uint8) # for only keypoints
        annotated_img = DrawingDebug.draw_landmarks_fast(np_landmarks, image)
        #annotated_img = image
        for i, indices in enumerate(self.signal_calculator.ear_indices):
            annotated_img = DrawingDebug.draw_landmarks_fast(np_landmarks, annotated_img, index=indices[:6].astype(int),
                                                             color=colors[i % len(colors)])

        self.update_video_display(annotated_img)
        self.update_csv(np_landmarks,transformation_matrix, ear_values, ear_values_corrected,result)
        #print(f"Time since frame read in ms {int(time.time()*1000) - timestamp_ms}, processing done")

    def update_video_display(self, image):
        self.annotated_landmarks = cv2.flip(image, 1)
        # TODO: Check if the image_q instructions can be removed
        if self.image_q.full():
            self.image_q.get()
        self.image_q.put(cv2.flip(image, 1))
        self.fps = self.fps_counter()

    def update_csv(self, np_landmarks, transformation_matrix, ear_values, ear_values_corrected, result):
        # record csv and also gesture for data capturing
        if self.write_csv:
            gesture = "neutral"
            if self.calibrate_pose or self.calibrate_neutral:
                gesture = self.calibration_name
            elif keyboard.is_pressed("q"):
                gesture = "jawOpen"
            elif keyboard.is_pressed("w"):
                gesture = "smile"
            elif keyboard.is_pressed("e"):
                gesture = "frown"
            elif keyboard.is_pressed("r"):
                gesture = "cheekPuff"
            elif keyboard.is_pressed("t"):
                gesture = "mouthPucker"
            elif keyboard.is_pressed("z"):
                gesture = "blinkLeft"
            elif keyboard.is_pressed("u"):
                gesture = "blinkRight"
            elif keyboard.is_pressed("i"):
                gesture = "browUp"
            elif keyboard.is_pressed("o"):
                gesture = "browDown"
            elif keyboard.is_pressed("p"):
                gesture = "browUpLeft"
            elif keyboard.is_pressed("a"):
                gesture = "browUpRight"
            elif keyboard.is_pressed("s"):
                gesture = "noseSneer"

            row = [time.time(), *np_landmarks.astype(np.float32).flatten(), *transformation_matrix.astype(np.float32).flatten(),*ear_values.astype(np.float32).flatten(),
                   *ear_values_corrected.astype(np.float32).flatten(), gesture, *result.values()]
            self.csv_writer.writerow(row)



if __name__ == '__main__':
    demo = Demo()
    demo.start()
    demo.start_tracking()
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    while True:
        try:
            img = demo.image_q.get(block=False)
            img = cv2.putText(img, demo.mouse.tracking_mode.name, (20, 40), font,
                                fontScale, color, thickness, cv2.LINE_AA)
            img = cv2.putText(img, demo.mouse.mode.name, (400, 40), font,
                                fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
        except queue.Empty:
            time.sleep(0.01)
