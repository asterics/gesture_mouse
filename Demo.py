import dataclasses
import time
from threading import Thread
import socket
import json
from typing import Dict
import os

import PIL.Image
import mediapipe as mp
import cv2
import numpy as np
from PySide6.QtCore import QThread
import keyboard



import Mouse
import DrawingDebug
import SignalsCalculator
import monitor
from Signal import Signal
from KalmanFilter1D import Kalman1D
import FPSCounter

from pyLiveLinkFace import PyLiveLinkFace, FaceBlendShape

mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_connections = mp.solutions.face_mesh_connections

colors = [(166,206,227),(31,120,180),(178,223,138),(51,160,44),(251,154,153),(227,26,28),(253,191,111),(255,127,0),(202,178,214),(106,61,154),(255,255,153),(177,89,40), (0,255,0), (0,0,255), (0,255,255), (255,255,255)]
class Demo(QThread):
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.mouse_enabled = False
        self.mouse_absolute = True
        self.mouse: Mouse.Mouse = Mouse.Mouse()

        self.frame_width, self.frame_height = (640, 480)
        self.annotated_landmarks = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.int8)
        self.fps_counter = FPSCounter.FPSCounter(20)
        self.fps = 0
        self.cam_cap = None

        self.UDP_PORT = 11111
        self.socket = None

        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
               max_num_faces=1,
               refine_landmarks=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5)

        self.camera_parameters = (500, 500, 640 / 2, 480 / 2)
        self.signal_calculator = SignalsCalculator.SignalsCalculater(camera_parameters=self.camera_parameters,
                                                                     frame_size=(self.frame_width, self.frame_height))
        self.signal_calculator.set_filter_value("screen_xy", 0.022)

        self.use_mediapipe = True
        self.filter_landmarks = True
        self.landmark_kalman = [Kalman1D(R=0.0065 ** 2) for _ in range(468)] #TODO: improve values, maybe move to calculator (mediapipe landmark smoothing calculator)

        # Calibration
        self.calibration_samples = dict()

        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.VideoWriter: cv2.VideoWriter = cv2.VideoWriter("dummy.mp4", self.fourcc, 30, (self.frame_width, self.frame_height))
        self.calibrate_neutral: bool = False
        self.neutral_signals = []
        self.pose_signals = []

        self.calibrate_pose: bool = False

        # add hotkey
        # TODO: how to handle activate mouse / toggle mouse etc. by global hotkey
        # keyboard.add_hotkey("esc", lambda: self.stop())
        keyboard.add_hotkey("alt + 1", lambda: self.toggle_gesture_mouse())  # TODO: Linux alternative
        keyboard.add_hotkey("m", lambda: self.toggle_mouse_mode())
        keyboard.add_hotkey("c", lambda: self.mouse.centre_mouse())
        keyboard.on_press_key("r", lambda e: self.disable_gesture_mouse())
        keyboard.on_release_key("r", lambda e: self.enable_gesture_mouse())
        # add mouse_events
        self.raw_signal = SignalsCalculator.SignalsResult()
        self.transformed_signals = SignalsCalculator.SignalsResult()
        self.signals: Dict[str, Signal] = {}

        self.disable_gesture_mouse()

    def run(self):
        self.is_running = True
        while self.is_running:
            if self.use_mediapipe:
                self.setup_signals("config/mediapipe_default.json")
                self.__start_camera()
                self.__run_mediapipe()
                self.__stop_camera()
            else:
                self.setup_signals("config/iphone_default.json")
                self.__start_socket()
                self.__run_livelinkface()
                self.__stop_socket()

    def __run_mediapipe(self):
        while self.is_running and self.cam_cap.isOpened() and self.use_mediapipe:
            success, image = self.cam_cap.read()

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

            result = self.signal_calculator.process_ear(np_landmarks)

            # only check videowriter not none? #
            if self.calibrate_neutral and success:
                self.VideoWriter.write(image)
                self.neutral_signals.append(list(result.values()))
                continue

            if self.calibrate_pose and success:
                self.VideoWriter.write(image)
                self.pose_signals.append(list(result.values()))
                continue
            ########

            for signal_name in self.signals:
                value = result[signal_name]
                self.signals[signal_name].set_value(value)

                self.mouse.process_signal(self.signals)
                # Debug
            black = np.zeros((self.frame_height, self.frame_height, 3)).astype(np.uint8)
            self.annotated_landmarks = DrawingDebug.draw_landmarks_fast(np_landmarks, black)
            for i, indices in enumerate(self.signal_calculator.ear_indices):
                self.annotated_landmarks = DrawingDebug.draw_landmarks_fast(np_landmarks, self.annotated_landmarks, index=indices, color=colors[i%len(colors)])

            self.fps = self.fps_counter()

    def __run_livelinkface(self):
        while self.is_running and not self.use_mediapipe:
            try:
                data, addr = self.socket.recvfrom(1024)
                success, live_link_face = PyLiveLinkFace.decode(data)
            except socket.error:
                success = False

            if success:
                for signal_name in self.signals:
                    value = live_link_face.get_blendshape(FaceBlendShape[signal_name])
                    self.signals[signal_name].set_value(value)
                if self.mouse_enabled:
                    self.mouse.process_signal(self.signals)
            self.msleep(16)

    def __start_camera(self):
        self.cam_cap = cv2.VideoCapture(0)
        #self.cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        #self.cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        #self.cam_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P',
        #                                                             'G'))  # From https://forum.opencv.org/t/videoio-v4l2-dev-video0-select-timeout/8822/4 for linux

    def __stop_camera(self):
        if self.cam_cap is not None:
            self.cam_cap.release()

    def __start_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(0)
        self.socket.bind(("", self.UDP_PORT))

    def __stop_socket(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def stop(self):
        self.is_running = False

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

    def toggle_mouse_mode(self):
        self.mouse.toggle_mode()

    def setup_signals(self, json_path: str):
        """
        Reads a config file and setup ups the available signals.
        :param json_path: Path to json
        """
        parsed_signals = json.load(open(json_path, "r"))
        self.signals = dict()
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

    def calibrate_signal(self, calibration_sample, name):
        neutral_samples = np.array(calibration_sample[name]["neutral"])
        pose_samples = np.array(calibration_sample[name]["pose"])
        neutral_samples = neutral_samples[len(neutral_samples) // 4:3 * len(neutral_samples) // 4]
        pose_samples = pose_samples[len(pose_samples) // 4:3 * len(neutral_samples) // 4]
        signal = self.signals.get(name)
        min_value = max_value = 0
        if signal is not None:
            min_value = np.percentile(neutral_samples, 75)
            max_value = np.percentile(pose_samples, 25)
            signal.set_threshold(min_value, max_value)
        return min_value, max_value

    # Combine these methods?
    def calibrate_neutral_start(self, name):
        if not os.path.exists(f"calibration/{name}"):
            os.mkdir(f"calibration/{name}")
        self.calibrate_neutral = True
        self.VideoWriter.open(f"calibration/{name}/{name}_neutral.mp4", self.fourcc, 30, (self.frame_width,self.frame_height))

    def calibrate_neutral_stop(self, name):
        self.VideoWriter.release()
        self.calibrate_neutral = False
    def calibrate_pose_start(self, name):
        if not os.path.exists(f"calibration/{name}"):
            os.mkdir(f"calibration/{name}")
        self.calibrate_pose = True
        self.VideoWriter.open(f"calibration/{name}/{name}_pose.mp4", self.fourcc, 30, (self.frame_width,self.frame_height))
    def calibrate_pose_stop(self, name):
        self.VideoWriter.release()
        self.calibrate_pose = False
        self.calibration_samples[name] = {"neutral": self.neutral_signals, "pose": self.pose_signals}
    #####
    def recalibrate(self):
        print(f"=== Recalibrating === with f{len(self.calibration_samples)}")
        print(self.calibration_samples)

        if len(self.calibration_samples)==0:
            print("Nothing to calibrate")
            return
        data_array = []
        label_array = []
        for name in self.calibration_samples:
            print(name)
            for label, data in self.calibration_samples[name].items():
                data = data[len(data)//4:3*len(data)//4]
                data_array.extend(data)
                if label == "neutral":
                    label_array.extend([label]*len(data))
                else:
                    label_array.extend([name] * len(data))
        data_array = np.array(data_array)
        print(data_array, data_array.shape)
        print(label_array, len(label_array))


if __name__ == '__main__':
    demo = Demo()
    demo.run()
    bla = input("Press any key to stop")