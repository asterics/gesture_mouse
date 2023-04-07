import dataclasses
import time
from threading import Thread
import socket
import json
from typing import Dict

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


class Demo(QThread):
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.mouse_enabled = False
        self.mouse_absolute = True
        self.mouse: Mouse.Mouse = Mouse.Mouse()

        self.frame_width, self.frame_height = (1280, 720)
        self.annotated_landmarks = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.int8)
        self.fps_counter = FPSCounter.FPSCounter(20)
        self.fps = 0
        self.cam_cap = None

        self.UDP_PORT = 11111
        self.socket = None

        self.camera_parameters = (1000, 1000, 1280 / 2, 720 / 2)
        self.signal_calculator = SignalsCalculator.SignalsCalculater(camera_parameters=self.camera_parameters,
                                                                     frame_size=(self.frame_width, self.frame_height))
        self.signal_calculator.set_filter_value("screen_xy", 0.022)

        self.use_mediapipe = True
        self.filter_landmarks = True
        self.landmark_kalman = [Kalman1D(R=0.008 ** 2) for _ in range(468)]

        # Calibration
        self.calibration_samples = dict()

        # add hotkey
        # TODO: how to handle activate mouse / toggle mouse etc. by global hotkey
        # keyboard.add_hotkey("esc", lambda: self.stop())
        keyboard.add_hotkey("alt + 1", lambda: self.toggle_gesture_mouse())  # TODO: Linux alternative
        keyboard.add_hotkey("m", lambda: self.toggle_mouse_mode())
        # add mouse_events
        self.raw_signal = SignalsCalculator.SignalsResult()
        self.transformed_signals = SignalsCalculator.SignalsResult()
        self.signals: Dict[str, Signal] = {}

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
        with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
            while self.is_running and self.cam_cap.isOpened() and self.use_mediapipe:
                success, image = self.cam_cap.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)
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

                result = self.signal_calculator.process(np_landmarks)

                for signal_name in self.signals:
                    value = result[signal_name]
                    self.signals[signal_name].set_value(value)

                if self.mouse_enabled:
                    self.mouse.process_signal(self.signals)
                # Debug
                #self.annotated_landmarks = DrawingDebug.annotate_landmark_image(landmarks, image)
                # DrawingDebug.show_por(x_pixel, y_pixel, self.monitor.w_pixels, self.monitor.h_pixels)

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

    def __start_camera(self):
        self.cam_cap = cv2.VideoCapture(0)
        self.cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
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
        self.calibration_samples.update(calibration_sample)
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


if __name__ == '__main__':
    demo = Demo()
    demo.run()
    bla = input("Press any key to stop")