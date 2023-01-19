import dataclasses
import time
from threading import Thread
import socket

import keyboard
import mediapipe as mp
import cv2
import numpy as np

import Mouse
import DrawingDebug
import SignalsCalculator
import monitor


from pyLiveLinkFace import PyLiveLinkFace, FaceBlendShape

mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_connections = mp.solutions.face_mesh_connections


class Demo(Thread):
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.mouse_enabled = False
        self.mouse_absolute = True
        self.mouse = Mouse.Mouse()

        self.frame_width, self.frame_height = (1280, 720)
        self.cam_cap = None

        self.UDP_PORT = 11111
        self.socket = None

        self.monitor = monitor.monitor()

        self.camera_parameters = (800, 800, 1280/2, 720/2)
        self.signal_calculator = SignalsCalculator.SignalsCalculater(camera_parameters=self.camera_parameters)
        self.signal_calculator.set_filter_value("screen_xy", 0.022)

        self.use_mediapipe = False

        # add hotkey
        keyboard.add_hotkey("esc", lambda: self.stop())
        keyboard.add_hotkey("alt + 1", lambda: self.toggle_gesture_mouse())
        keyboard.add_hotkey("m", lambda: self.toggle_mouse_mode())
        # add mouse_events
        self.signal_settings = {
            "pitch": {
                "min": -90,
                "max": +90,
                "filter_value": 0.001
            },
            "yaw": {
                "min": -90,
                "max": +90,
                "filter_value": 0.001
            },
            "roll": {
                "min": -90,
                "max": +90,
                "filter_value": 0.001
            },
            "mouth_open": {
                "min": -90,
                "max": +90,
                "filter_value": 0.001
            }
        }
        self.raw_signal = SignalsCalculator.SignalsResult()
        self.transformed_signals = SignalsCalculator.SignalsResult()

    def run(self):
        self.is_running = True
        while self.is_running:
            if self.use_mediapipe:
                self.__start_camera()
                self.__run_mediapipe()
                self.__stop_camera()
            else:
                self.__start_socket()
                self.__run_livelinkface()
                self.__stop_socket()

    def __run_mediapipe(self):
        with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
            while self.is_running and self.cam_cap.isOpened() and self.use_mediapipe:
                success, image = self.cam_cap.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if not results.multi_face_landmarks:
                    continue
                landmarks = results.multi_face_landmarks[0]
                np_landmarks = np.array(
                    [(lm.x * self.frame_width, lm.y * self.frame_height, lm.z * self.frame_width) for lm in
                     landmarks.landmark])
                result = self.signal_calculator.process(np_landmarks)

                ## Calculate point on screen

                x_pixel, y_pixel = result.screen_xy.get()

                self.raw_signal = result

                if self.mouse_enabled:
                    self.mouse.move(x_pixel, y_pixel, self.mouse_absolute)
                # Debug
                DrawingDebug.show_landmarks(landmarks, image)
                # DrawingDebug.show_por(x_pixel, y_pixel, self.monitor.w_pixels, self.monitor.h_pixels)

    def __run_livelinkface(self):
        while self.is_running and not self.use_mediapipe:
            try:
                data, addr = self.socket.recvfrom(1024)
                success, live_link_face = PyLiveLinkFace.decode(data)
            except socket.error:
                success = False

            if success:
                self.raw_signal.pitch.set(live_link_face.get_blendshape(FaceBlendShape.HeadPitch))
                self.raw_signal.yaw.set(live_link_face.get_blendshape(FaceBlendShape.HeadYaw))
                self.raw_signal.roll.set(live_link_face.get_blendshape(FaceBlendShape.HeadYaw))
                self.raw_signal.mouth_puck.set(live_link_face.get_blendshape(FaceBlendShape.MouthPucker))
                self.raw_signal.jaw_open.set(live_link_face.get_blendshape(FaceBlendShape.JawOpen))
                self.raw_signal.debug1.set(live_link_face.get_blendshape(FaceBlendShape.BrowInnerUp))
                self.raw_signal.debug2.set(live_link_face.get_blendshape(FaceBlendShape.EyeBlinkRight))
                self.raw_signal.debug3.set(live_link_face.get_blendshape(FaceBlendShape.CheekPuff))

    def __start_camera(self):
        self.cam_cap = cv2.VideoCapture(0)
        self.cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

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

    def record_neutral(self):
        with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
            success, image = self.cam_cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if not results.multi_face_landmarks:
                return False

            landmarks = results.multi_face_landmarks[0]
            np_landmarks = np.array(
                [(lm.x * self.frame_width, lm.y * self.frame_height) for lm in
                 landmarks.landmark])
            self.signal_calculator.process_neutral(np_landmarks)

            return True

    def stop(self):
        self.is_running = False
        self.cam_cap.release()

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
        print(name)
        self.signal_calculator.set_filter_value(name, filter_value)
        signal = getattr(self.raw_signal, name, None)
        if signal is not None:
            signal.set_filter_value(filter_value)

    def set_use_mediapipe(self, selected):
        self.use_mediapipe = selected


    def toggle_mouse_mode(self):
        self.mouse_absolute = not self.mouse_absolute
