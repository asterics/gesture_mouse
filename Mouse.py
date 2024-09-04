import time
from enum import Enum
from pynput import mouse
import math
# import pygame
import screeninfo

import numpy as np

from util import clamp
from KalmanFilter1D import Kalman1D


class IteratorEnum(Enum):
    def next(self):
        cls = self.__class__
        members = list(cls)
        index = (members.index(self) + 1) % len(members)
        return members[index]

    def prev(self):
        cls = self.__class__
        members = list(cls)
        index = (members.index(self) - 1) % len(members)
        return members[index]


class MouseMode(IteratorEnum):
    ABSOLUTE = 0
    RELATIVE = 1
    JOYSTICK = 2
    HYBRID = 3
    # DPAD = 5


class TrackingMode(IteratorEnum):
    MEDIAPIPE = 0
    PNP = 1
    NOSE = 2


class Mouse:
    def __init__(self):
        # bookkeeping for relative mouse
        self.x = 0
        self.y = 0

        self.pitch = 0
        self.yaw = 0
        self.pitch0 = 0.5
        self.yaw0 = 0.5

        # Sensitivity and Acceleration
        self.x_sensitivity = 1.
        self.y_sensitivity = 1.

        self.x_acceleration = 1.25
        self.y_acceleration = 1.25

        # Multiscreen support
        # TODO: multiscreen?
        self.monitor_index = 0
        self.monitors_list = screeninfo.get_monitors()
        print(self.monitors_list)
        default_screen = self.monitors_list[self.monitor_index]
        self.h_pixels = default_screen.height
        self.w_pixels = default_screen.width
        self.monitor_x_offset = default_screen.x
        self.monitor_y_offset = default_screen.y

        # Mouse Mode
        self.mode: MouseMode = MouseMode.ABSOLUTE
        self.mouse_listener = None
        self.mouse_controller: mouse.Controller = mouse.Controller()

        # Hybrid Mode
        self.in_finezone=True
        self.fine_zone_center=np.array([0,0])
        self.fine_zone_pixel_radius = 0.

        # Tracking Mode
        self.tracking_mode = TrackingMode.MEDIAPIPE
        # To toggle mouse movement
        self.mouse_movement_enabled = True
        self.is_dragging = False
        self.precision_mode = False

        # Filtering
        self.filter_value = 0.01
        self.kalman_filter = Kalman1D(sz=100, R=self.filter_value ** 2)
        self.filter_mouse_position = True

        # Gestures
        # To toggle mouse gestures
        self.mouse_gesture_enabled = True

    def move(self, pitch: float, yaw: float):
        if self.mode == MouseMode.ABSOLUTE:
            # x_new = self.w_pixels * 2 * (yaw - 0.25) + self.monitor_x_offset
            # y_new = self.h_pixels * 2 * (pitch - 0.25) + self.monitor_y_offset
            # self.move_mouse(x_new - self.x, y_new - self.y)
            self.move_absolute(pitch, yaw)
        elif self.mode == MouseMode.RELATIVE:
            self.move_relative(pitch, yaw)
        elif self.mode == MouseMode.JOYSTICK:
            self.joystick_mouse(pitch, yaw)
        elif self.mode == MouseMode.HYBRID:
            self.hybrid_mouse_joystick(pitch, yaw)

    def move_absolute(self, pitch, yaw):
        pitch = (pitch - self.pitch0)
        yaw = (yaw - self.yaw0)

        if self.tracking_mode == TrackingMode.NOSE:
            pitch *= 6
            yaw *= 6

        w = self.monitors_list[self.monitor_index].width
        h = self.monitors_list[self.monitor_index].height
        x = self.monitor_x_offset + w / 2 + w * self.x_sensitivity * yaw
        y = self.monitor_y_offset + h / 2 + h * self.y_sensitivity * pitch
        self.move_mouse(x - self.x, y - self.y)

    def move_relative(self, pitch, yaw):

        # Todo: use time to make it framerate independent
        dy = (pitch - self.pitch)
        dx = (yaw - self.yaw)
        if self.tracking_mode==TrackingMode.NOSE:
            dy*=6
            dx*=6


        self.dx = dx
        self.dy = dy

        # Maybe scale by monitor size
        mouse_speed_x, mouse_speed_y = self.calculate_mouse_speed(dx, dy)

        self.move_mouse(self.w_pixels * mouse_speed_x,
                        self.h_pixels * mouse_speed_y)  # TODO filtering makes incremental updates less impactful?

    def joystick_mouse(self, pitch, yaw):
        pitch = (pitch - self.pitch0)
        yaw = (yaw - self.yaw0)

        if self.tracking_mode==TrackingMode.NOSE:
            pitch*=6
            yaw*=6

        dead_zone = 0

        mouse_speed_x, mouse_speed_y = self.calculate_mouse_speed(yaw, pitch,
                                                                  (dead_zone, dead_zone, dead_zone, dead_zone))

        mouse_speed_x = mouse_speed_x * self.w_pixels / 12
        mouse_speed_y = mouse_speed_y * self.h_pixels / 12

        self.move_mouse(mouse_speed_x, mouse_speed_y)

    def hybrid_mouse_joystick(self, pitch: float, yaw: float) -> None:
        # TODO: Absolut/relativ in der Mitte, Dead, Zone abziehen
        fine_zone = 0.12

        pitch = (pitch - self.pitch0)
        yaw = (yaw - self.yaw0)
        if self.tracking_mode==TrackingMode.NOSE:
            pitch*=6
            yaw*=6

        yaw_pitch = np.array([yaw, pitch])
        yaw_pitch_squared_norm = np.dot(yaw_pitch,yaw_pitch)
        yaw_pitch_norm = np.sqrt(yaw_pitch_squared_norm)
        w = self.monitors_list[self.monitor_index].width
        h = self.monitors_list[self.monitor_index].height

        if yaw_pitch_norm<fine_zone:
            if not self.in_finezone:
                self.in_finezone = True
                self.x,self.y = x,y = self.mouse_controller.position
                incoming_position = np.array([x,y])
                #calculate center of fine zone
                self.fine_zone_center=incoming_position-np.array([5*self.x_sensitivity,5*self.y_sensitivity])*yaw_pitch*np.array([w,w])*fine_zone

            new_pos = self.fine_zone_center + np.array([5*self.x_sensitivity,5*self.y_sensitivity])*yaw_pitch*np.array([w,w])*fine_zone
            x, y = new_pos
            self.move_mouse(x - self.x, y - self.y)
            return

        self.in_finezone=False

        yaw_pitch_norm = yaw_pitch_norm
        new_yaw_pitch = yaw_pitch*((yaw_pitch_norm-fine_zone)/yaw_pitch_norm)

        mouse_speed_x, mouse_speed_y = self.calculate_mouse_speed(new_yaw_pitch[0], new_yaw_pitch[1])

        mouse_speed_x = mouse_speed_x * self.w_pixels / 6
        mouse_speed_y = mouse_speed_y * self.h_pixels / 6

        self.move_mouse(mouse_speed_x, mouse_speed_y)

    def update(self, x, y):
        self.x = x
        self.y = y
        return True

    def process_signal(self, signals):
        # TODO: move this around, possibilities: MosueAction / select signals in demo / select signals in mouse
        upDown = "upDown"
        leftRight = "leftRight"

        pitch = signals[upDown].scaled_value
        yaw = signals[leftRight].scaled_value
        if self.mouse_movement_enabled:
            self.move(pitch, yaw)
        else:
            self.dx = 0
            self.dy = 0
        self.pitch = pitch
        self.yaw = yaw

    def enable_gestures(self):
        self.mouse_gesture_enabled = True

    def disable_gestures(self):
        self.mouse_gesture_enabled = False

    def toggle_gestures(self):
        self.mouse_gesture_enabled = not self.mouse_gesture_enabled

    def click(self, button):
        """
        Clicks the mouse button specified with button.
        :param button: str, one of the buttons to click
        :return:
        """
        if self.mouse_gesture_enabled:
            self.mouse_controller.click(button)

    def double_click(self, button):
        """
        Double-clicks the mouse button specified with button.
        :param button: str, one of the buttons to click
        :return:
        """
        if self.mouse_gesture_enabled:
            self.mouse_controller.click(button, 2)

    def drag_drop(self):
        if self.mouse_gesture_enabled:
            if self.is_dragging:
                self.mouse_controller.release(mouse.Button.left)
                self.is_dragging = False
            else:
                self.mouse_controller.press(mouse.Button.left)
                self.is_dragging = True
            print(self.is_dragging)

    def toggle_mouse_movement(self):
        if self.mouse_movement_enabled:
            self.disable_mouse_movement()
        else:
            self.enable_mouse_movement()
    def enable_mouse_movement(self):
        print("enable mouse movement")
        self.mouse_movement_enabled = True

    def disable_mouse_movement(self):
        print("disable mouse movement")
        self.mouse_movement_enabled = False

    def toggle_mode(self):
        self.mode = self.mode.next()
        print(f"Mouse mode is {self.tracking_mode.name}")

    def set_mouse_mode(self, mode: str):
        try:
            self.mode = MouseMode[mode]
        except KeyError:
            print(f"Mouse mode {mode} is not a valid mode")

    def centre_mouse(self):
        # TODO: also center pitch / yaw zero point, or different position
        self.kalman_filter = Kalman1D(R=self.filter_value ** 2)
        x_new = self.monitor_x_offset + self.w_pixels // 2
        y_new = self.monitor_y_offset + self.h_pixels // 2
        self.x = x_new
        self.y = y_new
        self.pitch0 = self.pitch
        self.yaw0 = self.yaw
        self.mouse_controller.position = (x_new, y_new)
        self.in_finezone = False

    def switch_monitor(self):
        self.monitor_index = (self.monitor_index + 1) % len(self.monitors_list)
        screen = self.monitors_list[self.monitor_index]
        self.h_pixels = screen.height
        self.w_pixels = screen.width
        self.monitor_x_offset = screen.x
        self.monitor_y_offset = screen.y


    def set_x_sensitivity(self, value: float):
        self.x_sensitivity = value

    def set_y_sensitivity(self, value: float):
        self.y_sensitivity = value

    def set_x_acceleration(self, value: float):
        self.x_acceleration = value

    def set_y_acceleration(self, value: float):
        self.y_acceleration = value

    def toggle_precision_mode(self):
        self.precision_mode = not self.precision_mode
        print(f"Precision mode enabled: {self.precision_mode}")
        if self.precision_mode:
            self.x_sensitivity = self.x_sensitivity / 5.
            self.y_sensitivity = self.y_sensitivity / 5.
        else:
            self.x_sensitivity = self.x_sensitivity * 5.
            self.y_sensitivity = self.y_sensitivity * 5.

    def calculate_mouse_speed(self, x_value, y_value, dead_zone=None):
        # default value
        if dead_zone is None:
            dead_zone = (0, 0, 0, 0)

        mouse_speed_x = mouse_speed_y = 0
        if x_value < dead_zone[0]:
            text = "Looking Left"
            mouse_speed_x = -self.x_sensitivity * math.pow(abs(x_value), self.x_acceleration)
        if x_value > dead_zone[1]:
            text = "Looking Right"
            mouse_speed_x = self.x_sensitivity * math.pow(abs(x_value), self.x_acceleration)
        if y_value < dead_zone[2]:
            text = "Looking Down"
            mouse_speed_y = -self.y_sensitivity * math.pow(abs(y_value), self.y_acceleration)
        if y_value > dead_zone[3]:
            text = "Looking Up"
            mouse_speed_y = self.y_sensitivity * math.pow(abs(y_value), self.y_acceleration)

        mouse_speed_x = 0.01 / math.pow(0.01, self.x_acceleration) * mouse_speed_x
        mouse_speed_y = 0.01 / math.pow(0.01, self.y_acceleration) * mouse_speed_y

        return mouse_speed_x, mouse_speed_y

    def move_mouse(self, x_speed, y_speed):
        x_speed_filtered = y_speed_filtered = 0
        # Somehow floating point errors have a bias which accumulate, this is a hacky solution for it
        if abs(x_speed) <= 1:
            x_speed = 0
        if abs(y_speed) <= 1:
            y_speed = 0
        x, y = self.mouse_controller.position
        self.x = self.x + x_speed
        self.y = self.y + y_speed

        if self.filter_mouse_position:
            output_tracked = self.kalman_filter.update(self.x + 1j * self.y)
            x_new_filtered, y_new_filtered = np.real(output_tracked), np.imag(output_tracked)
            x_speed_filtered = x_new_filtered - x
            y_speed_filtered = y_new_filtered - y
        else:
            x_speed_filtered = x_speed
            y_speed_filtered = y_speed

        self.mouse_controller.move(x_speed_filtered, y_speed_filtered)

    def set_filter_value(self, value):
        self.filter_value = value
        self.kalman_filter = Kalman1D(R=self.filter_value ** 2)

    def set_filter_enabled(self, enabled):
        self.filter_mouse_position = enabled

    def set_tracking_mode(self, tracking_mode: str):
        try:
            self.tracking_mode = TrackingMode[tracking_mode]
        except KeyError:
            print(f"Tracking mode {tracking_mode} is not a valid mode")

    def toggle_tracking_mode(self):
        self.tracking_mode = self.tracking_mode.next()
        print(f"Tracking mode is {self.tracking_mode.name}")

if __name__ == '__main__':
    mouse = Mouse()
    mouse.centre_mouse()
    mouse.set_filter_enabled(False)
    mouse.set_mouse_mode(MouseMode.ABSOLUTE.name)
    mouse.set_tracking_mode(TrackingMode.NOSE.name)
    mouse.enable_gestures()

    for i in range(1000):
        time.sleep(0.01)
        mouse.move_mouse(0, 0)

    print("finished")
