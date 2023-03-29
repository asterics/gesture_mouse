from enum import Enum
from pynput import mouse
import math
# import pygame
import screeninfo


class MouseMode(Enum):
    ABSOLUTE = 1
    RELATIVE = 2
    JOYSTICK = 3

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


class Mouse:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.pitch = 0
        self.yaw = 0
        monitors = screeninfo.get_monitors()
        print(monitors)
        default_screen = monitors[0]  # TODO: multiscreen?
        self.mode: MouseMode = MouseMode.ABSOLUTE
        self.h_pixels = default_screen.height
        self.w_pixels = default_screen.width
        self.mouse_listener = None
        self.mouse_controller: mouse.Controller = mouse.Controller()

    def move(self, pitch: int, yaw: int):
        if self.mode == MouseMode.ABSOLUTE:
            self.x = self.w_pixels * yaw
            self.y = self.h_pixels * pitch
            self.mouse_controller.position = (self.x, self.y)
        elif self.mode == MouseMode.RELATIVE:
            self.move_relative(pitch, yaw)
        elif self.mode == MouseMode.JOYSTICK:
            self.joystick_mouse(pitch, yaw)

    def move_relative(self, pitch, yaw):
        # Todo: use time to make it framerate independent
        dy = (pitch - self.pitch)
        dx = (yaw - self.yaw)

        mouse_speed_co = 1.1  # Todo: Param for gui
        mouse_speed_max = 25
        acceleration = 3

        # TODO: Threshold / Deadzone
        mouse_speed_x = mouse_speed_y = 0
        if dx < -0.001:
            mouse_speed_x = -min(math.pow(mouse_speed_co, abs(dx * acceleration)), mouse_speed_max) + 1
        elif dx > 0.001:
            mouse_speed_x = min(math.pow(mouse_speed_co, abs(dx * acceleration)), mouse_speed_max) - 1
        if dy < -0.001:
            mouse_speed_y = -min(math.pow(mouse_speed_co, abs(dy * acceleration)), mouse_speed_max) + 1
        elif dy > 0.001:
            mouse_speed_y = min(math.pow(mouse_speed_co, abs(dy * acceleration)), mouse_speed_max) - 1

        mouse_speed_x *= 2 * self.w_pixels
        mouse_speed_y *= self.h_pixels

        self.x += mouse_speed_x
        self.y += mouse_speed_y

        self.pitch = pitch
        self.yaw = yaw

        self.mouse_controller.move(mouse_speed_x, mouse_speed_y)

    def joystick_mouse(self, pitch, yaw):
        pitch = 50*(pitch-0.5)
        yaw = 50*(yaw-0.5)
        mouse_speed_co = 1.1
        mouse_speed_max = 25
        acceleration = 3

        threshold = (-2, 2, -0, 2)

        mouse_speed_x = 0
        mouse_speed_y = 0

        # See where the user's head tilting
        if yaw < threshold[0]:
            text = "Looking Left"
            mouse_speed_x = -1 * min(math.pow(mouse_speed_co, abs(yaw * acceleration)), mouse_speed_max)+1
        if yaw > threshold[1]:
            text = "Looking Right"
            mouse_speed_x = min(math.pow(mouse_speed_co, abs(yaw * acceleration)), mouse_speed_max)-1
        if pitch < threshold[2]:
            text = "Looking Down"
            mouse_speed_y = min(math.pow(mouse_speed_co, abs(pitch * acceleration)), mouse_speed_max)+1
        if pitch > threshold[3]:
            text = "Looking Up"
            mouse_speed_y = -1 * min(math.pow(mouse_speed_co, abs(pitch * acceleration)), mouse_speed_max)-1

        # print(text)
        self.mouse_controller.move(mouse_speed_x, mouse_speed_y)

    def update(self, x, y):
        self.x = x
        self.y = y
        return True

    def process_signal(self, signals):
        # TODO: move this around, possibilities: MosueAction / select signals in demo / select signals in mouse
        updown = "HeadPitch"
        leftright = "HeadYaw"
        pitch = (1 - signals[updown].scaled_value)
        yaw = (1 - signals[leftright].scaled_value)
        self.move(pitch, yaw)

    def enable_gesture(self):
        pass

    def click(self, button):
        """
        Clicks the mouse button specified with button.
        :param button: str, one of the buttons to click
        :return:
        """
        self.mouse_controller.click(button)

    def double_click(self, button):
        """
        Double-clicks the mouse button specified with button.
        :param button: str, one of the buttons to click
        :return:
        """
        self.mouse_controller.click(button, 2)

    def disable_gesture(self):
        pass

    def toggle_mode(self):
        self.mode = self.mode.next()
