from enum import Enum
from pynput import mouse
import math
# import pygame
import screeninfo
from util import clamp


class MouseMode(Enum):
    ABSOLUTE = 1
    RELATIVE = 2
    JOYSTICK = 3
    HYBRID = 4

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
        # bookkeeping for relative mouse
        self.x = 0
        self.y = 0

        self.pitch = 0
        self.yaw = 0

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

        # To disable mouse
        self.mouse_enabled = True
        self.is_dragging = False
        self.precision_mode = False

    def move(self, pitch: float, yaw: float):
        if self.mode == MouseMode.ABSOLUTE:
            self.x =self.w_pixels * 2*(yaw-0.25) + self.monitor_x_offset
            self.y =self.h_pixels * 2*(pitch-0.25) + self.monitor_y_offset
            self.mouse_controller.position = (self.x, self.y)
        elif self.mode == MouseMode.RELATIVE:
            self.move_relative(pitch, yaw)
        elif self.mode == MouseMode.JOYSTICK:
            self.joystick_mouse(pitch, yaw)
        elif self.mode == MouseMode.HYBRID:
            self.hybrid_mouse_joystick(pitch,yaw)

    def move_relative(self, pitch, yaw):

        # Todo: use time to make it framerate independent
        dy = (pitch - self.pitch)
        dx = (yaw - self.yaw)

        self.dx = dx
        self.dy = dy
        print(dx,dy)

        mouse_speed_x = mouse_speed_y = 0
        if dx < -0.00:
            mouse_speed_x = -self.x_sensitivity*math.pow(abs(dx), self.x_acceleration)
        elif dx > 0.00:
            mouse_speed_x = self.x_sensitivity*math.pow(abs(dx), self.x_acceleration)
        if dy < -0.00:
            mouse_speed_y = -self.y_sensitivity*math.pow(abs(dy), self.y_acceleration)
        elif dy > 0.00:
            mouse_speed_y = self.y_sensitivity*math.pow(abs(dy), self.y_acceleration)

        # Maybe scale by monitor size
        mouse_speed_x = 0.01/math.pow(0.01,self.x_acceleration) * mouse_speed_x
        mouse_speed_y = 0.01/math.pow(0.01,self.y_acceleration) * mouse_speed_y

        self.x += self.w_pixels*mouse_speed_x
        self.y += self.h_pixels*mouse_speed_y

        self.mouse_controller.move(self.w_pixels*mouse_speed_x, self.h_pixels*mouse_speed_y)

    def joystick_mouse(self, pitch, yaw):
        pitch = (pitch - 0.5)
        yaw = (yaw - 0.5)

        threshold = (0., 0., 0., 0.)

        mouse_speed_x = 0
        mouse_speed_y = 0

        # See where the user's head tilting
        if yaw < 0.00:
            mouse_speed_x = -self.x_sensitivity*math.pow(abs(yaw), self.x_acceleration)
        elif yaw > 0.00:
            mouse_speed_x = self.x_sensitivity*math.pow(abs(yaw), self.x_acceleration)
        if pitch < 0.00:
            mouse_speed_y = -self.y_sensitivity*math.pow(abs(pitch), self.y_acceleration)
        elif pitch > 0.00:
            mouse_speed_y = self.y_sensitivity*math.pow(abs(pitch), self.y_acceleration)

        mouse_speed_x = 0.01/math.pow(0.01,self.x_acceleration) * mouse_speed_x
        mouse_speed_y = 0.01/math.pow(0.01,self.y_acceleration) * mouse_speed_y

        mouse_speed_x = mouse_speed_x*self.w_pixels/8
        mouse_speed_y = mouse_speed_y*self.h_pixels/8


        self.mouse_controller.move(mouse_speed_x, mouse_speed_y)

    def hybrid_mouse_joystick(self, pitch:float, yaw:float) -> None:
        dead_zone = 2.5
        fine_zone = 9.
        mouse_speed_co = 1.1
        mouse_speed_max = 25
        acceleration = 2

        pitch = -50 * (pitch - 0.5)
        yaw = 50 * (yaw - 0.5)

        if abs(yaw) < dead_zone and abs(pitch) < dead_zone:
            return
        if abs(yaw) < fine_zone and abs(pitch) < fine_zone:
            self.x = self.x + 0.25*yaw
            self.y = self.y - 0.25*pitch
            self.mouse_controller.position = (self.x,self.y)
            return

        print(abs(yaw),abs(pitch))
        mouse_speed_x = mouse_speed_y = 0
        if yaw < 0:
            text = "Looking Left"
            mouse_speed_x = -1 * min(math.pow(mouse_speed_co, abs(yaw * acceleration)), mouse_speed_max) + 1
        if yaw > 0:
            text = "Looking Right"
            mouse_speed_x = min(math.pow(mouse_speed_co, abs(yaw * acceleration)), mouse_speed_max) - 1
        if pitch < 0:
            text = "Looking Down"
            mouse_speed_y = min(math.pow(mouse_speed_co, abs(pitch * acceleration)), mouse_speed_max) + 1
        if pitch > 0:
            text = "Looking Up"
            mouse_speed_y = -1 * min(math.pow(mouse_speed_co, abs(pitch * acceleration)), mouse_speed_max) - 1
        self.x, self.y = self.mouse_controller.position
        self.x = self.x + mouse_speed_x
        self.y = self.y + mouse_speed_y

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
        if self.mouse_enabled:
            self.move(pitch, yaw)
        else:
            self.dx = 0
            self.dy = 0
        self.pitch = pitch
        self.yaw = yaw


    def enable_gesture(self):
        self.mouse_enabled = True

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

    def drag_drop(self):
        if self.is_dragging:
            self.mouse_controller.release(mouse.Button.left)
            self.is_dragging = False
        else:
            self.mouse_controller.press(mouse.Button.left)
            self.is_dragging = True
        print(self.is_dragging)

    def disable_gesture(self):
        self.mouse_enabled = False

    def toggle_active(self):
        self.mouse_enabled = not self.mouse_enabled

    def toggle_mode(self):
        self.mode = self.mode.next()

    def centre_mouse(self):
        self.x = self.monitor_x_offset + self.w_pixels//2
        self.y = self.monitor_y_offset + self.h_pixels//2
        self.mouse_controller.position = (self.x, self.y)

    def switch_monitor(self):
        self.monitor_index = (self.monitor_index+1)%len(self.monitors_list)
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
        if self.precision_mode:
            self.x_sensitivity = self.x_sensitivity/5.
            self.y_sensitivity = self.y_sensitivity/5.
        else:
            self.x_sensitivity = self.x_sensitivity*5.
            self.y_sensitivity = self.y_sensitivity*5.

