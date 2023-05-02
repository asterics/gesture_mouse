#!/usr/bin/env python3

import json
import os.path
import time
import uuid
from typing import List, Dict
from collections import deque

from pynput import mouse
from pynput import keyboard
# import pygame
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui
import numpy as np

import Demo
import Signal
from gui_widgets import LogarithmicSlider
import re


class PlotLine:
    def __init__(self, pen, plot_data_item: pg.PlotDataItem):
        self.x = deque(maxlen=100)
        self.y = deque(maxlen=100)
        self.length = 0
        self.max_length = 100
        self.pen = pen
        self.plot_data_item = plot_data_item

    def plot(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.plot_data_item.setData(self.x, self.y, pen=self.pen)

    def set_visible(self, visibility):
        self.plot_data_item.setVisible(visibility)


class SignalVis(QtWidgets.QWidget):
    def __init__(self):
        super(SignalVis, self).__init__()
        self.plot_area: pg.PlotWidget = pg.PlotWidget()
        self.plot_item: pg.PlotItem = self.plot_area.getPlotItem()
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.plot_area)
        self.raw_selection_button = QtWidgets.QRadioButton("Raw Values")
        self.layout.addWidget(self.raw_selection_button)
        self.layout.setAlignment(self.raw_selection_button, QtCore.Qt.AlignmentFlag.AlignTop)
        self.raw_selection_button.toggled.connect(self.toggle_raw)
        self.raw_values = False
        self.plot_area.setBackground('w')
        self.lines = {}
        self.index = 0

    def add_line(self, name: str):
        pen = pg.mkPen(color=pg.intColor(self.index, 8, 2))
        data_line = self.plot_area.plot(x=[90, -90] * 50, y=[0] * 100, pen=pen)
        plot_handler = PlotLine(pen, data_line)
        self.lines[name] = plot_handler
        self.index = self.index + 1
        return plot_handler

    def update_plot(self, signals):
        x = time.time()
        for name, plot in self.lines.items():
            signal = signals.get(name)
            if signal is None:
                continue
            if self.raw_values:
                y = signals[name].raw_value.get()
                plot.plot(x, y)
            else:
                y = signals[name].scaled_value
                plot.plot(x, y)

    def toggle_raw(self, checked):
        print(checked)
        self.raw_values = checked


class SignalSetting(QtWidgets.QWidget):
    def __init__(self, name: str, min_value, max_value, min_filter=0.0001, max_filter=1., demo=None):
        super().__init__()
        self.name = name
        self.name_label = QtWidgets.QLabel(name)

        self.demo = demo

        self.lower_value = QtWidgets.QDoubleSpinBox()
        self.lower_value.setSingleStep(0.01)
        self.lower_value.setMinimum(-100.)
        self.lower_value.setMaximum(100.)
        self.lower_value.setValue(min_value)

        self.higher_value = QtWidgets.QDoubleSpinBox()
        self.higher_value.setSingleStep(0.01)
        self.higher_value.setMinimum(-100.)
        self.higher_value.setMaximum(100.)
        self.higher_value.setValue(max_value)

        self.filter_slider = LogarithmicSlider(orientation=QtCore.Qt.Orientation.Horizontal)
        self.filter_slider.setMinimum(min_filter)
        self.filter_slider.setMaximum(max_filter)

        self.visualization_checkbox = QtWidgets.QCheckBox("Visualize")
        self.visualization_checkbox.setChecked(True)

        self.calibrate_button = QtWidgets.QPushButton("Calibrate Thresholds")
        self.calibrate_button.clicked.connect(self.calibrate_signal)

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.visualization_checkbox)
        self.layout.addWidget(QtWidgets.QLabel("Signal range"))
        self.layout.addWidget(self.lower_value)
        self.layout.addWidget(self.higher_value)
        self.layout.addWidget(QtWidgets.QLabel("Filter"))
        self.layout.addWidget(self.filter_slider)
        self.layout.addWidget(self.calibrate_button)

        self.filter_slider.doubleValueChanged.connect(lambda value: print(value))

    def calibrate_signal(self):
        print("Calibration start")
        self.calib_diag = CalibrationDialog(self.demo, self.name)
        self.calib_diag.accepted.connect(self.accept_calibration)
        self.calib_diag.webcam_timer.start()
        self.calib_diag.open()
        # self.calibration_dialog.show()

    def accept_calibration(self):
        min_value = self.calib_diag.min_value
        max_value = self.calib_diag.max_value
        self.lower_value.setValue(min_value)
        self.higher_value.setValue(max_value)


class SignalTab(QtWidgets.QWidget):
    signal_added = QtCore.Signal(dict)

    def __init__(self, demo, json_path):
        super().__init__()
        self.demo: Demo.Demo = demo
        self.setWindowTitle("Signals Visualization")
        self.signals_vis = SignalVis()
        self.signals_vis.setMaximumHeight(250)
        self.signals_vis.setMinimumHeight(100)
        size_policy = self.signals_vis.sizePolicy()
        size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Maximum)
        size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
        self.signals_vis.setSizePolicy(size_policy)
        self.add_signal_button = QtWidgets.QPushButton("Record new Signal")
        self.add_signal_button.clicked.connect(self.add_new_signal)
        self.save_signals_button = QtWidgets.QPushButton("Save Profile")
        self.load_signals_button = QtWidgets.QPushButton("Load Profile")
        self.save_signals_button.clicked.connect(self.save_signals)
        self.load_signals_button.clicked.connect(self.load_action)

        self.layout = QtWidgets.QVBoxLayout(self)

        self.layout.addWidget(self.signals_vis)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)

        self.signal_settings = dict()
        self.setting_widget = QtWidgets.QWidget()
        self.signal_config = dict()
        self.load_signals(json_path)

        self.layout.addWidget(self.scroll_area)
        button_layout = QtWidgets.QHBoxLayout(self)
        button_layout.addWidget(self.add_signal_button)
        button_layout.addStretch()
        button_layout.addWidget(self.save_signals_button)
        button_layout.addWidget(self.load_signals_button)
        self.layout.addLayout(button_layout)


    def update_plots(self, signals):
        self.signals_vis.update_plot(signals)

    def add_new_signal(self):
        self.sig_diag = AddSignalDialog(self.demo)
        self.sig_diag.accepted.connect(self.accept_new_signal)
        self.sig_diag.webcam_timer.start()
        self.sig_diag.open()

    def accept_new_signal(self):
        signal_name = self.sig_diag.new_name.text()

        new_singal = {
            "name": signal_name,
            "lower_threshold": 0.,
            "higher_threshold": 1.,
            "filter_value": 0.0001
        }

        setting = SignalSetting(signal_name, 0., 1., demo=self.demo)
        handler = self.signals_vis.add_line(signal_name)

        setting.visualization_checkbox.stateChanged.connect(handler.set_visible)
        setting.visualization_checkbox.setChecked(False)

        setting.filter_slider.doubleValueChanged.connect(
            lambda x, name=signal_name: self.demo.set_filter_value(name, x))
        setting.lower_value.valueChanged.connect(
            lambda x, name=signal_name: self.demo.signals[name].set_lower_threshold(x))
        setting.higher_value.valueChanged.connect(
            lambda x, name=signal_name: self.demo.signals[name].set_higher_threshold(x))

        setting.filter_slider.setValue(0.0001)

        self.setting_widget.layout().addWidget(setting)
        self.signal_settings[signal_name] = setting

        self.signal_added.emit(new_singal)

    def save_signals(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select profile save file", "./config",
                                                             "JSON (*.json)")
        print(file_name) #TODO: no file selected
        self.demo.save_signals(file_name)

    def load_action(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select profile to load", "./config",
                                                             "JSON (*.json)")
        print(file_name) #TODO: no file selected
        self.load_signals(file_name)

    def load_signals(self, json_path):
        # Clear widget
        self.setting_widget = QtWidgets.QWidget()
        self.setting_widget.setLayout(QtWidgets.QVBoxLayout())
        self.signal_settings = dict()

        #Load json
        self.signal_config: dict = json.load(open(json_path, "r"))

        for json_signal in self.signal_config["signals"]:
            signal_name = json_signal["name"]
            lower_threshold = json_signal["lower_threshold"]
            higher_threshold = json_signal["higher_threshold"]
            filter_value = json_signal["filter_value"]

            setting = SignalSetting(signal_name, lower_threshold, higher_threshold, demo=self.demo)
            handler = self.signals_vis.add_line(signal_name)

            setting.visualization_checkbox.stateChanged.connect(handler.set_visible)
            setting.visualization_checkbox.setChecked(False)

            setting.filter_slider.doubleValueChanged.connect(
                lambda x, name=signal_name: self.demo.set_filter_value(name, x))
            setting.lower_value.valueChanged.connect(
                lambda x, name=signal_name: self.demo.signals[name].set_lower_threshold(x))
            setting.higher_value.valueChanged.connect(
                lambda x, name=signal_name: self.demo.signals[name].set_higher_threshold(x))

            setting.filter_slider.setValue(filter_value)

            self.setting_widget.layout().addWidget(setting)
            self.signal_settings[signal_name] = setting

        # load in demo
        self.demo.setup_signals(json_path)
        self.scroll_area.setWidget(self.setting_widget)

class CalibrationDialog(QtWidgets.QDialog):
    # TODO: add videorecording for data collection?
    def __init__(self, demo, name):
        super().__init__()
        self.demo: Demo.Demo = demo
        self.name = name
        self.label = QtWidgets.QLabel(name)
        self.calibration_samples = {name: {"neutral": [], "pose": []}}
        self.min_value = 0.
        self.max_value = 0.
        self.recording_neutral = False
        self.recording_max_pose = False

        self.do_action_label = QtWidgets.QLabel()
        self.neutral_timer = QtCore.QTimer(self)
        self.neutral_timer.setSingleShot(True)
        self.neutral_timer.setInterval(2000)

        self.pose_timer = QtCore.QTimer(self)
        self.pose_timer.setSingleShot(True)
        self.pose_timer.setInterval(2000)

        ## Webcam Image
        self.webcam_label = QtWidgets.QLabel()
        self.webcam_label.setMinimumSize(640, 480)
        self.webcam_label.setMaximumSize(1280, 720)
        self.webcam_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.webcam_timer = QtCore.QTimer(self)
        self.webcam_timer.setInterval(30)
        self.webcam_timer.timeout.connect(self.update_image)
        self.qt_image = QtGui.QImage(np.zeros((640, 480, 30), dtype=np.uint8), 480, 640,
                                     QtGui.QImage.Format.Format_BGR888)

        QBtn = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.start_calibration)
        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.do_action_label)
        self.layout.addWidget(self.webcam_label)
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.buttonBox)

        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

    def update_image(self):
        w = self.webcam_label.width()
        h = self.webcam_label.height()
        image = self.demo.annotated_landmarks
        signal = self.demo.signals.get(self.name)
        if signal is not None:
            if self.recording_neutral:
                self.calibration_samples[self.name]["neutral"].append(signal.raw_value.get())
            elif self.recording_max_pose:
                self.calibration_samples[self.name]["pose"].append(signal.raw_value.get())

        self.qt_image = QtGui.QImage(image, image.shape[1], image.shape[0], QtGui.QImage.Format.Format_BGR888)
        self.qt_image = self.qt_image.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
        self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(self.qt_image))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.webcam_label.resizeEvent(event)
        w = self.webcam_label.width()
        h = self.webcam_label.height()
        self.qt_image = self.qt_image.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(self.qt_image))

    def accept(self) -> None:
        self.webcam_timer.stop()
        print(self.calibration_samples)
        print(len(self.calibration_samples[self.name]["neutral"]))
        print(len(self.calibration_samples[self.name]["pose"]))
        self.min_value, self.max_value = self.demo.calibrate_signal(calibration_sample=self.calibration_samples,
                                                                    name=self.name)
        super().accept()

    def reject(self) -> None:
        self.webcam_timer.stop()
        super().reject()

    def start_calibration(self):
        self.do_action_label.setText("Neutral Pose")
        self.neutral_timer.timeout.connect(self.record_gesture)
        self.recording_neutral = True
        self.neutral_timer.start()

    def record_gesture(self):
        # TODO: save videos to create dataset?
        self.do_action_label.setText("Maximum Gesture")
        self.pose_timer.timeout.connect(self.finish_recording)
        self.recording_neutral = False
        self.recording_max_pose = True
        self.pose_timer.start()

    def finish_recording(self):
        self.recording_max_pose = False
        self.do_action_label.setText("Finished")


class AddSignalDialog(QtWidgets.QDialog):
    def __init__(self, demo):
        super().__init__()
        self.demo: Demo.Demo = demo

        self.name = "NewPosers"

        self.recording_neutral = False
        self.recording_max_pose = False

        self.do_action_label = QtWidgets.QLabel()
        self.neutral_timer = QtCore.QTimer(self)
        self.neutral_timer.setSingleShot(True)
        self.neutral_timer.setInterval(4000)

        self.pose_timer = QtCore.QTimer(self)
        self.pose_timer.setSingleShot(True)
        self.pose_timer.setInterval(4000)

        ## Webcam Image
        self.webcam_label = QtWidgets.QLabel()
        self.webcam_label.setMinimumSize(640, 480)
        self.webcam_label.setMaximumSize(1280, 720)
        self.webcam_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.webcam_timer = QtCore.QTimer(self)
        self.webcam_timer.setInterval(30)
        self.webcam_timer.timeout.connect(self.update_image)
        self.qt_image = QtGui.QImage(np.zeros((640, 480, 30), dtype=np.uint8), 480, 640,
                                     QtGui.QImage.Format.Format_BGR888)

        QBtn = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.start_calibration)
        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.name_label = QtWidgets.QLabel("Name")
        self.new_name = QtWidgets.QLineEdit()

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.do_action_label)
        self.layout.addWidget(self.webcam_label)
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.name_label)
        self.button_layout.addWidget(self.new_name)
        self.button_layout.addWidget(self.buttonBox)

        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

    def update_image(self):
        w = self.webcam_label.width()
        h = self.webcam_label.height()
        image = self.demo.annotated_landmarks
        self.qt_image = QtGui.QImage(image, image.shape[1], image.shape[0], QtGui.QImage.Format.Format_BGR888)
        self.qt_image = self.qt_image.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
        self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(self.qt_image))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.webcam_label.resizeEvent(event)
        w = self.webcam_label.width()
        h = self.webcam_label.height()
        self.qt_image = self.qt_image.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(self.qt_image))

    def accept(self) -> None:
        name = self.new_name.text()
        if name == "":
            msgBox = QtWidgets.QMessageBox()
            msgBox.setWindowTitle("Error")
            msgBox.setText("Error occured")
            msgBox.setInformativeText("Name is missing")
            msgBox.exec()
            return
        self.webcam_timer.stop()
        self.demo.recalibrate(name)
        super().accept()

    def reject(self) -> None:
        self.webcam_timer.stop()
        super().reject()

    def start_calibration(self):
        name = self.new_name.text()
        if name == "":
            msgBox = QtWidgets.QMessageBox()
            msgBox.setWindowTitle("Error")
            msgBox.setText("Error occured")
            msgBox.setInformativeText("Name is missing")
            msgBox.exec()
            return

        self.do_action_label.setText("Neutral Pose")
        self.neutral_timer.timeout.connect(self.record_gesture)
        self.recording_neutral = True
        self.demo.calibrate_neutral_start(name)
        self.neutral_timer.start()
        self.setStyleSheet("background-color:rgb(255,0,0)")


    def record_gesture(self):
        # TODO: save videos to create dataset?
        name = self.new_name.text()
        self.demo.calibrate_neutral_stop(name)
        self.demo.calibrate_pose_start(name)
        self.do_action_label.setText("Maximum Gesture")
        self.pose_timer.timeout.connect(self.finish_recording)
        self.recording_neutral = False
        self.recording_max_pose = True
        self.pose_timer.start()
        self.setStyleSheet("background-color:rgb(0,0,255)")

    def finish_recording(self):
        name = self.new_name.text()
        self.recording_max_pose = False
        self.demo.calibrate_pose_stop(name)
        self.do_action_label.setText("Finished")
        self.setStyleSheet("")


class DebugVisualizetion(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.webcam_label = QtWidgets.QLabel()
        self.webcam_label.setMinimumSize(1, 1)
        self.webcam_label.setMaximumSize(1280, 720)
        self.qt_image = QtGui.QImage()
        self.webcam_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.showMessage("FPS: ")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.webcam_label)
        self.layout.addWidget(self.status_bar)

    def update_image(self, image):
        w = self.webcam_label.width()
        h = self.webcam_label.height()
        self.qt_image = QtGui.QImage(image, image.shape[1], image.shape[0], QtGui.QImage.Format.Format_BGR888)
        self.qt_image = self.qt_image.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
        self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(self.qt_image))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.webcam_label.resizeEvent(event)
        self.status_bar.resizeEvent(event)
        w = self.webcam_label.width()
        h = self.webcam_label.height()
        self.qt_image = self.qt_image.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(self.qt_image))


class GeneralTab(QtWidgets.QWidget):
    def __init__(self, demo):
        super().__init__()
        self.demo: Demo.Demo = demo
        self.mediapipe_selector_button = QtWidgets.QCheckBox(text="Use web cam tracking.")
        self.mediapipe_selector_button.setChecked(self.demo.use_mediapipe)
        self.mediapipe_selector_button.clicked.connect(lambda selected: self.demo.set_use_mediapipe(selected))
        self.landmark_filter_button = QtWidgets.QCheckBox(text="Filter Landmarks.")
        self.landmark_filter_button.setChecked(self.demo.filter_landmarks)
        self.landmark_filter_button.clicked.connect(lambda selected: self.demo.set_filter_landmarks(selected))
        self.debug_window = DebugVisualizetion()
        self.debug_window_button = QtWidgets.QPushButton("Open Debug Menu")
        self.debug_window_button.clicked.connect(self.toggle_debug_window)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.mediapipe_selector_button)
        self.layout.addWidget(self.landmark_filter_button)
        self.layout.addWidget(self.debug_window_button)
        self.layout.addStretch()

    def toggle_debug_window(self):
        self.debug_window.show()

    def update_debug_visualization(self):
        self.debug_window.update_image(self.demo.annotated_landmarks)
        self.debug_window.status_bar.showMessage(f"FPS: {self.demo.fps}, Mode: {self.demo.mouse.mode}")


class MouseTab(QtWidgets.QWidget):
    def __init__(self, demo):
        super().__init__()
        self.demo: Demo.Demo = demo
        layout = QtWidgets.QVBoxLayout(self)

        self.left_click_uid = uuid.uuid4()
        self.left_click_settings = MouseClickSettings("Left Click", self.left_click_uid, demo)
        self.left_click_signal = "-"
        self.left_click_settings.signal_selector.currentTextChanged.connect(self.set_left_click)

        self.right_click_uid = uuid.uuid4()
        self.right_click_settings = MouseClickSettings("Right Click", self.right_click_uid, demo)
        self.right_click_signal = "-"
        self.right_click_settings.signal_selector.currentTextChanged.connect(self.set_right_click)

        self.double_click_uid = uuid.uuid4()
        self.double_click_settings = MouseClickSettings("Double Click", self.double_click_uid, demo)
        self.double_click_signal = "-"
        self.double_click_settings.signal_selector.currentTextChanged.connect(self.set_double_click)

        layout.addWidget(self.left_click_settings)
        layout.addWidget(self.right_click_settings)
        layout.addWidget(self.double_click_settings)
        layout.addStretch()

    def set_signal_selector(self, signals: List[str]):
        self.left_click_settings.signal_selector.clear()
        self.left_click_settings.signal_selector.addItems("-")
        self.left_click_settings.signal_selector.addItems(signals)
        self.right_click_settings.signal_selector.clear()
        self.right_click_settings.signal_selector.addItems("-")
        self.right_click_settings.signal_selector.addItems(signals)
        self.double_click_settings.signal_selector.clear()
        self.double_click_settings.signal_selector.addItems("-")
        self.double_click_settings.signal_selector.addItems(signals)

    def set_left_click(self, selected_text: str):
        if selected_text == "":
            return
        if self.left_click_signal != "-":
            self.demo.signals[self.left_click_signal].remove_action(self.left_click_uid)
        self.left_click_signal = selected_text
        if selected_text == "-":
            return
        action = Signal.Action()
        action.up_action = lambda: self.demo.mouse.click(mouse.Button.left)
        action.set_threshold(self.left_click_settings.threshold.value())
        action.set_delay(self.left_click_settings.delay.value())
        self.demo.signals[selected_text].add_action(self.left_click_uid, action)

    def set_right_click(self, selected_text: str):
        if selected_text == "":
            return
        if self.right_click_signal != "-":
            self.demo.signals[self.right_click_signal].remove_action(self.right_click_uid)
        self.right_click_signal = selected_text
        if selected_text == "-":
            return
        action = Signal.Action()
        action.up_action = lambda: self.demo.mouse.click(mouse.Button.right)
        action.set_threshold(self.right_click_settings.threshold.value())
        action.set_delay(self.right_click_settings.delay.value())
        self.demo.signals[selected_text].add_action(self.double_click_uid, action)

    def set_double_click(self, selected_text: str):
        if selected_text == "":
            return
        if self.double_click_signal != "-":
            self.demo.signals[self.double_click_signal].remove_action(self.double_click_uid)
        self.double_click_signal = selected_text
        if selected_text == "-":
            return
        action = Signal.Action()
        action.up_action = lambda: self.demo.mouse.double_click(mouse.Button.left)
        action.set_threshold(self.double_click_settings.threshold.value())
        action.set_delay(self.double_click_settings.delay.value())
        self.demo.signals[selected_text].add_action(self.double_click_uid, action)


class MouseClickSettings(QtWidgets.QWidget):
    def __init__(self, name, uid, demo):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        self.demo = demo
        self.uid = uid
        self.label = QtWidgets.QLabel(name)
        self.threshold = QtWidgets.QDoubleSpinBox(self)
        self.threshold.setMinimum(0.)
        self.threshold.setMaximum(1.)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.5)
        self.threshold.valueChanged.connect(self.threshold_changed)
        self.delay = QtWidgets.QDoubleSpinBox(self)
        self.delay.setMinimum(0.)
        self.delay.setMaximum(2.)
        self.delay.setSingleStep(0.01)
        self.delay.setValue(0.5)
        self.delay.valueChanged.connect(self.delay_changed)
        self.signal_selector = QtWidgets.QComboBox()

        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel("Threshold"))
        layout.addWidget(self.threshold)
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel("Delay"))
        layout.addWidget(self.delay)
        layout.addStretch(1)
        layout.addWidget(self.signal_selector)
        layout.addStretch(10)

    def threshold_changed(self, new_threshold):
        signal = self.demo.signals.get(self.signal_selector.currentText(), None)
        if signal is None:
            return  # no signal detected
        action = signal.actions.get(self.uid, None)
        if action is None:
            return  # action not defined
        action.set_threshold(new_threshold)

    def delay_changed(self, new_delay):
        signal = self.demo.signals.get(self.signal_selector.currentText(), None)
        if signal is None:
            return  # no signal detected
        action = signal.actions.get(self.uid, None)
        if action is None:
            return  # action not defined
        action.set_delay(new_delay)


class KeyboardActionWidget(QtWidgets.QWidget):
    remove_clicked = QtCore.Signal()
    action_updated = QtCore.Signal()

    def __init__(self, name: uuid.UUID):
        super().__init__()
        self.name: uuid.UUID = name
        self.current_signal: str = ""

        self.layout = QtWidgets.QHBoxLayout(self)
        self.threshold = QtWidgets.QDoubleSpinBox(self)
        self.threshold.setMinimum(0.)
        self.threshold.setMaximum(1.)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.5)
        self.threshold.valueChanged.connect(self._emit_updated)
        self.delay = QtWidgets.QDoubleSpinBox(self)
        self.delay.setMinimum(0.)
        self.delay.setMaximum(2.)
        self.delay.setSingleStep(0.01)
        self.delay.setValue(0.5)
        self.delay.valueChanged.connect(self._emit_updated)
        self.signal_selector = QtWidgets.QComboBox()
        self.signal_selector.currentTextChanged.connect(self._emit_updated)
        self.action_trigger_selector = QtWidgets.QComboBox()
        self.action_trigger_selector.addItems(["-", "up", "down", "hold high", "hold low"])
        self.action_trigger_selector.currentTextChanged.connect(self._emit_updated)
        self.action_type_selector = QtWidgets.QComboBox()
        self.action_type_selector.addItems(["-", "press", "release", "press and release"])
        self.action_type_selector.currentTextChanged.connect(self._emit_updated)
        self.key_input = QtWidgets.QKeySequenceEdit()
        self.key_input.setClearButtonEnabled(True)
        self.key_input.editingFinished.connect(self._emit_updated)
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_clicked.emit)
        self.layout.addWidget(self.signal_selector)
        self.layout.addWidget(QtWidgets.QLabel("Threshold"))
        self.layout.addWidget(self.threshold)
        self.layout.addWidget(QtWidgets.QLabel("Delay"))
        self.layout.addWidget(self.delay)
        self.layout.addWidget(QtWidgets.QLabel("Trigger"))
        self.layout.addWidget(self.action_trigger_selector)
        self.layout.addWidget(QtWidgets.QLabel("Type"))
        self.layout.addWidget(self.action_type_selector)
        self.layout.addWidget(self.key_input)
        self.layout.addWidget(self.remove_button)

    def set_signal_selector(self, signals: List[str]):
        self.signal_selector.clear()
        self.signal_selector.addItems("-")
        self.signal_selector.addItems(signals)
        self.signal_selector.adjustSize()

    def _emit_updated(self):
        self.action_updated.emit()


class KeyboardTab(QtWidgets.QWidget):
    def __init__(self, demo):
        super().__init__()
        self.demo: Demo.Demo = demo
        self.add_action_button = QtWidgets.QPushButton("Add Action")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.add_action_button, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.add_action_button.clicked.connect(self.add_action)

        ## Scrollbar
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setWidgetResizable(True)

        self.actions_widget = QtWidgets.QWidget()
        self.actions_widget.setLayout(QtWidgets.QVBoxLayout())
        self.actions_widget.layout().addStretch()
        self.scroll_area.setWidget(self.actions_widget)
        self.layout.addWidget(self.scroll_area)

        button_layout = QtWidgets.QHBoxLayout()
        self.save_actions_button = QtWidgets.QPushButton("Save profile")
        self.save_actions_button.clicked.connect(self.save_action)
        self.load_actions_button = QtWidgets.QPushButton("Load profile")
        self.load_actions_button.clicked.connect(self.load_profile)
        # self.layout.addStretch()

        button_layout.addStretch()
        button_layout.addWidget(self.load_actions_button)
        button_layout.addWidget(self.save_actions_button)

        self.layout.addLayout(button_layout)

        self.actions: Dict[uuid.UUID, KeyboardActionWidget] = {}
        self.signals: List[str] = []

        self.keyboard_controller: keyboard.Controller = keyboard.Controller()

    def add_action(self):
        name = uuid.uuid4()
        action_widget = KeyboardActionWidget(name=name)
        self.actions_widget.layout().insertWidget(self.actions_widget.layout().count() - 1, action_widget)
        # self.layout.insertWidget(self.layout.count() - 2, action_widget)
        self.actions[name] = action_widget
        action_widget.remove_clicked.connect(self.remove_action)
        action_widget.action_updated.connect(self.update_action)
        action_widget.set_signal_selector(self.signals)

    def set_signals(self, signals: List[str]):
        # Todo: remove actions, then load saved, then set combo-boxes
        self.signals = signals
        for action in self.actions.values():
            action.set_signal_selector(signals)

    def remove_action(self):
        action_widget = self.sender()
        print(action_widget)
        self.actions.pop(action_widget.name, None)
        self.actions_widget.layout().removeWidget(action_widget)
        # Get signal
        signal = self.demo.signals.get(action_widget.current_signal, None)
        if signal is not None:
            # delete old signal
            signal.remove_action(action_widget.name)

        action_widget.close()

    def update_action(self):
        action_widget: KeyboardActionWidget = self.sender()
        uid = action_widget.name
        new_signal = action_widget.signal_selector.currentText()
        trigger = action_widget.action_trigger_selector.currentText()
        action_type = action_widget.action_type_selector.currentText()
        key_sequence = action_widget.key_input.keySequence()
        key_sequence_string = key_sequence.toString().lower()
        threshold = action_widget.threshold.value()
        delay = action_widget.delay.value()

        print(f"{uid} / {new_signal} / {trigger} / {action_type} / {key_sequence_string} / {threshold}")
        # delete old signal
        signal = self.demo.signals.get(action_widget.current_signal, None)
        if signal is not None:
            # delete old signal
            signal.remove_action(action_widget.name)
        action_widget.current_signal = new_signal
        # Get new signal
        signal = self.demo.signals.get(new_signal, None)
        if signal is None:
            return  # No signal with this name, i.e no selected
        if key_sequence_string == "":
            return

        # TODO: move into keyboard class

        parsed_hotkeys = []
        for hotkey in re.split(r',\s', key_sequence_string):
            hotkey_string = re.sub(r'([a-z]{2,})', r'<\1>', hotkey)
            hotkey_string = hotkey_string.replace("del", "delete")
            hotkey_string = hotkey_string.replace("capslock", "caps_lock")
            # TODO: find missmatched strings
            parsed_hotkeys.append(keyboard.HotKey.parse(hotkey_string))
        print("Parsed hotkey ", parsed_hotkeys)

        # create new action
        new_action = Signal.Action()
        new_action.threshold = threshold
        action_function = None
        if action_type == "press":
            def action_function():
                for key_combo in parsed_hotkeys:
                    for key in key_combo:
                        self.keyboard_controller.press(key)

        elif action_type == "release":
            def action_function():
                for key_combo in reversed(parsed_hotkeys):
                    for key in reversed(key_combo):
                        self.keyboard_controller.release(key)

        elif action_type == "press and release":
            def action_function():
                for key_combo in parsed_hotkeys:
                    for key in key_combo:
                        self.keyboard_controller.press(key)
                for key_combo in reversed(parsed_hotkeys):
                    for key in reversed(key_combo):
                        self.keyboard_controller.release(key)
        else:
            return

        if trigger == "up":
            new_action.set_up_action(action_function)
        elif trigger == "down":
            new_action.set_down_action(action_function)
        elif trigger == "hold high":
            new_action.set_high_hold_action(action_function)
        elif trigger == "hold low":
            new_action.set_low_hold_action(action_function)
        else:
            return
        new_action.set_delay(delay)
        signal.add_action(uid, new_action)

    def save_action(self, filename):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select profile save file", "./config/profiles",
                                                             "JSON (*.json)")
        print(file_name)
        serial_actions = []
        for action in self.actions.values():
            threshold = action.threshold.value()
            trigger = action.action_trigger_selector.currentText()
            signal = action.signal_selector.currentText()
            action_type = action.action_type_selector.currentText()
            key = action.key_input.keySequence().toString()
            delay = action.delay.value()
            serial_action = {
                "action": f"keyboard_key",
                "signal": signal,
                "threshold": threshold,
                "delay": delay,
                "trigger": trigger,
                "action_type": action_type,
                "key": key
            }
            serial_actions.append(serial_action)
        with open(file_name, "w") as f:
            json.dump(serial_actions, f, indent=2)

    def load_profile(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select profile to load", "./config/profiles",
                                                             "JSON (*.json)")
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"No such file or directory: {file_name}")
        for action in self.actions.values():
            signal_name = action.current_signal
            signal = self.demo.signals.get(signal_name, None)
            if signal is not None:
                signal.remove_action(action.name)
            self.actions_widget.layout().removeWidget(action)
            action.close()
        self.actions.clear()
        with open(file_name, "r") as f:
            json_profile = json.load(f)
            for action in json_profile:
                action_mapping = action["action"]
                if action_mapping != "keyboard_key":
                    continue
                signal = action.get("signal", "")
                threshold = float(action.get("threshold", "0.5"))
                trigger = action.get("trigger", "")
                action_type = action.get("action_type", "")
                key = action.get("key", "")
                delay = float(action.get("delay", "0.5"))

                # add widget
                name = uuid.uuid4()
                action_widget = KeyboardActionWidget(name=name)
                action_widget.set_signal_selector(self.signals)
                action_widget.signal_selector.setCurrentText(signal)
                action_widget.threshold.setValue(threshold)
                action_widget.delay.setValue(delay)
                action_widget.action_trigger_selector.setCurrentText(trigger)
                action_widget.action_type_selector.setCurrentText(action_type)
                action_widget.key_input.setKeySequence(key)
                self.actions[action_widget.name] = action_widget
                self.actions_widget.layout().insertWidget(self.actions_widget.layout().count() - 1, action_widget)
                action_widget.remove_clicked.connect(self.remove_action)
                action_widget.action_updated.connect(self.update_action)
                action_widget.action_updated.emit()  # create associated action


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.demo = Demo.Demo()

        self.central_widget = QtWidgets.QTabWidget()

        self.signal_tab_iphone = SignalTab(self.demo, "config/iphone_default.json")
        self.signal_tab_mediapipe = SignalTab(self.demo, "config/mediapipe_default.json")

        self.signals_tab = QtWidgets.QStackedWidget()
        self.signals_tab.addWidget(self.signal_tab_iphone)
        self.signals_tab.addWidget(self.signal_tab_mediapipe)
        self.signal_tab_iphone.signal_added.connect(lambda: print("not implemented yet"))
        self.signal_tab_mediapipe.signal_added.connect(self.add_signal)

        self.general_tab = GeneralTab(self.demo)
        self.general_tab.mediapipe_selector_button.clicked.connect(lambda selected: self.change_signals_tab(selected))
        self.keyboard_tab = KeyboardTab(self.demo)
        self.mouse_tab = MouseTab(self.demo)

        self.central_widget.addTab(self.general_tab, "General")
        self.central_widget.addTab(self.keyboard_tab, "Keyboard")
        self.central_widget.addTab(self.mouse_tab, "Mouse")
        self.central_widget.addTab(self.signals_tab, "Signal")

        self.setCentralWidget(self.central_widget)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start()

        self.change_signals_tab(True)
        ## Signals
        self.demo.start()

    def update_plots(self):
        # TODO: move up again
        self.selected_signals.update_plots(self.demo.signals)
        self.general_tab.update_debug_visualization()

    def change_signals_tab(self, checked: bool):
        if checked:
            self.signals_tab.setCurrentIndex(1)
            self.selected_signals = self.signal_tab_mediapipe
        else:
            self.signals_tab.setCurrentIndex(0)
            self.selected_signals = self.signal_tab_iphone
        self.mouse_tab.set_signal_selector(list(self.selected_signals.signal_settings.keys()))
        self.keyboard_tab.set_signals(list(self.selected_signals.signal_settings.keys()))

    def add_signal(self, new_signal: dict):
        self.mouse_tab.set_signal_selector(list(self.selected_signals.signal_settings.keys()))
        self.keyboard_tab.set_signals(list(self.selected_signals.signal_settings.keys()))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.demo.stop()
        self.demo.quit()
        event.accept()


def test_gui():
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(1280, 720)
    window.show()
    app.exec()
    print("hallo")


if __name__ == '__main__':
    test_gui()
