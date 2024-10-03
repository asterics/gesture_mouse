#!/usr/bin/env python3

# update splash screen
import platform

if platform.system() == 'Windows':
    try:
        import pyi_splash
        # Update the text on the splash screen
        pyi_splash.update_text("Importing modules...")
    except Exception as inst:
        print(f"Splash screen not supported on this platform: {inst}")

import json
import os.path
import time
import uuid
from typing import List, Dict
from collections import deque
from functools import partial

from pynput import mouse
from pynput import keyboard
# import pygame
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QFileDialog
import numpy as np

import Demo
import Mouse
import Gesture
import util
from gui_widgets import LogarithmicSlider, ColoredDoubleSlider, DoubleSlider, StyledMouseSlider
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
        pen = pg.mkPen(color=pg.intColor(self.index, 8, 2), width=2)
        data_line = self.plot_area.plot(x=[90, -90] * 50, y=[0] * 100, pen=pen)
        plot_handler = PlotLine(pen, data_line)
        self.lines[name] = plot_handler
        self.index = self.index + 1
        return plot_handler

    def remove_line(self, name):
        print(name)
        handler = self.lines.pop(name,None)
        if handler is not None:
            handler.plot_data_item.setData()

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


class SignalSetting(QtWidgets.QFrame):
    deleted = QtCore.Signal(str)
    save_triggered = QtCore.Signal()
    def __init__(self, name: str, min_value, max_value, min_filter=0.0001, max_filter=1., demo=None):
        super().__init__()
        self.name = name
        print(name)
        self.name_label = QtWidgets.QLabel(name)

        self.demo:Demo.Demo = demo

        self.lower_value = QtWidgets.QDoubleSpinBox()
        self.lower_value.setSingleStep(0.01)
        self.lower_value.setMinimum(-100.)
        self.lower_value.setMaximum(100.)
        self.lower_value.setValue(min_value)
        self.lower_value.valueChanged.connect(self.set_lower_threshold)
        self.lower_value.valueChanged.connect(lambda : self.save_triggered.emit())

        self.higher_value = QtWidgets.QDoubleSpinBox()
        self.higher_value.setSingleStep(0.01)
        self.higher_value.setMinimum(-100.)
        self.higher_value.setMaximum(100.)
        self.higher_value.setValue(max_value)
        self.higher_value.valueChanged.connect(self.set_higher_threshold)
        self.higher_value.valueChanged.connect(lambda: self.save_triggered.emit())

        self.filter_slider = LogarithmicSlider(orientation=QtCore.Qt.Orientation.Horizontal)
        self.filter_slider.setMinimum(min_filter)
        self.filter_slider.setMaximum(max_filter)
        self.filter_slider.doubleValueChanged.connect(self.set_filter_value)
        filter_value_indicator = QtWidgets.QLabel("0")
        self.filter_slider.doubleValueChanged.connect(lambda value:filter_value_indicator.setText(f"{value:.4f}"))
        self.filter_slider.doubleValueChanged.connect(lambda: self.save_triggered.emit())

        self.visualization_checkbox = QtWidgets.QCheckBox("Visualize")
        self.visualization_checkbox.setChecked(True)

        self.calibrate_button = QtWidgets.QPushButton("Calibrate Thresholds")
        self.calibrate_button.clicked.connect(self.calibrate_signal)

        self.delete_button= QtWidgets.QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_signal)

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.name_label, stretch=1)
        self.layout.addWidget(self.visualization_checkbox)
        self.layout.addWidget(QtWidgets.QLabel("GestureSignal range"))
        self.layout.addWidget(self.lower_value)
        self.layout.addWidget(self.higher_value)
        self.layout.addWidget(QtWidgets.QLabel("Filter"))
        self.layout.addWidget(self.filter_slider, stretch=1)
        self.layout.addWidget(filter_value_indicator)
        self.layout.addWidget(self.calibrate_button)
        self.layout.addWidget(self.delete_button)
        self.layout.addStretch(2)


        self.filter_slider.doubleValueChanged.connect(lambda value: print(value))

        self.setFrameShape(QtWidgets.QFrame.Shape.Box)

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
        self.save_triggered.emit()

    def delete_signal(self):
        print(f"Delete in signal settings with name {self.name}")
        self.demo.delete_signal(self.name)
        self.demo.recalibrate()
        self.deleteLater()
        self.deleted.emit(self.name)
        self.save_triggered.emit()

    def debug_check(self):
        print(self.name)

    def set_lower_threshold(self, value):
        self.demo.signals[self.name].set_lower_threshold(value)

    def set_higher_threshold(self, value):
        self.demo.signals[self.name].set_higher_threshold(value)

    def set_filter_value(self, value):
        signal = self.demo.signals.get(self.name,None)
        if signal is not None:
            self.demo.signals[self.name].set_filter_value(value)


class SignalTab(QtWidgets.QWidget):
    signals_updated = QtCore.Signal()

    def __init__(self, demo, json_path, tracker_name):
        super().__init__()
        self.demo: Demo.Demo = demo
        self.tracker_name = tracker_name
        self.setWindowTitle("Signals Visualization")
        self.signals_vis = SignalVis()
        self.signals_vis.setMaximumHeight(250)
        self.signals_vis.setMinimumHeight(100)
        size_policy = self.signals_vis.sizePolicy()
        size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Maximum)
        size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
        self.signals_vis.setSizePolicy(size_policy)
        self.add_signal_button = QtWidgets.QPushButton("Record new GestureSignal")
        self.add_signal_button.clicked.connect(self.add_new_signal)
        self.save_signals_button = QtWidgets.QPushButton("Save Profile")
        self.load_signals_button = QtWidgets.QPushButton("Load Profile")
        self.save_signals_button.clicked.connect(self.save_signals)
        self.load_signals_button.clicked.connect(self.load_signals_dialog)

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

    def delete_signal(self, name):
        print(f"delete signal with name: {name} in SignalTab")
        self.signals_vis.remove_line(name)
        self.signal_settings.pop(name,None)
        self.signals_updated.emit()

    def accept_new_signal(self):
        signal_name = self.sig_diag.new_name.text()

        new_singal = {
            "name": signal_name,
            "lower_threshold": 0.,
            "higher_threshold": 1.,
            "filter_value": 0.0001
        }

        success = self.demo.recalibrate()

        if not success:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setWindowTitle("Error")
            msgBox.setText("Error occured")
            msgBox.setInformativeText("Not able to add new signal")
            msgBox.exec()
            return

        self.demo.add_signal(signal_name)

        self.signal_settings[signal_name]  = SignalSetting(signal_name, 0., 1., demo=self.demo)
        handler = self.signals_vis.add_line(signal_name)

        self.signal_settings[signal_name].visualization_checkbox.stateChanged.connect(handler.set_visible)
        self.signal_settings[signal_name].visualization_checkbox.setChecked(False)

        self.signal_settings[signal_name].filter_slider.setValue(0.0001)
        self.signal_settings[signal_name].deleted.connect(self.delete_signal)
        #self.signal_settings[signal_name].save_triggered.connect(
        #    lambda: self.save_signals(f"config/{self.tracker_name}_signal_latest.json"))

        self.setting_widget.layout().addWidget(self.signal_settings[signal_name])

        self.signals_updated.emit()

    def save_signal_dialog(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select profile save file", "./config",
                                                             "JSON (*.json)")
        self.save_signals(file_name)
    def save_signals(self, file_name):

        if file_name=="":
            return # no file selected

        self.demo.save_signals(file_name)

    def load_signals_dialog(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select profile to load", "./config",
                                                             "JSON (*.json)")
        if file_name == "":
            return  # no file selected

        self.load_signals(file_name)

    def load_signals(self, json_path):
        # Clear widget
        self.setting_widget = QtWidgets.QWidget()
        self.setting_widget.setLayout(QtWidgets.QVBoxLayout())
        self.signal_settings = dict()

        #Load json
        try:
            self.signal_config: dict = json.load(open(json_path, "r"))
        except FileNotFoundError:
            print("File not found")
            return

        for json_signal in self.signal_config["signals"]:
            signal_name = json_signal["name"]
            lower_threshold = json_signal["lower_threshold"]
            higher_threshold = json_signal["higher_threshold"]
            filter_value = json_signal["filter_value"]

            self.signal_settings[signal_name] = SignalSetting(signal_name, lower_threshold, higher_threshold, demo=self.demo)
            handler = self.signals_vis.add_line(signal_name)

            self.signal_settings[signal_name].visualization_checkbox.stateChanged.connect(handler.set_visible)
            self.signal_settings[signal_name].visualization_checkbox.setChecked(False)

            self.signal_settings[signal_name].filter_slider.setValue(filter_value)
            self.signal_settings[signal_name].deleted.connect(self.delete_signal)
            #self.signal_settings[signal_name].save_triggered.connect(lambda : self.save_signals(f"config/{self.tracker_name}_signal_latest.json"))
            self.setting_widget.layout().addWidget(self.signal_settings[signal_name])

            #self.signal_added.emit()

        self.signals_updated.emit()
        #self.save_signals(f"config/{self.tracker_name}_signal_latest.json")
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
        self.neutral_timer.setInterval(5000)

        self.pose_timer = QtCore.QTimer(self)
        self.pose_timer.setSingleShot(True)
        self.pose_timer.setInterval(5000)

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
        super().accept()

    def reject(self) -> None:
        self.webcam_timer.stop()
        self.neutral_timer.blockSignals(True)
        self.pose_timer.blockSignals(True)
        self.neutral_timer.stop()
        self.pose_timer.stop()
        self.neutral_timer.blockSignals(False)
        self.pose_timer.blockSignals(False)
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
        self.setStyleSheet("background-color:rgb(18,102,80)")


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
        self.setStyleSheet("background-color:rgb(96,70,8)")

    def finish_recording(self):
        name = self.new_name.text()
        self.recording_max_pose = False
        self.demo.calibrate_pose_stop(name)
        self.do_action_label.setText("Finished")
        self.setStyleSheet("")


class DebugVisualizetion(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowTitle("Gesture Mouse - Live Debug")
        self.setMaximumSize(Demo.VID_RES_X, Demo.VID_RES_Y)

        self.webcam_label = QtWidgets.QLabel()
        self.webcam_label.setMinimumSize(1, 1)
        self.qt_image = QtGui.QImage()
        self.webcam_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.showMessage("FPS: ")
        self.status_bar_gestures = QtWidgets.QStatusBar()
        self.status_bar_gestures.showMessage("")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.status_bar)
        self.layout.addWidget(self.webcam_label)
        self.layout.addWidget(self.status_bar_gestures)


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
    mode_changed = QtCore.Signal(str) # Change to enum
    def __init__(self, demo):
        super().__init__()
        self.demo: Demo.Demo = demo

        #add group box for video source and other settings
        self.video_source_grp=QGroupBox("Camera / Video source")
        self.filter_grp=QGroupBox("Filter settings")

        self.mediapipe_selector_button = QtWidgets.QCheckBox(text="Use web cam tracking.")
        self.mediapipe_selector_button.setChecked(self.demo.use_mediapipe)
        self.mediapipe_selector_button.clicked.connect(lambda selected: self.demo.set_use_mediapipe(selected))


        self.landmark_filter_button = QtWidgets.QCheckBox(text="Filter Landmarks.")
        self.landmark_filter_button.setChecked(self.demo.filter_landmarks)
        self.landmark_filter_button.clicked.connect(lambda selected: self.demo.set_filter_landmarks(selected))

        self.debug_window = DebugVisualizetion()
        self.debug_window_button = QtWidgets.QPushButton("Open Camera/Video Display")
        self.debug_window_button.clicked.connect(self.toggle_debug_window)

        self.vid_source_start = QtWidgets.QPushButton("Start tracking")
        self.vid_source_start.clicked.connect(self.demo.start_tracking)
        self.vid_source_stop = QtWidgets.QPushButton("Stop tracking")
        self.vid_source_stop.clicked.connect(self.demo.stop_tracking)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.video_source_grp)
        self.layout.addWidget(self.filter_grp)

        self.vid_main_layout=QtWidgets.QVBoxLayout()
        self.vid_mode_layout=QtWidgets.QHBoxLayout()
        self.vid_mode_grp=QtWidgets.QGroupBox("Video mode selection")

        self.vid_mode_grp.setLayout(self.vid_mode_layout)

        self.vid_webcam_grp=QGroupBox("Use webcam")
        self.vid_webcam_grp.setCheckable(True)
        self.vid_webcam_grp.setChecked(self.demo.use_mediapipe)
        self.vid_webcam_grp.toggled.connect(self.webcam_grp_toggled)
        self.vid_webcam_layout=QtWidgets.QFormLayout()
        self.vid_webcam_grp.setLayout(self.vid_webcam_layout)

        self.vid_iphone3d_grp=QGroupBox("Use iPhone 3D camera")
        self.vid_iphone3d_grp.setCheckable(True)
        self.vid_iphone3d_grp.setChecked(not self.demo.use_mediapipe)
        self.vid_iphone3d_grp.toggled.connect(self.iphone_grp_toggled)
        self.vid_iphone3d_layout=QtWidgets.QFormLayout()
        self.vid_iphone3d_layout.addRow(QtWidgets.QLabel("My IP address: "),QtWidgets.QLabel(self.demo.my_ip))
        self.vid_iphone3d_layout.addRow(QtWidgets.QLabel("My UPD port: "), QtWidgets.QLabel(str(self.demo.UDP_PORT)))
        self.vid_iphone3d_grp.setLayout(self.vid_iphone3d_layout)

        self.vid_vidfile_grp=QGroupBox("Use video file")
        self.vid_vidfile_grp.setCheckable(True)
        self.vid_vidfile_grp.setChecked(False)
        self.vid_vidfile_grp.toggled.connect(self.vidfile_grp_toggled)
        self.vid_vidfile_layout=QtWidgets.QFormLayout()
        self.vid_vidfile_openfile = QtWidgets.QPushButton("Select video file")
        self.vid_vidfile_openfile.clicked.connect(self.open_file_dialog)
        self.vid_vidfile_layout.addWidget(self.vid_vidfile_openfile)
        self.vid_vidfile_grp.setLayout(self.vid_vidfile_layout)

        self.csv_record_group = QGroupBox("Record all signals")
        self.csv_record_group.setCheckable(True)
        self.csv_record_group.setChecked(False)
        self.csv_record_group.toggled.connect(self.csv_grp_toggled)
        label = QtWidgets.QLabel("Helper Mode to evaluate system")
        self.csv_grp_layout = QtWidgets.QFormLayout()
        self.csv_grp_layout.addWidget(label)
        self.csv_record_group.setLayout(self.csv_grp_layout)



        self.vid_mode_layout.addWidget(self.vid_webcam_grp)
        self.vid_mode_layout.addWidget(self.vid_iphone3d_grp)
        self.vid_mode_layout.addWidget(self.vid_vidfile_grp)
        self.vid_mode_layout.addWidget(self.csv_record_group)

        self.vid_webcam_device=QtWidgets.QComboBox()
        webcam_available_ports,self.vid_webcam_devices,webcam_non_working_ports=util.list_camera_ports()
        self.vid_webcam_devices=map(str,self.vid_webcam_devices)
        print(self.vid_webcam_devices)
        self.vid_webcam_device.addItems(self.vid_webcam_devices)
        self.vid_webcam_device.currentTextChanged.connect(lambda arg__1: self.demo.update_webcam_device_selection(arg__1))
        self.vid_webcam_layout.addRow(QtWidgets.QLabel("Camera device"),self.vid_webcam_device)

        self.csv_write_group = QGroupBox("CSV Settings")
        self.csv_start_button = QtWidgets.QPushButton("Start")
        self.csv_stop_button = QtWidgets.QPushButton("Stop")
        self.csv_file_selection_button = QtWidgets.QPushButton("Select file location")
        self.csv_file_label = QtWidgets.QLabel("File Location")
        self.csv_file_path = ""
        # Events
        self.csv_file_selection_button.clicked.connect(self.csv_save_dialog)
        self.csv_start_button.clicked.connect(self.start_csv_recording)
        self.csv_stop_button.clicked.connect(self.demo.stop_write_csv)

        self.vid_mode_layout.addWidget(self.vid_webcam_grp)
        self.vid_mode_layout.addWidget(self.vid_iphone3d_grp)
        self.vid_mode_layout.addWidget(self.vid_vidfile_grp)

        self.vid_main_layout.addWidget(self.vid_mode_grp)
        self.vid_main_layout.addWidget(self.vid_source_start)
        self.vid_main_layout.addWidget(self.vid_source_stop)
        self.vid_main_layout.addWidget(self.debug_window_button)
        self.video_source_grp.setLayout(self.vid_main_layout)

        self.filter_grp_layout = QtWidgets.QVBoxLayout()
        self.filter_grp_layout.addWidget(self.landmark_filter_button)
        self.filter_grp.setLayout(self.filter_grp_layout)

        self.csv_writer_layout = QtWidgets.QHBoxLayout()
        self.csv_writer_layout.addWidget(self.csv_file_selection_button)
        self.csv_writer_layout.addWidget(self.csv_file_label)
        self.csv_writer_layout.addStretch()
        self.csv_writer_layout.addWidget(self.csv_start_button)
        self.csv_writer_layout.addWidget(self.csv_stop_button)
        self.csv_write_group.setLayout(self.csv_writer_layout)
        self.layout.addWidget(self.csv_write_group)

        self.layout.addStretch()

    def open_file_dialog(self):
        fileName = QFileDialog.getOpenFileName(self, "Open File")
        print(f"selected file {fileName[0]}")
        if fileName[0]:
            self.demo.update_webcam_video_file_selection(fileName[0])

    def csv_save_dialog(self):
        file_name = QFileDialog.getSaveFileName(self, "Select File", filter="*.csv")[0]
        if file_name:
            self.csv_file_path = file_name
            self.csv_file_label.setText(self.csv_file_path)

    def start_csv_recording(self):
        self.demo.start_write_csv(self.csv_file_path)

    def toggle_debug_window_globally(self):
        """
        Need a specific method for the global hook, because directly calling self.debug_window.show() freezes the GUI.
        This is probably because the GlobalHook is executed in a another thread and causes a dead lock.
        """
        print("called by global hotkey")
        self.debug_window_button.click()

    def toggle_debug_window(self):
        if self.debug_window.isVisible():
            print("hiding window...")
            self.debug_window.hide()
            #self.topLevelWidget().show()
        else:
            print("showing window...")
            self.debug_window.show()
            #self.topLevelWidget().hide()

    def update_debug_visualization(self):
        self.debug_window.update_image(self.demo.annotated_landmarks)
        self.debug_window.status_bar.showMessage(f"FPS: {int(self.demo.fps)}, M: {self.demo.mouse.mode.name}, T: {self.demo.mouse.tracking_mode.name}")

        # Check all gestures if they are activated and update status bar
        # TODO: Use listener pattern or queue to notify about changes?
        active_gestures = ""
        for signal in self.demo.signals.values():
            for action in signal.actions.values():
                if action.is_activated:
                    if active_gestures != "":
                        active_gestures += ", "
                    active_gestures += signal.name

        self.debug_window.status_bar_gestures.showMessage(active_gestures)

    def webcam_grp_toggled(self, on:bool):
        if on:
            self.vid_iphone3d_grp.blockSignals(True)
            self.vid_iphone3d_grp.setChecked(False)
            self.vid_iphone3d_grp.blockSignals(False)

            self.vid_vidfile_grp.blockSignals(True)
            self.vid_vidfile_grp.setChecked(False)
            self.vid_vidfile_grp.blockSignals(False)
            self.demo.use_mediapipe = True
            self.mode_changed.emit("WEBCAM")
        else:
            self.vid_webcam_grp.blockSignals(True)
            self.vid_webcam_grp.setChecked(False)
            self.vid_webcam_grp.blockSignals(False)

    def iphone_grp_toggled(self, on:bool):
        if on:
            self.vid_webcam_grp.blockSignals(True)
            self.vid_webcam_grp.setChecked(False)
            self.vid_webcam_grp.blockSignals(False)

            self.vid_vidfile_grp.blockSignals(True)
            self.vid_vidfile_grp.setChecked(False)
            self.vid_vidfile_grp.blockSignals(False)

            self.csv_record_group.blockSignals(True)
            self.csv_record_group.setChecked(False)
            self.csv_record_group.blockSignals(False)

            self.demo.use_mediapipe = False
            self.mode_changed.emit("IPHONE")
        else:
            self.vid_iphone3d_grp.blockSignals(True)
            self.vid_iphone3d_grp.setChecked(False)
            self.vid_iphone3d_grp.blockSignals(False)

    def vidfile_grp_toggled(self, on:bool):
        if on:
            self.vid_webcam_grp.blockSignals(True)
            self.vid_webcam_grp.setChecked(False)
            self.vid_webcam_grp.blockSignals(False)

            self.vid_iphone3d_grp.blockSignals(True)
            self.vid_iphone3d_grp.setChecked(False)
            self.vid_iphone3d_grp.blockSignals(False)

            self.csv_record_group.blockSignals(True)
            self.csv_record_group.setChecked(False)
            self.csv_record_group.blockSignals(False)

            self.demo.use_mediapipe = True
            self.mode_changed.emit("VIDEOFILE")
        else:
            self.vid_vidfile_grp.blockSignals(True)
            self.vid_vidfile_grp.setChecked(False)
            self.vid_vidfile_grp.blockSignals(False)

    def csv_grp_toggled(self, on:bool):
        if on:
            self.vid_webcam_grp.blockSignals(True)
            self.vid_webcam_grp.setChecked(False)
            self.vid_webcam_grp.blockSignals(False)

            self.vid_iphone3d_grp.blockSignals(True)
            self.vid_iphone3d_grp.setChecked(False)
            self.vid_iphone3d_grp.blockSignals(False)

            self.vid_vidfile_grp.blockSignals(True)
            self.vid_vidfile_grp.setChecked(False)
            self.vid_vidfile_grp.blockSignals(False)

            self.demo.recording_mode=True
            self.demo.use_mediapipe=False
            self.mode_changed.emit("VIDEOFILE")
        else:
            self.csv_record_group.blockSignals(True)
            self.csv_record_group.setChecked(False)
            self.csv_record_group.blockSignals(False)


class MouseTab(QtWidgets.QWidget):
    def __init__(self, demo):
        super().__init__()
        self.demo: Demo.Demo = demo
        self.click_settings = ["Left", "Right", "Double", "Drag and Drop", "Pause", "Center"]


        outer_layout = QtWidgets.QVBoxLayout(self)
        debug_layout = QtWidgets.QVBoxLayout()
        debug_frame = QtWidgets.QFrame()
        debug_frame.setFrameShape(QtWidgets.QFrame.Box)
        debug_frame.setLayout(debug_layout)
        upper_outer_layout = QtWidgets.QHBoxLayout()
        actions_layout = QtWidgets.QVBoxLayout()
        action_frame = QtWidgets.QFrame()
        action_frame.setFrameShape(QtWidgets.QFrame.Box)
        action_frame.setLayout(actions_layout)
        settings_layout = QtWidgets.QFormLayout()
        settings_frame = QtWidgets.QFrame()
        settings_frame.setFrameShape(QtWidgets.QFrame.Box)
        settings_frame.setLayout(settings_layout)
        outer_layout.addLayout(upper_outer_layout)
        #outer_layout.addStretch()
        outer_layout.addWidget(debug_frame)

        upper_outer_layout.addWidget(action_frame, stretch=1)
        upper_outer_layout.addWidget(settings_frame, stretch=1)

        actions_layout.addWidget(QtWidgets.QLabel("Click Settings"))
        settings_layout.addRow("Mouse Settings", None)

        debug_layout.addWidget(QtWidgets.QLabel("Information Screen"))

        # Click settings
        self.mouse_settings = {}
        self.mouse_settings["left_click"] = MouseClickSettings("Left",self.demo,lambda : self.demo.mouse.click(mouse.Button.left))
        self.mouse_settings["right_click"] = MouseClickSettings("Right",self.demo,lambda : self.demo.mouse.click(mouse.Button.right))
        self.mouse_settings["double_click"] = MouseClickSettings("Double Click", self.demo,lambda : self.demo.mouse.double_click(mouse.Button.left))
        self.mouse_settings["drag_drop"] = MouseClickSettings("Drag and Drop",self.demo,lambda : self.demo.mouse.drag_drop())
        self.mouse_settings["pause"] = MouseClickSettings("Pause", self.demo, lambda : self.demo.mouse.toggle_mouse_movement())
        self.mouse_settings["center"] = MouseClickSettings("Center", self.demo, lambda : self.demo.mouse.centre_mouse())
        self.mouse_settings["switch_mode"] = MouseClickSettings("Switch Mode", self.demo, lambda : self.demo.mouse.toggle_mode())
        self.mouse_settings["switch_monitor"] = MouseClickSettings("Switch Monitor", self.demo, lambda : self.demo.mouse.switch_monitor())
        self.mouse_settings["precision_mode"] = MouseClickSettings("Toggle Precision Mode", self.demo, lambda : self.demo.mouse.toggle_precision_mode())

        for mouse_setting in self.mouse_settings.values():
            actions_layout.addWidget(mouse_setting)
            mouse_setting.trigger_save.connect(self.save_lates)


        # Mouse Settings
        self.x_sensitivity_slider = StyledMouseSlider(decimals=3)
        self.x_sensitivity_slider.setValue(self.demo.mouse.x_sensitivity)
        self.x_sensitivity_slider.doubleValueChanged.connect(self.demo.mouse.set_x_sensitivity)
        self.x_sensitivity_slider.doubleValueChanged.connect(self.save_lates)

        self.y_sensitivity_slider = StyledMouseSlider(decimals=3)
        self.y_sensitivity_slider.setValue(self.demo.mouse.y_sensitivity)
        self.y_sensitivity_slider.doubleValueChanged.connect(self.demo.mouse.set_y_sensitivity)
        self.y_sensitivity_slider.doubleValueChanged.connect(self.save_lates)

        self.x_acceleration_slider = StyledMouseSlider(decimals=3)
        self.x_acceleration_slider.setValue(self.demo.mouse.x_acceleration)
        self.x_acceleration_slider.doubleValueChanged.connect(self.demo.mouse.set_x_acceleration)
        self.x_acceleration_slider.doubleValueChanged.connect(self.save_lates)

        self.y_acceleration_slider = StyledMouseSlider(decimals=3)
        self.y_acceleration_slider.setValue(self.demo.mouse.y_acceleration)
        self.y_acceleration_slider.doubleValueChanged.connect(self.demo.mouse.set_y_acceleration)
        self.y_acceleration_slider.doubleValueChanged.connect(self.save_lates)

        self.smoothing_toggle = QtWidgets.QCheckBox()
        self.smoothing_toggle.setChecked(self.demo.mouse.filter_mouse_position)
        self.smoothing_toggle.toggled.connect(self.demo.mouse.set_filter_enabled)
        self.smoothing_toggle.toggled.connect(self.save_lates)

        self.smoothing_value = LogarithmicSlider()
        self.smoothing_value.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.smoothing_value.setMinimum(0.001)
        self.smoothing_value.setMaximum(0.1)
        self.smoothing_value.setValue(self.demo.mouse.filter_value)
        self.smoothing_value.doubleValueChanged.connect(self.demo.mouse.set_filter_value)
        self.smoothing_value.doubleValueChanged.connect(self.save_lates)

        self.tracking_mode_selector = QtWidgets.QComboBox()
        self.tracking_mode_selector.addItems([mode.name for mode in Mouse.TrackingMode])
        self.tracking_mode_selector.currentTextChanged.connect(self.demo.mouse.set_tracking_mode)
        self.tracking_mode_selector.currentTextChanged.connect(self.save_lates)

        self.mouse_mode_selector = QtWidgets.QComboBox()
        self.mouse_mode_selector.addItems([mode.name for mode in Mouse.MouseMode])
        self.mouse_mode_selector.currentTextChanged.connect(self.demo.mouse.set_mouse_mode)
        self.mouse_mode_selector.currentTextChanged.connect(self.save_lates)

        self.save_button = QtWidgets.QPushButton("Save Profile")
        self.save_button.clicked.connect(self.save_profile_dialog)
        self.load_button = QtWidgets.QPushButton("Load Profile")
        self.load_button.clicked.connect(self.load_profile_dialog)


        settings_layout.addRow("x-Sensitivity",self.x_sensitivity_slider)
        settings_layout.addRow("y-Sensitivity",self.y_sensitivity_slider)
        settings_layout.addRow("x-Acceleration", self.x_acceleration_slider)
        settings_layout.addRow("y-Acceleration", self.y_acceleration_slider)
        settings_layout.addRow("Filter Mouse Position", self.smoothing_toggle)
        settings_layout.addRow("Filter Value", self.smoothing_value)
        settings_layout.addRow("Tracking Mode", self.tracking_mode_selector)
        settings_layout.addRow("Mouse Mode", self.mouse_mode_selector)

        debug_layout.addWidget(QtWidgets.QLabel("Hier stehen Infos"))
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)
        debug_layout.addLayout(button_layout)
        debug_layout.addStretch()



    def set_signal_selector(self, signals: List[str]):
        for mouse_setting in self.mouse_settings.values():
            signal_combobox = mouse_setting.signal_selector
            signal_combobox.clear()
            signal_combobox.addItems("-")
            signal_combobox.addItems(signals)
            signal_combobox.adjustSize()

    def update_signal_values(self, signals):
        for mouse_setting in self.mouse_settings.values():
            signal_name = mouse_setting.current_signal
            signal = signals.get(signal_name, None)
            signal_value = 0.
            if signal is not None:
                signal_value = signal.scaled_value
            mouse_setting.threshold.updateBackground(signal_value)
            mouse_setting.threshold.repaint()

    def save_profile_dialog(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select profile save file", "./config/profiles",
                                                             "JSON (*.json)")
        self.save_profile(file_name)

    def save_profile(self, file_name):
        print(file_name)
        mouse_settings = {}
        for mouse_setting_name, mouse_setting_widget in self.mouse_settings.items():
            mouse_settings[mouse_setting_name]=mouse_setting_widget.as_dict()


        cursor_settings = {
            "x_sensitivity": self.x_sensitivity_slider.value(),
            "y_sensitivity": self.y_sensitivity_slider.value(),
            "x_acceleration": self.x_acceleration_slider.value(),
            "y_acceleration": self.y_acceleration_slider.value(),
            "filter_enabled": self.smoothing_toggle.isChecked(),
            "filter_value": self.smoothing_value.value(),
            "tracking_mode": self.tracking_mode_selector.currentText(),
            "mouse_mode": self.mouse_mode_selector.currentText()
        }

        with open(file_name, "w") as f:
            json.dump({
                "mouse_setting": mouse_settings,
                "cursor_setting": cursor_settings
            }, f, indent=2)

    def load_profile_dialog(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select profile to load", "./config/profiles",
                                                             "JSON (*.json)")
        self.load_profile(file_name)

    def load_profile(self, file_name):
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"No such file or directory: {file_name}")
        # TODO: maybe clear stuff, but will prbably do while loading
        with open(file_name, "r") as f:
            json_profile = json.load(f)
            mouse_settings = json_profile.get("mouse_setting", {})
            cursor_settings = json_profile.get("cursor_setting", {})

            for mouse_setting_name, mouse_setting_values in mouse_settings.items():
                setting = self.mouse_settings.get(mouse_setting_name, None)
                if setting is not None:
                    setting.from_dict(mouse_setting_values)

            x_sensitivity =  cursor_settings.get("x_sensitivity", 1.)
            self.x_sensitivity_slider.setValue(x_sensitivity)
            self.x_sensitivity_slider.emitDoubleValueChanged()
            y_sensitivity =  cursor_settings.get("y_sensitivity", 1.)
            self.y_sensitivity_slider.setValue(y_sensitivity)
            self.y_sensitivity_slider.emitDoubleValueChanged()
            x_acceleration =  cursor_settings.get("x_acceleration", 1.25)
            self.x_acceleration_slider.setValue(x_acceleration)
            self.x_acceleration_slider.emitDoubleValueChanged()
            y_acceleration =  cursor_settings.get("y_acceleration", 1.25)
            self.y_acceleration_slider.setValue(y_acceleration)
            self.y_acceleration_slider.emitDoubleValueChanged()
            filter_enabled =  cursor_settings.get("filter_enabled", True)
            self.smoothing_toggle.setChecked(filter_enabled)
            self.smoothing_toggle.toggled.emit(filter_enabled)
            filter_value =  cursor_settings.get("filter_value", 0.01)
            self.smoothing_value.setValue(filter_value)
            self.smoothing_value.emitDoubleValueChanged()
            tracking_mode =  cursor_settings.get("tracking_mode", "MEDIAPIPE")
            self.tracking_mode_selector.setCurrentText(tracking_mode)
            self.tracking_mode_selector.currentTextChanged.emit(tracking_mode)
            mouse_mode =  cursor_settings.get("mouse_mode", "ABSOLUTE")
            self.mouse_mode_selector.setCurrentText(mouse_mode)
            self.mouse_mode_selector.currentTextChanged.emit(mouse_mode)

    def save_lates(self):
        file_name="config/profiles/mouse_latest.json"
        self.save_profile(file_name)



class MouseClickSettings(QtWidgets.QWidget):
    trigger_save = QtCore.Signal()
    def __init__(self, name, demo, mouse_callback):
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        self.demo: Demo.Demo = demo
        self.uid = uuid.uuid4()
        self.label = QtWidgets.QLabel(name)
        self.threshold = ColoredDoubleSlider()
        self.threshold.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.threshold.setMinimum(0.)
        self.threshold.setMaximum(1.)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.5)
        self.threshold.doubleValueChanged.connect(self.threshold_changed)
        self.delay = QtWidgets.QDoubleSpinBox(self)
        self.delay.setMinimum(0.)
        self.delay.setMaximum(2.)
        self.delay.setSingleStep(0.01)
        self.delay.setValue(0.5)
        self.delay.valueChanged.connect(self.delay_changed)
        self.signal_selector = QtWidgets.QComboBox()
        self.signal_selector.currentTextChanged.connect(self.signal_changed)

        self.callback = mouse_callback
        self.current_signal = "-"
        self.action = Gesture.GestureAction()
        self.action.set_action_callable(Gesture.ActionTrigger.UP_ACTION.name, self.callback)
        self.action.set_delay(self.delay.value())
        self.action.set_threshold(self.threshold.value())

        layout.addWidget(self.label,0,0,1,1)
        layout.addWidget(self.signal_selector,0,1,1,1)
        layout.addWidget(QtWidgets.QLabel("Delay"),0,2,1,1,alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.delay,0,3,1,1)
        layout.addWidget(QtWidgets.QLabel("Threshold"),1,0,1,1)
        layout.addWidget(self.threshold,1,1,1,3)

    def signal_changed(self, current_text):
        # remove action from old signal
        self.trigger_save.emit()
        old_signal = self.demo.signals.get(self.current_signal,None)
        if old_signal is not None:
            old_signal.remove_action(self.uid)
        self.current_signal = current_text

        #add action to selected signal
        signal = self.demo.signals.get(current_text, None)
        if signal is None:
            # No valid signal selected
            return
        signal.add_action(self.uid, self.action)



    def threshold_changed(self, new_threshold):
        self.trigger_save.emit()
        self.action.set_threshold(new_threshold)
        signal = self.demo.signals.get(self.signal_selector.currentText(), None)
        if signal is None:
            return  # no signal detected
        action = signal.actions.get(self.uid, None)
        if action is None:
            return  # action not defined
        action.set_threshold(new_threshold)

    def delay_changed(self, new_delay):
        self.trigger_save.emit()
        self.action.set_delay(new_delay)
        signal = self.demo.signals.get(self.signal_selector.currentText(), None)
        if signal is None:
            return  # no signal detected
        action = signal.actions.get(self.uid, None)
        if action is None:
            return  # action not defined
        action.set_delay(new_delay)

    def as_dict(self):
        settings = {
            "threshold": self.threshold.value(),
            "signal": self.signal_selector.currentText(),
            "delay": self.delay.value()
        }
        return settings

    def from_dict(self, setting:Dict):
        new_threshold = setting.get("threshold",0.5)
        self.threshold.setValue(new_threshold)
        self.threshold_changed(new_threshold)
        new_signal = setting.get("signal", "-")
        self.signal_selector.setCurrentText(new_signal)
        self.signal_changed(new_signal)
        new_delay = setting.get("delay", 0.5)
        self.delay.setValue(new_delay)
        self.delay_changed(new_delay)


class KeyboardActionWidget(QtWidgets.QWidget):
    remove_clicked = QtCore.Signal()
    action_updated = QtCore.Signal()

    def __init__(self, name: uuid.UUID):
        super().__init__()
        self.name: uuid.UUID = name
        self.current_signal: str = ""

        self.layout = QtWidgets.QHBoxLayout(self)
        self.threshold = ColoredDoubleSlider(self, decimals=3)
        self.threshold.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.threshold.setMinimum(0.)
        self.threshold.setMaximum(1.)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.5)
        self.threshold.doubleValueChanged.connect(self._emit_updated)
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
        self.add_action_button = QtWidgets.QPushButton("Add Gesture Action")
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
        self.save_actions_button.clicked.connect(self.save_profile_dialog)
        self.load_actions_button = QtWidgets.QPushButton("Load profile")
        self.load_actions_button.clicked.connect(self.load_profile_dialog)
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
        action_widget.action_updated.connect(self.update_and_save)
        action_widget.set_signal_selector(self.signals)
        self.save_profile("config/profiles/keyboard_latest.json")

    def set_signals(self, signals: List[str]):
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
        self.save_profile("config/profiles/keyboard_latest.json")

    def update_and_save(self):
        self.update_action()
        self.save_profile("config/profiles/keyboard_latest.json")

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
        new_action = Gesture.GestureAction()
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
            new_action.set_action_callable(Gesture.ActionTrigger.UP_ACTION.name, action_function)
        elif trigger == "down":
            new_action.set_action_callable(Gesture.ActionTrigger.DOWN_ACTION.name, action_function)
        elif trigger == "hold high":
            new_action.set_action_callable(Gesture.ActionTrigger.HIGH_HOLD_ACTION.name, action_function)
        elif trigger == "hold low":
            new_action.set_action_callable(Gesture.ActionTrigger.LOW_HOLD_ACTION.name, action_function)
        else:
            return
        new_action.set_delay(delay)
        signal.add_action(uid, new_action)

    def save_profile_dialog(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select profile save file", "./config/profiles",
                                                             "JSON (*.json)")
        self.save_profile(file_name)

    def save_profile(self, file_name):
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

    def load_profile_dialog(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select profile to load", "./config/profiles",
                                                             "JSON (*.json)")
        self.load_profile(file_name)

    def load_profile(self, file_name):
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
                action_widget.action_updated.connect(self.update_and_save)
                action_widget.action_updated.emit()  # create associated action

    def update_signal_values(self, signals):
        for action_name in self.actions:
            action_widget = self.actions[action_name]
            signal_name = action_widget.current_signal
            signal = signals.get(signal_name, None)
            signal_value = 0.
            if signal is not None:
                signal_value = signal.scaled_value
            action_widget.threshold.updateBackground(signal_value)
            action_widget.threshold.repaint()



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.demo = Demo.Demo()
        self.central_widget = QtWidgets.QTabWidget()

        self.signal_tab_iphone = SignalTab(self.demo, "config/iphone_default.json", "iphone")
        self.signal_tab_mediapipe = SignalTab(self.demo, "config/mediapipe_blendshape.json", "mediapipe")

        self.signals_tab = QtWidgets.QStackedWidget()
        self.signals_tab.addWidget(self.signal_tab_iphone)
        self.signals_tab.addWidget(self.signal_tab_mediapipe)
        self.signal_tab_iphone.signals_updated.connect(self.update_signals)
        self.signal_tab_mediapipe.signals_updated.connect(self.update_signals)

        self.general_tab = GeneralTab(self.demo)
        #self.general_tab.mediapipe_selector_button.clicked.connect(lambda selected: self.change_signals_tab(selected))
        self.general_tab.mode_changed.connect(self.change_mode)

        self.keyboard_tab = KeyboardTab(self.demo)
        self.keyboard_tab.load_profile("config/profiles/keyboard_latest.json")
        self.mouse_tab = MouseTab(self.demo)
        self.mouse_tab.load_profile("config/profiles/mouse_latest.json")

        self.central_widget.addTab(self.general_tab, "General")
        self.central_widget.addTab(self.keyboard_tab, "Keyboard")
        self.central_widget.addTab(self.mouse_tab, "Mouse")
        self.central_widget.addTab(self.signals_tab, "GestureSignal")

        self.setCentralWidget(self.central_widget)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start()

        self.change_mode("WEBCAM")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        ## Signals
        self.demo.start()

        self.setup_global_hooks()

    def setup_global_hooks(self):
        """
        Sets up global hooks for the application.
        """
        # add hotkey
        print("Starting global hotkeys")
        keyboard.GlobalHotKeys({
            '<ctrl>+<alt>+w': self.general_tab.toggle_debug_window_globally,
            '<ctrl>+<alt>+e': self.demo.toggle_gesture_mouse,
            '<ctrl>+<alt>+g': self.demo.toggle_gestures,
            '<ctrl>+<alt>+m': self.demo.toggle_mouse_movement,
            '<ctrl>+<alt>+v': self.demo.toggle_tracking,
            '<shift>+<alt>+m': self.demo.toggle_mouse_mode,
            '<shift>+<alt>+c': self.demo.mouse.centre_mouse,
            '<shift>+<alt>+s': self.demo.mouse.switch_monitor,
            '<shift>+<alt>+r': self.demo.mouse.toggle_tracking_mode
        }).start()


        #keyboard.add_hotkey("alt + 1", lambda: self.toggle_gesture_mouse())  # TODO: Linux alternative
        #keyboard.add_hotkey("alt + g", lambda: self.toggle_gestures())
        #keyboard.add_hotkey("alt + m", lambda: self.toggle_mouse_movement())
        #keyboard.add_hotkey("alt + t", lambda: self.toggle_tracking())
        #keyboard.add_hotkey("m", lambda: self.toggle_mouse_mode())
        #keyboard.add_hotkey("c", lambda: self.mouse.centre_mouse())
        #keyboard.add_hotkey(".", lambda: self.mouse.switch_monitor())
        #keyboard.add_hotkey("t", lambda: self.mouse.toggle_tracking_mode())
        # keyboard.on_press_key("r", lambda e: self.disable_gesture_mouse())
        # keyboard.on_release_key("r", lambda e: self.enable_gesture_mouse())
        # add mouse_events

    def update_plots(self):
        # TODO: move up again
        self.selected_signals.update_plots(self.demo.signals)
        self.general_tab.update_debug_visualization()
        self.keyboard_tab.update_signal_values(self.demo.signals)
        self.mouse_tab.mouse_mode_selector.setCurrentText(self.demo.mouse.mode.name)
        self.mouse_tab.update_signal_values(self.demo.signals)

    def change_mode(self, mode:str):
        if mode == "WEBCAM":
            self.signals_tab.setCurrentIndex(1)
            self.selected_signals = self.signal_tab_mediapipe
        elif mode == "IPHONE":
            self.signals_tab.setCurrentIndex(0)
            self.selected_signals = self.signal_tab_iphone
        else:
            self.signals_tab.setCurrentIndex(1)
            self.selected_signals = self.signal_tab_mediapipe

        self.update_signals()

    def update_signals(self):
        self.mouse_tab.set_signal_selector(list(self.selected_signals.signal_settings.keys()))
        self.keyboard_tab.set_signals(list(self.selected_signals.signal_settings.keys()))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.demo.stop()
        QApplication.closeAllWindows()
        event.accept()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        #self.demo.disable_gesture_mouse()
        # TODO: disable actions when window has focus
        super().focusInEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        #print("Lost Focus")
        #self.demo.enable_gesture_mouse()
        super().focusOutEvent(event)

def test_gui():
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(1280, 720)
    window.setWindowTitle("Gesture Mouse")

    if platform.system() == 'Windows':
        # close splash screen
        try:
            import pyi_splash
            pyi_splash.close()
        except Exception as inst:
            print(f"Splash screen not supported on this platform: {inst}")

    window.show()
    window.activateWindow()
    app.exec()
        
if __name__ == '__main__':
    test_gui()
