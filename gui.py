#!/usr/bin/env python3

import json
import time
import uuid
from typing import List, Dict

from pynput import mouse
from pynput import keyboard
import pygame
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui

import Demo
import Signal
from gui_widgets import LogarithmicSlider
import re


class PlotLine:
    def __init__(self, pen, plot_data_item: pg.PlotDataItem):
        self.x = [0.] * 100
        self.y = [0.] * 100
        self.length = 0
        self.max_length = 100
        self.pen = pen
        self.plot_data_item = plot_data_item

    def plot(self, x, y):
        if self.length < self.max_length:
            self.x[self.length] = x
            self.y[self.length] = y
            self.length += 1
        else:
            self.x = self.x[1:]
            self.x.append(x)
            self.y = self.y[1:]
            self.y.append(y)
        self.plot_data_item.setData(self.x, self.y, pen=self.pen)

    def set_visible(self, visibility):
        self.plot_data_item.setVisible(visibility)


class SignalVis(pg.PlotWidget):
    def __init__(self):
        super(SignalVis, self).__init__()
        self.setBackground('w')
        self.lines = {}

    def add_line(self, name: str):
        pen = pg.mkPen(color=(255, 0, 0))
        data_line = self.plot(x=[90, -90] * 50, y=[0] * 100, pen=pen)
        plot_handler = PlotLine(pen, data_line)
        self.lines[name] = plot_handler
        return plot_handler

    def update_plot(self, signals):
        x = time.time()
        for name, plot in self.lines.items():
            y = signals[name].scaled_value
            plot.plot(x, y)


class SignalSetting(QtWidgets.QWidget):
    def __init__(self, name: str, min_value, max_value, min_filter=0.0001, max_filter=1.):
        super().__init__()
        self.name_label = QtWidgets.QLabel(name)

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

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.visualization_checkbox)
        self.layout.addWidget(QtWidgets.QLabel("Signal range"))
        self.layout.addWidget(self.lower_value)
        self.layout.addWidget(self.higher_value)
        self.layout.addWidget(QtWidgets.QLabel("Filter"))
        self.layout.addWidget(self.filter_slider)

        self.filter_slider.doubleValueChanged.connect(lambda value: print(value))


class SignalTab(QtWidgets.QWidget):
    def __init__(self, demo, json_path):
        super().__init__()
        self.demo = demo
        self.signal_config_defaults = json.load(open(json_path, "r"))
        self.setWindowTitle("Signals Visualization")
        self.signals_vis = SignalVis()
        self.signals_vis.setMaximumHeight(250)
        self.signals_vis.setMinimumHeight(100)
        size_policy = self.signals_vis.sizePolicy()
        size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Maximum)
        size_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
        self.signals_vis.setSizePolicy(size_policy)

        self.signal_settings = dict()
        self.layout = QtWidgets.QVBoxLayout(self)

        self.layout.addWidget(self.signals_vis)
        self.setting_widget = QtWidgets.QWidget()
        self.setting_widget.setLayout(QtWidgets.QVBoxLayout())

        for json_signal in self.signal_config_defaults:
            signal_name = json_signal["name"]
            lower_threshold = json_signal["lower_threshold"]
            higher_threshold = json_signal["higher_threshold"]
            filter_value = json_signal["filter_value"]

            setting = SignalSetting(signal_name, lower_threshold, higher_threshold)
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

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.setting_widget)
        self.layout.addWidget(self.scroll_area)

    def update_plots(self, signals):
        self.signals_vis.update_plot(signals)


class DebugVisualizetion(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.webcam_label = QtWidgets.QLabel()
        self.webcam_label.setMinimumSize(1, 1)
        self.webcam_label.setMaximumSize(1280, 720)
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
        self.demo = demo
        self.mediapipe_selector_button = QtWidgets.QCheckBox(text="Use web cam tracking.")
        self.mediapipe_selector_button.setChecked(False)
        self.mediapipe_selector_button.clicked.connect(lambda selected: self.demo.set_use_mediapipe(selected))
        self.landmark_filter_button = QtWidgets.QCheckBox(text="Filter Landmarks.")
        self.landmark_filter_button.setChecked(False)
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
        self.debug_window.status_bar.showMessage(f"FPS: {self.demo.fps}")


class MouseTab(QtWidgets.QWidget):
    def __init__(self, demo):
        super().__init__()
        self.demo: Demo.Demo = demo
        layout = QtWidgets.QVBoxLayout(self)
        self.left_click_settings = MouseClickSettings("Left Click")
        self.left_click_signal = "-"
        self.left_click_uid = uuid.uuid4()
        self.left_click_settings.signal_selector.currentTextChanged.connect(self.set_left_click)
        self.right_click_settings = MouseClickSettings("Right Click")
        self.right_click_signal = "-"
        self.right_click_uid = uuid.uuid4()
        self.right_click_settings.signal_selector.currentTextChanged.connect(self.set_right_click)
        self.double_click_settings = MouseClickSettings("Double Click")
        self.double_click_signal = "-"
        self.double_click_uid = uuid.uuid4()
        self.double_click_settings.signal_selector.currentTextChanged.connect(self.set_double_click)
        layout.addWidget(self.left_click_settings)
        layout.addWidget(self.right_click_settings)
        layout.addWidget(self.double_click_settings)

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
        self.demo.signals[selected_text].add_action(self.double_click_uid, action)


class MouseClickSettings(QtWidgets.QWidget):
    def __init__(self, name):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        self.label = QtWidgets.QLabel(name)
        self.signal_selector = QtWidgets.QComboBox()
        layout.addWidget(self.label)
        layout.addWidget(self.signal_selector)


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
        self.signal_selector = QtWidgets.QComboBox()
        self.signal_selector.currentTextChanged.connect(self._emit_updated)
        self.action_trigger_selector = QtWidgets.QComboBox()
        self.action_trigger_selector.addItems(["-", "up", "down", "hold high", "hold low"])
        self.action_trigger_selector.currentTextChanged.connect(self._emit_updated)
        self.action_type_selector = QtWidgets.QComboBox()
        self.action_type_selector.addItems(["-", "press", "release", "hold", "press and release"])
        self.action_type_selector.currentTextChanged.connect(self._emit_updated)
        self.key_input = QtWidgets.QKeySequenceEdit()
        self.key_input.setClearButtonEnabled(True)
        self.key_input.editingFinished.connect(self._emit_updated)
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_clicked.emit)
        self.layout.addWidget(self.signal_selector)
        self.layout.addWidget(self.threshold)
        self.layout.addWidget(self.action_trigger_selector)
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
        button_layout = QtWidgets.QHBoxLayout()
        self.save_actions_button = QtWidgets.QPushButton("Save profile")
        self.save_actions_button.clicked.connect(self.save_action)
        self.load_actions_button = QtWidgets.QPushButton("Load profile")
        self.load_actions_button.clicked.connect(self.load_profile)
        self.layout.addStretch()

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
        self.layout.insertWidget(self.layout.count() - 2, action_widget)
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
        self.layout.removeWidget(action_widget)
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
        elif action_type == "hold":
            # Todo: Is this needed? What should this mode do
            def action_function():
                for key_combo in parsed_hotkeys:
                    for key in key_combo:
                        self.keyboard_controller.press(key)
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
            serial_action = {
                "action": f"keyboard_key",
                "signal": signal,
                "threshold": threshold,
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
        for action in self.actions.values():
            signal_name = action.current_signal
            signal = self.demo.signals.get(signal_name, None)
            if signal is not None:
                signal.remove_action(action.name)
            self.layout.removeWidget(action)
            action.close()
        self.actions.clear()
        with open(file_name, "r") as f:
            json_profile = json.load(f)
            for action in json_profile:
                action_mapping = action["action"]
                if action_mapping != "keyboard_key":
                    continue
                signal = action["signal"]
                threshold = float(action["threshold"])
                trigger = action["trigger"]
                action_type = action["action_type"]
                key = action["key"]

                # add widget
                name = uuid.uuid4()
                action_widget = KeyboardActionWidget(name=name)
                action_widget.set_signal_selector(self.signals)
                action_widget.signal_selector.setCurrentText(signal)
                action_widget.threshold.setValue(threshold)
                action_widget.action_trigger_selector.setCurrentText(trigger)
                action_widget.action_type_selector.setCurrentText(action_type)
                action_widget.key_input.setKeySequence(key)
                self.actions[action_widget.name] = action_widget
                self.layout.insertWidget(self.layout.count() - 2, action_widget)
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

        self.change_signals_tab(False)

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
