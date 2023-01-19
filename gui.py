import sys

from PySide6 import QtWidgets, QtCore, QtGui
from gui_widgets import LogarithmicSlider
import pyqtgraph as pg
import time
import pygame

import Demo
import SignalsCalculator


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
            y = getattr(signals, name).get()
            plot.plot(x, y)


class SignalSetting(QtWidgets.QWidget):
    def __init__(self, name: str, min_value, max_value, min_filter=0.0001, max_filter=1.):
        super().__init__()
        self.name_label = QtWidgets.QLabel(name)

        self.lower_value = QtWidgets.QDoubleSpinBox()
        self.lower_value.setMaximum(max_value)
        self.lower_value.setMinimum(min_value)

        self.higher_value = QtWidgets.QDoubleSpinBox()
        self.higher_value.setMaximum(max_value)
        self.higher_value.setMinimum(min_value)

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
    def __init__(self, demo):
        #TODO: move signals to file
        super().__init__()
        self.demo = demo
        self.setWindowTitle("Signals Visualization")
        self.signals_vis = SignalVis()

        self.pitch = SignalSetting("pitch", -90., 90.)
        handler = self.signals_vis.add_line("pitch")
        self.pitch.visualization_checkbox.stateChanged.connect(handler.set_visible)
        self.pitch.visualization_checkbox.setChecked(False)
        self.pitch.filter_slider.doubleValueChanged.connect(lambda x: self.demo.set_filter_value("pitch", x))

        self.roll = SignalSetting("roll", -90., 90.)
        handler = self.signals_vis.add_line("roll")
        self.roll.visualization_checkbox.stateChanged.connect(handler.set_visible)
        self.roll.visualization_checkbox.setChecked(False)
        self.roll.filter_slider.doubleValueChanged.connect(lambda x: self.demo.set_filter_value("roll", x))

        self.yaw = SignalSetting("yaw", -90., 90.)
        handler = self.signals_vis.add_line("yaw")
        self.yaw.visualization_checkbox.stateChanged.connect(handler.set_visible)
        self.yaw.visualization_checkbox.setChecked(False)
        self.yaw.filter_slider.doubleValueChanged.connect(lambda x: self.demo.set_filter_value("yaw", x))

        self.jaw_open = SignalSetting("jaw_open", 0, 50)
        handler = self.signals_vis.add_line("jaw_open")
        self.jaw_open.visualization_checkbox.stateChanged.connect(handler.set_visible)
        self.jaw_open.visualization_checkbox.setChecked(False)
        self.jaw_open.filter_slider.doubleValueChanged.connect(lambda x: self.demo.set_filter_value("jaw_open", x))

        self.mouth_puck = SignalSetting("mouth_puck", 0, 50)
        handler = self.signals_vis.add_line("mouth_puck")
        self.mouth_puck.visualization_checkbox.stateChanged.connect(handler.set_visible)
        self.mouth_puck.visualization_checkbox.setChecked(False)
        self.mouth_puck.filter_slider.doubleValueChanged.connect(lambda x: self.demo.set_filter_value("mouth_puck", x))

        self.debug1 = SignalSetting("debug1", 0, 50)
        handler = self.signals_vis.add_line("debug1")
        self.debug1.visualization_checkbox.stateChanged.connect(handler.set_visible)
        self.debug1.visualization_checkbox.setChecked(False)
        self.debug1.filter_slider.doubleValueChanged.connect(lambda x: self.demo.set_filter_value("debug1", x))

        self.debug2 = SignalSetting("debug2", 0, 50)
        handler = self.signals_vis.add_line("debug2")
        self.debug2.visualization_checkbox.stateChanged.connect(handler.set_visible)
        self.debug2.visualization_checkbox.setChecked(False)
        self.debug2.filter_slider.doubleValueChanged.connect(lambda x: self.demo.set_filter_value("debug2", x))

        self.debug3 = SignalSetting("debug3", 0, 50)
        handler = self.signals_vis.add_line("debug3")
        self.debug3.visualization_checkbox.stateChanged.connect(handler.set_visible)
        self.debug3.visualization_checkbox.setChecked(False)
        self.debug3.filter_slider.doubleValueChanged.connect(lambda x: self.demo.set_filter_value("debug3", x))

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.signals_vis)
        self.layout.addWidget(self.pitch)
        self.layout.addWidget(self.roll)
        self.layout.addWidget(self.yaw)
        self.layout.addWidget(self.jaw_open)
        self.layout.addWidget(self.mouth_puck)
        self.layout.addWidget(self.debug1)
        self.layout.addWidget(self.debug2)
        self.layout.addWidget(self.debug3)

    def update_plots(self, signals):
        self.signals_vis.update_plot(signals)


class GeneralTab(QtWidgets.QWidget):
    def __init__(self, demo):
        super().__init__()
        self.demo = demo
        self.mediapipe_selector_button = QtWidgets.QRadioButton(text="Use web cam tracking.")
        self.mediapipe_selector_button.setChecked(False)
        self.mediapipe_selector_button.clicked.connect(lambda selected: self.demo.set_use_mediapipe(selected))
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.mediapipe_selector_button)

class MouseTab(QtWidgets.QWidget):
    def __init__(self, demo):
        super().__init__()


class KeyboardTab(QtWidgets.QWidget):
    def __init__(self, demo):
        super().__init__()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.demo = Demo.Demo()

        self.central_widget = QtWidgets.QTabWidget()
        self.signal_tab = SignalTab(self.demo)
        self.general_tab = GeneralTab(self.demo)
        self.keyboard_tab = KeyboardTab(self.demo)
        self.mouse_tab = MouseTab(self.demo)

        self.central_widget.addTab(self.general_tab, "General")
        self.central_widget.addTab(self.keyboard_tab, "Keyboard")
        self.central_widget.addTab(self.mouse_tab, "Mouse")
        self.central_widget.addTab(self.signal_tab, "Signal")

        self.setCentralWidget(self.central_widget)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start()

        ## Signals

        self.demo.start()

    def update_plots(self):
        # TODO: move up again
        self.signal_tab.update_plots(self.demo.raw_signal)


def test_gui():
    pygame.init()
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    app.exec()


if __name__ == '__main__':
    test_gui()
