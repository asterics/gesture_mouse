from PySide6.QtWidgets import QSlider, QStyle
from PySide6.QtCore import Signal, Qt, QPoint
from PySide6.QtGui import QPaintEvent, QPainter, QBrush, QColorConstants, QPen, QFontMetrics
import math


class DoubleSlider(QSlider):
    # create our our signal that we can connect to if necessary
    doubleValueChanged = Signal(float)

    def __init__(self, decimals=3, *args, **kargs):
        super(DoubleSlider, self).__init__(*args, **kargs)
        self._multi = 10 ** decimals

        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value()) / self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))

class ColoredDoubleSlider(QSlider):
    # create our our signal that we can connect to if necessary
    doubleValueChanged = Signal(float)

    def __init__(self,parent=None, decimals=3, *args, **kargs):
        super(ColoredDoubleSlider, self).__init__(parent)
        self._multi = 10 ** decimals

        self.valueChanged.connect(self.emitDoubleValueChanged)
        self.background_value = 0.

    def emitDoubleValueChanged(self):
        value = float(super(ColoredDoubleSlider, self).value()) / self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(ColoredDoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(ColoredDoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(ColoredDoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(ColoredDoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(ColoredDoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(ColoredDoubleSlider, self).setValue(int(value * self._multi))

    def updateBackground(self, value):
        assert 0 <= value <= 1
        self.background_value = value

    def paintEvent(self, ev: QPaintEvent) -> None:
        orientation = self.orientation()
        qpainter = QPainter(self)
        qpainter.setBrush(QBrush(QColorConstants.Red))
        qpainter.begin(self)
        if orientation == Qt.Orientation.Horizontal:
            qpainter.fillRect(0,0,self.background_value*self.width(), self.height(),QColorConstants.Svg.lightblue)
        else:
            qpainter.fillRect(0, (1.-self.background_value) * self.height(), self.width(), self.background_value * self.height(), QColorConstants.Svg.lightblue)
        qpainter.end()
        super().paintEvent(ev)


class LogarithmicSlider(QSlider):
    # create our our signal that we can connect to if necessary
    doubleValueChanged = Signal(float)

    def __init__(self, *args, **kargs):
        super(LogarithmicSlider, self).__init__(*args, **kargs)
        decimals = 5
        self._multi = 10 ** decimals
        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = 10**(float(super(LogarithmicSlider, self).value()) / self._multi)
        self.doubleValueChanged.emit(value)

    def value(self):
        return 10**(float(super(LogarithmicSlider, self).value()) / self._multi)

    def setMinimum(self, value):
        if value <= 0: raise ValueError("Value has to be bigger than 0")
        return super(LogarithmicSlider, self).setMinimum(math.log10(value) * self._multi)

    def setMaximum(self, value):
        if value < 0: raise ValueError("Value has to be bigger than 0")
        return super(LogarithmicSlider, self).setMaximum(math.log10(value) * self._multi)

    def setSingleStep(self, value):
        return super(LogarithmicSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(LogarithmicSlider, self).singleStep()) / self._multi

    def setValue(self, value):

        print(int(value*self._multi))
        super(LogarithmicSlider, self).setValue(int(math.log10(value) * self._multi))

class StyledMouseSlider(DoubleSlider):
    def __init__(self, decimals=3, *args, **kargs):
        super().__init__(decimals, *args, **kargs)
        self.setMinimum(0)
        self.setMaximum(3)
        self.setValue(1.)
        self.setOrientation(Qt.Orientation.Horizontal)
        self.setTracking(False)
        self.setTickInterval(10**decimals)
        self.setTickPosition(self.TickPosition.TicksBelow)

