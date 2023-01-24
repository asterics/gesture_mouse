import pickle

from SignalsCalculator import FilteredFloat
import keyboard
from typing import Callable
import mouse


def null_f():
    pass


class Action:
    def __init__(self):
        self.old_value: float = 0.
        self.up_action: Callable[[], None] = null_f
        self.down_action: Callable[[], None] = null_f
        self.hold_action: Callable[[], None] = null_f
        self.threshold: float = 0.5

    def update(self, value: float):
        """
        Updates the value and triggers functions according to the value of threshold and old value.
        down action if value <= threshold < old_value
        up action if value > threshold >= old_value
        hold_action if value > threshold and old_value > threshold
        sets old_value to value
        :param value: new signal value for this action
        """
        if value <= self.threshold < self.old_value:
            self.down_action()
        elif value > self.threshold >= self.old_value:
            self.up_action()
        elif value > self.threshold and self.old_value >= self.threshold:
            self.hold_action()

        self.old_value = value

    def set_up_action(self, action: Callable[[], None]):
        """
        Sets the action for exceeding the threshold, i.e. value > threshold >= old_value:
        :param action: Function to be executed when threshold is exceeded
        """
        self.up_action = action

    def set_down_action(self, action: Callable[[], None]):
        """
        Sets the action for falling below the threshold, i.e. value <= threshold < old_value
        :param action: Function to be executed when threshold is exceeded
        """
        self.down_action = action

    def set_hold_action(self, action: Callable[[], None]):
        """
        Sets the action for staying above the threshold, i.e. value > threshold and old_value > threshold
        :param action:
        """
        self.hold_action = action

    def set_threshold(self, value: float):
        """
        Sets the threshold for this action
        :param value: New threshold
        """
        self.threshold = value


class Signal:
    def __init__(self, name: str):
        self.name = name
        self.raw_value: FilteredFloat = FilteredFloat(0, 0.0001)
        self.scaled_value: float = 0.
        self.action: Action = Action()  # TODO: maybe list (multiple actions triggered with one event)
        self.lower_threshold: float = 0.
        self.higher_threshold: float = 1.

    def set_value(self, value):
        """
        Sets the value of the signal and scales the result between 0 and 1 according to the lower and higher threshold.
        If lower > higher threshold then the sign will be flipped (higher threshold -> 0, lower_threshold -> 1).
        It then updates the action associated with this signal
        :param value: new value of signal
        """
        filtered_value = self.raw_value.set(value)
        self.scaled_value = max(
            min((filtered_value - self.lower_threshold) / (self.higher_threshold - self.lower_threshold), 0.), 1.)
        self.action.update(value)

    def set_threshold(self, lower_threshold: float, higher_threshold: float):
        """
        Sets the lower and higher threshold. Keeps the old threshold if lower or higher threshold is None
        :param lower_threshold: New value for lower threshold or None
        :param higher_threshold: New value for higher threshold or None
        """
        if lower_threshold is not None:
            self.lower_threshold = lower_threshold
        if higher_threshold is not None:
            self.higher_threshold = higher_threshold

    def set_filter_value(self, filter_value):
        """
        Sets the R parameter for the Kalman filter.
        :param filter_value: new value for filter, higher = stronger filter
        :return:
        """
        self.raw_value.set_filter_value(filter_value)

