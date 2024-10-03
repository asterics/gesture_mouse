import pickle
from enum import Enum

from SignalsCalculator import FilteredFloat
from typing import Callable, Dict
import uuid
import time


def null_f():
    pass

class ActionTrigger(Enum):
    UP_ACTION = 0
    DOWN_ACTION = 1
    HIGH_HOLD_ACTION = 2
    LOW_HOLD_ACTION = 3

class GestureAction:
    def __init__(self):
        # dictionary of actions to be executed when conditions are met
        self.action_callables = dict()

        self.old_value: float = 0.

        # up action helpers
        self.up_starttime: float = 0.
        self.up_action_active: bool = False
        self.up_activated = False

        # down action helpers
        self.down_action_active: bool = False
        self.down_starttime: float = 0.
        self.down_activated: bool = False

        # hold low helpers
        self.hold_low_active: bool = False

        # hold high helpers
        self.hold_high_active = False

        self.threshold: float = 0.5
        self.delay: float = 0.5

    def update(self, value: float):
        """
        Updates the value and triggers functions according to the value of threshold and old value.
        down action if value <= threshold < old_value
        up action if value > threshold >= old_value
        high_hold_action if value > threshold and old_value > threshold
        low_hold_action if value <= threshold and old_value <= threshold
        sets old_value to value
        :param value: new signal value for this action
        """
        new_time = time.time()

        if value <= self.threshold < self.old_value:
            # Start down action
            self.down_action_active = True
            self.down_starttime = time.time()

            # Start hold low action
            self.hold_low_starttime = time.time()
            self.hold_low_active = True

            # Stop up action
            self.up_action_active = False
            self.up_activated = False

            # Stop hold high action
            self.hold_high_active = False

        elif value > self.threshold >= self.old_value:
            # Start up action
            self.up_action_active = True
            self.up_starttime = time.time()

            # Start hold high action
            self.hold_high_starttime = time.time()
            self.hold_high_active = True

            # Stop down action
            self.down_action_active = False
            self.down_activated = False

            # Stop hold low action
            self.hold_low_active = False

        elif value > self.threshold:
            # check delay
            if (new_time - self.up_starttime) >= self.delay:
                if not self.up_activated:
                    self.execute_action(ActionTrigger.UP_ACTION.name)
                    self.up_activated = True

                self.execute_action(ActionTrigger.HIGH_HOLD_ACTION.name)
        elif value <= self.threshold:
            # check delay
            if (new_time - self.down_starttime) >= self.delay:
                if not self.down_activated:
                    self.execute_action(ActionTrigger.DOWN_ACTION.name)
                    self.down_activated = True

                self.execute_action(ActionTrigger.LOW_HOLD_ACTION.name)
        self.old_value = value

    def set_action_callable(self, action_key: str, action_callable: Callable[[], None]):
        """
        Sets the action for exceeding the threshold, i.e. value > threshold >= old_value:
        :param action: Function to be executed when threshold is exceeded
        """
        self.action_callables[action_key] = action_callable

    def set_threshold(self, value: float):
        """
        Sets the threshold for this action
        :param value: New threshold
        """
        self.threshold = value

    def set_delay(self, value: float):
        """
        Sets the amount of time a signal has to be present until the action is performed
        :param value: activation time in seconds, has to ber >= 0
        :return:
        """
        assert value >= 0
        self.delay = value

    def execute_action(self, action_key: str):
        """
        Executes the action with the key action_key
        :param action_key: key of action to execute
        """

        # execute action only if not null
        if action_key and action_key in self.action_callables:
            print(f"executing action {action_key}")
            self.action_callables[action_key]()

    @property
    def is_activated(self):
        # return True if up or down action is activated and up_action or down_action is not None
        if (ActionTrigger.UP_ACTION.name in self.action_callables and self.up_action_active == True):
            return True
        elif (ActionTrigger.DOWN_ACTION.name in self.action_callables and self.down_action_active == True):
            return True
        elif (ActionTrigger.HIGH_HOLD_ACTION.name in self.action_callables and self.hold_high_active == True):
            return True
        elif (ActionTrigger.LOW_HOLD_ACTION.name in self.action_callables and self.hold_low_active == True):
            return True

        return False

class GestureSignal:
    def __init__(self, name: str):
        self.name = name
        self.raw_value: FilteredFloat = FilteredFloat(0, 0.0001)
        self.scaled_value: float = 0.
        self.actions: Dict[uuid.UUID, GestureAction] = {}
        self.lower_threshold: float = 0.
        self.higher_threshold: float = 1.
        self.actions_enabled=True

    def set_value(self, value):
        """
        Sets the value of the signal and scales the result between 0 and 1 according to the lower and higher threshold.
        If lower > higher threshold then the sign will be flipped (higher threshold -> 0, lower_threshold -> 1).
        It then updates the action associated with this signal
        :param value: new value of signal
        """
        self.raw_value.set(value)
        filtered_value = self.raw_value.get()
        self.scaled_value = max(
            min((filtered_value - self.lower_threshold) / (self.higher_threshold - self.lower_threshold), 1.), 0.)
        if self.actions_enabled:
            for action in self.actions.values():
                action.update(self.scaled_value)

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

    def set_lower_threshold(self, lower_threshold: float):
        """
        Sets a new lower threshold, to scale signal into 0,1 range
        :param lower_threshold: New threshold
        :return:
        """
        print(self.name, lower_threshold)
        self.lower_threshold = lower_threshold

    def set_higher_threshold(self, higher_threshold: float):
        """
        Sets a new higher threshold, to scale signal into 0,1 range
        :param higher_threshold: New threshold
        :return:
        """
        print(self.name, higher_threshold)
        self.higher_threshold = higher_threshold

    def set_filter_value(self, filter_value):
        """
        Sets the R parameter for the Kalman filter.
        :param filter_value: new value for filter, higher = stronger filter
        :return:
        """
        print(self.name, filter_value)
        self.raw_value.set_filter_value(filter_value)

    def add_action(self, uid: uuid.UUID, action: GestureAction):
        """
        Adds action to a signal
        :param uid: uuid of action
        :param action: action
        :return:
        """
        self.actions[uid] = action

    def remove_action(self, uid):
        """
        Removes action with uuid uid
        :param uid: uuid of action to remove
        :return:
        """
        self.actions.pop(uid, None)

    def set_actions_active(self, enabled:bool):
        self.actions_enabled = enabled
