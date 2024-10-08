# Gesture Mouse

A program that allows to control the mouse and keyboard with head movement and facial gestures. This project is based on the 
google mediapipe library (https://ai.google.dev/edge/mediapipe/solutions/guide).

## Experimental

This software is experimental and not yet suitable for professional use. If there is a bug, please file an issue.

## Installation instructions
Tested with Python 3.10 and PySide6.4.3. 
1. Clone repository
2. Execute the following commands in the repository folder:

```bash
pip install poetry
poetry config virtualenvs.in-project true
poetry install --no-root
```

## Running Gesture Mouse

### Release

1. Extract zip file
2. Run `gesture-mouse.exe` (Windows) or `./gesture-mouse` (Linux)

### Development

```bash
poetry run python gui.py
```

## Hotkeys

* <kbd>Ctrl</kbd><kbd>Alt</kbd>+<kbd>v</kbd>: Start/Stop video and tracking
* <kbd>Ctrl</kbd><kbd>Alt</kbd>+<kbd>g</kbd>: Enable/Disable gestures
* <kbd>Ctrl</kbd><kbd>Alt</kbd>+<kbd>m</kbd>: Enable/Disable mouse movement
* <kbd>Ctrl</kbd><kbd>Alt</kbd>+<kbd>e</kbd>: Enable/Disable gestures and mouse movement
* <kbd>Shift</kbd><kbd>Alt</kbd>+<kbd>m</kbd>: Change mouse movement mode
* <kbd>Shift</kbd><kbd>Alt</kbd>+<kbd>r</kbd>: Change mouse tracking mode
* <kbd>Shift</kbd><kbd>Alt</kbd>+<kbd>c</kbd>: Center mouse
* <kbd>Shift</kbd><kbd>Alt</kbd>+<kbd>s</kbd>: Switch primary screen for mouse movement

## Creating a release

Use the github action to create a deployment file for each platform, see .github/workflows/pyinstaller_windows.yml.
You can also optionally create a release on github with the deployment file attached.

## Algorithms

The gesture calculation (e.g. eye-blink) uses the mediapipe facial landmark detection in combination with a modified eye aspect ratio (EAR) algorithm. The EAR algorithm helps to make the gesture invariant to head movements or rotations.
 * see [function SignalsCalculator.eye_aspect_ratio](https://github.com/asterics/gesture_mouse/blob/d59c84c273acace350a404e3fe110aad15be1885/SignalsCalculator.py#L330).
 * The idea is based on the article about [eye aspect algorithm for driver drowsiness detection](https://learnopencv.com/driver-drowsiness-detection-using-mediapipe-in-python/).

![Calculation of eye aspect algorithm](https://learnopencv.com/wp-content/uploads/2022/09/03-driver-drowsiness-detection-EAR-points-768x297.png)

## Links and Credits

The work for GestureMouse has been accomplished at the UAS Technikum Wien in course of the R&D-projects [WBT (MA23 project 26-02)](https://wbt.wien) and [Inclusion International (MA23 project 33-02)](https://www.technikum-wien.at/en/research-projects/inclusion-international/), which has been supported by the [City of Vienna](https://www.wien.gv.at/kontakte/ma23/index.html).

Have a look at the [AsTeRICS Foundation homepage](https://www.asterics-foundation.org) and our other Open Source AT projects:

* AsTeRICS: [AsTeRICS framework homepage](http://www.asterics.eu), [AsTeRICS framework GitHub](https://github.com/asterics/AsTeRICS): The AsTeRICS framework provides a much higher flexibility for building assistive solutions. 
The FLipMouse is also AsTeRICS compatible, so it is possible to use the raw input data for a different assistive solution.

* FABI: [FABI: Flexible Assistive Button Interface GitHub](https://github.com/asterics/FABI): The Flexible Assistive Button Interface (FABI) provides basically the same control methods (mouse, clicking, keyboard,...), but the input
is limited to simple buttons. Therefore, this interface is at a very low price (if you buy the Arduino Pro Micro from China, it can be under 5$).

* FLipMouse: [The FLipMouse controller](https://github.com/asterics/FLipMouse): a highly sensitive finger-/lip-controller for computers and mobile devices with minimal muscle movement.

* FLipPad: [The FLipPad controller](https://github.com/asterics/FLipPad): a flexible touchpad for controlling computers and mobile devices with minimal muscle movement.

* AsTeRICS Grid: [Asterics Grid AAC Web-App](https://grid.asterics.eu): an open source, cross plattform communicator / talker for Augmented and Alternative Communication (AAC).



# Support us
Please support the development by donating to the AsTeRICS Foundation:

<div>
<a title="Donate with PayPal" href="https://www.paypal.com/donate/?hosted_button_id=38AJJNS427MJ2" target="_blank" style="margin-right:3em">
<img src="https://github.com/asterics/AsTeRICS-Grid/raw/master/app/img/donate-paypal.png" width=300/></a>
<span>&nbsp;&nbsp;&nbsp;</span>
<a title="Donate at opencollective.com" href="https://opencollective.com/asterics-foundation" target="_blank">
<img src="https://github.com/asterics/AsTeRICS-Grid/raw/master/app/img/donate-open-collective.png" width=300/></a>
</div>

