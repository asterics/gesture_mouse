# Gesture Mouse

A library that allows to control the mouse and keyboard with head movement and face gestures. This project is based on the 
google mediapipe library (https://ai.google.dev/edge/mediapipe/solutions/guide).

## Installation instructions
Tested with Python 3.10 and PySide6.4.3. 
1. Clone repository
2. Create a sufolder `mkdir venv`
3. Create virtual environment  `python -m venv venv`
4. Activate virtual environment `venv\Scripts\activate.bat` (Linux: `. ./venv/bin/activate`)
5. Install packages `pip install -r requirements.txt`

## Running Gesture Mouse

### Release

1. Extract zip file
2. Run `gesture_mouse.exe` (Windows) or `./guesture_mouse` (Linux)

### Development

- `python gui.py` to start gui (Linux: `sudo ./venv/bin/python3.10 gui.py`)

## Hotkeys

- <kbd>Alt</kbd>+<kbd>1</kbd>` to toggle mouse movement and gestures.
- <kbd>Alt</kbd>+<kbd>m</kbd>` to toggle mouse movement.
- <kbd>Alt</kbd>+<kbd>g</kbd>` to toggle gestures.
- <kbd>Alt</kbd>+<kbd>t</kbd>` to toggle tracking (start or stop camera).

### Hotkeys Mouse Movement

- <kbd>m</kbd> iterate through mouse (ABSOLUTE, RELATIVE, JOYSTICK, HYBRID) modes
- <kbd>t</kbd> iterate through tracking mode (PNP, MEDIAPIPE, NOSE)
- <kbd>c</kbd> Center mouse position
- <kbd>.</kbd> Switch monitor in case there are several monitors applied. Ignored for RELATIVE mouse mode.

## Creating an exe distribution
To create a distribution folder which includes all necessery .dll and an executable one can use PyInstaller([https://pyinstaller.org](https://pyinstaller.org)). 
Instructions:
1. Follow the installation instructions
2. Activate virtual environment
3. Install PyInstaller `pip install pyinstaller`
4. Execute build process with  
`pyinstaller gui.py -D --add-data config;config --add-data data;data --collect-all mediapipe -n gesture-mouse --contents-directory .` on linux  
`pyinstaller gui.py -D --add-data config:config --add-data data:data --collect-all mediapipe -n gesture-mouse --contents-directory .` on windows 

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

