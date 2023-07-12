# Gesture Mouse

A library that allows to control the mouse and keyboard with head movement and face gestures. This project is based on the 
google mediapipe library (https://google.github.io/mediapipe/).

## Installation instructions
Tested with Python 3.10 and PySide6.4.3. 
1. Create virtual environment  `python venv venv`
2. Activate virtual environment `venv/Scripts/activate`
3. Install packages `pip install -r requirements.txt`

## Running Gesture Mouse
- `python gui.py` to start gui
- `Alt+1` to toggle mouse controlled by python or system.
- `Esc` to turn off program. (Used if you lose control over mouse)

**Note**: Under Linux the package keyboard needs root permissions, so run it like this:

```sudo ./venv/bin/python3.10 gui.py```

## Creating an exe distribution
To create a distribution folder wich includes all necessery .dll and an executable one can use PyInstaller([https://nuitka.net/](https://pyinstaller.org)). 
Instructions:
1. Activate virtual environment `venv/Scripts/activate` on windows and `source venv/bin/activate` on linux
2. Install PyInstaller `pip install pyinstaller`
3. Execute build process with  
`pyinstaller gui.py -D --add-data config;config --add-data data;data --collect-all mediapipe`
