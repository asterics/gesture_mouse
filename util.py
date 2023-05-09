import numpy as np
import cv2

def clamp(value, min_val, max_val):
    return min(max(value, min_val), max_val)


def clamp_np(value, min_val, max_val):
    return np.minimum(np.maximum(value, min_val), max_val)

def list_camera_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    Credits to this stackoverflow answer: https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing.
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
        camera.release()
    return available_ports,working_ports,non_working_ports

if __name__ == '__main__':
    list_camera_ports()
