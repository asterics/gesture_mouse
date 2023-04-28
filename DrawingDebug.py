import mediapipe as mp
import cv2
import numpy as np
from typing import List, Optional, Tuple

import util

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

tesselation_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
contours_drawing_spec = mp_drawing.DrawingSpec((0, 255, 0), thickness=1, circle_radius=1)
iris_drawing_spec = mp_drawing.DrawingSpec((255, 0, 0), thickness=1, circle_radius=1)

def annotate_landmark_image(landmarks, image):
    annotated_image = image.copy()
    ## different connections possible (lips, eye brows, etc)

    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=landmarks,
        connections=[],
        landmark_drawing_spec=tesselation_drawing_spec,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())

    return cv2.flip(annotated_image, 1)

def draw_landmarks_fast(np_landmarks: np.ndarray, image: np.ndarray, index: Optional[List[int]]=None, color: Optional[Tuple[int, int, int]]=(255, 0, 0)):
    frame_height, frame_width, _ = image.shape
    if index is None:
        index = range(468)
    pixel_landmarks = (np_landmarks[:,:2] * np.array((frame_width, frame_height))).astype(int)
    pixel_landmarks = util.clamp_np(pixel_landmarks, np.array([1,1]), np.array([frame_width-2,frame_height-2]))
    special_landmarks = pixel_landmarks[index]
    image[special_landmarks[:,1], special_landmarks[:,0], :] = color
    image[np.maximum(special_landmarks[:,1]-1,0), special_landmarks[:,0], :] = color
    image[np.minimum(special_landmarks[:,1]+1,frame_height-1), special_landmarks[:,0], :] = color
    image[special_landmarks[:,1], np.maximum(special_landmarks[:,0]-1,0), :] = color
    image[special_landmarks[:,1], np.minimum(special_landmarks[:,0]+1,frame_width), :] = color
    return image

def show_por(x_pixel, y_pixel, width, height):
    display = np.ones((height, width, 3), np.float32)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display, '.',
                (int(x_pixel), int(y_pixel)), font, 0.5,
                (0, 0, 255), 10,
                cv2.LINE_AA)
    cv2.namedWindow("por", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("por", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('por', display)
    cv2.waitKey(1)


def show_points(points, width, height):
    display = np.ones((height, width, 3), np.float32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for point in points:
        cv2.putText(display, ".", (int(point[0]), int(point[1])), font, 0.5, (0, 0, 255), 10, cv2.LINE_AA)

    cv2.namedWindow("por", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("por", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('por', display)
    cv2.waitKey(1)