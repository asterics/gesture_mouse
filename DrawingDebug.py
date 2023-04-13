import mediapipe as mp
import cv2
import numpy as np

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