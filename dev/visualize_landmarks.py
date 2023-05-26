from DrawingDebug import draw_landmarks_fast
import pandas as pd
import numpy as np
import cv2
import open3d as o3d
import time

show_3d = False

directions = ["neutral", "links", "rechts", "oben", "unten"]
direction = "bewegung"
pose = "neutral"

csv_base = "../tests/Patrick Link"
length=90

frame_height = 480
frame_width = 640

ear_names = list(map(lambda i: f"ear_{i}", range(18)))
corrected_ear_names = list(map(lambda i: f"corrected_ear_{i}", range(18)))
landmark_names = [f"landmark_{i}_{coord}" for i in range(478) for coord in ["x","y","z"]]
combined_df = pd.DataFrame()

df = pd.read_csv(f"{csv_base}/{pose}_{direction}.csv", delimiter=";")
landmarks = df[landmark_names]
print(landmarks)
np_landmarks = landmarks.to_numpy()
np_landmarks = np_landmarks.reshape(-1,478,3)
if show_3d:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    first_frame=True
    for r in np_landmarks:
        if first_frame:
            geometry = o3d.geometry.PointCloud()
            geometry.points = o3d.utility.Vector3dVector(r)
            vis.add_geometry(geometry)
            first_frame=False
        else:
            geometry.points = o3d.utility.Vector3dVector(r)
            vis.update_geometry(geometry)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.03)
else:
    for r in np_landmarks:
        black = np.zeros((frame_height, frame_width, 3)).astype(np.uint8)
        image = draw_landmarks_fast(r, black, index=list(range(478)))
        cv2.imshow("Landmarks", image)
        cv2.waitKey(15)
