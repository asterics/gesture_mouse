import matplotlib.pyplot as plt

from DrawingDebug import draw_landmarks_fast
import SignalsCalculator
import pandas as pd
import numpy as np
import cv2
import time

from dev.gesture_capture.calculate_normal_area import canonical_metric_landmarks

show_3d = False

directions = ["bewegt"]

pose = "neutral"

csv_base = "../tests/Patrick Link"
length=90

frame_height = 480
frame_width = 640

camera_parameters = (480, 480, 640 / 2, 480 / 2)
camera_matrix = np.array([
            [camera_parameters[0],0,camera_parameters[2],0],
            [0,camera_parameters[1],camera_parameters[3],0],
            [0,0,1,0]
        ])



signal_calculator = SignalsCalculator.SignalsCalculater(camera_parameters,(frame_width,frame_height))

ear_names = list(map(lambda i: f"ear_{i}", range(18)))
corrected_ear_names = list(map(lambda i: f"corrected_ear_{i}", range(18)))
landmark_names = [f"landmark_{i}_{coord}" for i in range(478) for coord in ["x","y","z"]]
facial_transformation_names = [f"transformation_matrix_{i}" for i in range(16)]
combined_df = pd.DataFrame()

# combine dataframes

combined_df = pd.DataFrame()
for direction in directions:
    df = pd.read_csv(f"{csv_base}/{pose}_{direction}.csv", delimiter=";")
    df["direction"]=direction
    combined_df = pd.concat([combined_df,df])
    #ear_values = df[ear_names]
    #corrected_ear_values = df[ear_names]
    #print(f"{direction}: {ear_values.mean()}")
    #print(corrected_ear_values.describe())
combined_df["direction"] =combined_df["direction"].astype("category")

landmarks = combined_df[landmark_names]
np_landmarks = landmarks.to_numpy()
np_landmarks = np_landmarks.reshape(-1,478,3)
used_landmarks = np_landmarks[:,:468,:2]*np.array([frame_width,frame_height])

facial_transformation_matrix=combined_df[facial_transformation_names].to_numpy().reshape(-1,4,4)
#debuf

ear_batch = np.array([signal_calculator.eye_aspect_ratio_batch(landmark, signal_calculator.ear_indices) for landmark in used_landmarks])
ear_reference = signal_calculator.eye_aspect_ratio_batch(canonical_metric_landmarks,signal_calculator.ear_indices)
p_hom = np.ones((468,4))
p_hom[:,:3]=canonical_metric_landmarks

camera_p =  np.matmul(camera_matrix@facial_transformation_matrix, p_hom.T).swapaxes(1,2)
projected_p = camera_p[:,:,:2]/camera_p[:,:,[2]]
# orig
# d1 = np.ones((18,4))
# d2 = np.ones((18,4))
# d3 = np.ones((18,4))
#
# d1[:,:3] = signal_calculator.ear_indices[:, 6:9]
# d2[:,:3] = signal_calculator.ear_indices[:, 9:12]
# d3[:,:3] = signal_calculator.ear_indices[:, 12:15]
#
# rotated_d1 = np.matmul(camera_matrix@facial_transformation_matrix, d1.T).swapaxes(1,2)
# rotated_d2 = np.matmul(camera_matrix@facial_transformation_matrix, d2.T).swapaxes(1,2)
# rotated_d3 = np.matmul(camera_matrix@facial_transformation_matrix, d3.T).swapaxes(1,2)
#
# projected_d1 = rotated_d1[:,:,:2]/(rotated_d1[:,:,[2]])
# projected_d2 = rotated_d2[:,:,:2]/(rotated_d2[:,:,[2]])
# projected_d3 = rotated_d3[:,:,:2]/(rotated_d3[:,:,[2]])


correction_factor = 1/np.array([signal_calculator.eye_aspect_ratio_batch(landmark, signal_calculator.ear_indices) for landmark in projected_p])

ear_corrected = ear_batch*correction_factor
ear_batch = ear_batch
#plt.plot(correction_factor[:,1])
#plt.plot(np.linalg.norm(projected_d3,axis=2)[:,1])
fig_line, axs_line = plt.subplots(6,3, figsize=(15,15))

axs_line = axs_line.flatten()
for i in range(17):
    #axs_line[i].plot(correction_factor[:,i],label=f"corr_{i}")
    axs_line[i].plot(ear_batch[:,i]/ear_batch[:,i].mean(),label=f"ear_{i}")
    axs_line[i].plot(ear_corrected[:,i]/ear_corrected[:,i].mean(),label=f"ear_cor_{i}")
    axs_line[i].legend()
plt.show()

