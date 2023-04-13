import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'dataset/weed'

training_data = []
for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path,img))
    if pic is None:
        continue
    pic = cv2.resize(pic,(128,128))
    training_data.append([pic])

np.save('processed_dataset/weed',np.array(training_data))