"""
@author: milad
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array

import os
from skimage.color import rgb2lab, lab2rgb, gray2rgb
import cv2
import numpy as np
#%%
# Read data and convert to array

#Size input images
SIZE = 256


batch_size = 64
datagen = ImageDataGenerator(rescale=1./255)  # to normlize all input images

from tqdm import tqdm
img_data=[]
path1 = '****' # copy and paste where your images are located ex: "lol_dataset/lol_dataset/our485/low/"
files=os.listdir(path1)
for i in tqdm(files):
    img=cv2.imread(path1+'/'+i,1)   #Change 0 to 1 for color images
    img=cv2.resize(img,(SIZE, SIZE))
    img_data.append(img_to_array(img))

    
#%%
# Read data and convert from RGB to LAB using skimage
path = '****' # copy and paste where your images are located ex: "lol_dataset/lol_dataset/our485/low/"
#Normalize images - divide by 255
train_datagen = ImageDataGenerator(rescale=1. / 255)

train = train_datagen.flow_from_directory(path, target_size=(SIZE, SIZE), batch_size=512, class_mode=None)

X =[]
Y =[]
for img in train[0]:
  try:
      lab = rgb2lab(img)
      X.append(lab[:,:,0]) 
      Y.append(lab[:,:,1:] / 128) #A and B values range from -127 to 128, 
      #so we divide the values by 128 to restrict values to between -1 and 1.
  except:
     print('error')

X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,)) #dimensions to be the same for X and Y
print(X.shape)
print(Y.shape)


# Read data and convert from RGB to LAB using openCV
img_data_LAB=[]
img_data_org = []
path1 = '****' # copy and paste where your images are located ex: "lol_dataset/lol_dataset/our485/low/"
files1=os.listdir(path1)
for i in tqdm(files1):
    img=cv2.imread(path1+'/'+i,1)   #Change 0 to 1 for color images
    img=cv2.resize(img,(SIZE, SIZE))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img[:, :, 1])
    avg_b = np.average(img[:, :, 2])
    img[:, :, 1] = img[:, :, 1] - ((avg_a - 128) * (img[:, :, 0] / 255.0) * 1.2)
    img[:, :, 2] = img[:, :, 2] - ((avg_b - 128) * (img[:, :, 0] / 255.0) * 1.2)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    img_data_LAB.append(img_to_array(img))
    

img_array_LAB = np.reshape(img_data_LAB, (len(img_data_LAB), SIZE, SIZE, 3))
img_array_LAB = img_array_LAB.astype('float32') / 255.
