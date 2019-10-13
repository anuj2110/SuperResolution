from keras.models import load_model
from Utils_model import *
import tensorflow as tf
import cv2
from PIL import Image

loss = VGG_LOSS((96,96,3))
model = load_model("gen_model3000.h5", custom_objects={'vgg_loss': loss.vgg_loss})

img  = cv2.imread("img1.jpg")
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = (img-127.5)/127.5

img = img.reshape((1,96,96,3))

image = model.predict(img)
image = (image*127.5) +127.5
cv2.imwrite("new.jpg",image.reshape((384,384,3)))
image = cv2.imread("new.jpg")
img = cv2.imread("img.jpg")
cv2.imshow("Original", img)
cv2.imshow("reconstructed", image)
cv2.waitKey(0)
