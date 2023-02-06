from keras.models import load_model
from keras import models
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
import pickle
import imutils
from tkinter import *
from PIL import Image

from tkinter import filedialog
#loading Python Imaging Library
from PIL import ImageTk, Image

root = Tk()
#Set Title as Image Loader
root.title("Image Loader")
  
#Set the resolution of window
root.geometry("550x300")

  
#Allow Window to be resizable
root.resizable(width = True, height = True)
##import button
def load():
     with open("output.img", "r") as f:
          output=f.read()
#Load Model:-
model = load_model("Model.h5")
mlb = pickle.loads(open("mlb.pickle1", "rb").read())

# Read an Input image:-
def select_image():
    image1=filedialog.askopenfilename()
    image = cv2.imread(image1)
    #image = cv2.equalizeHist(image)
    output = imutils.resize(image,width=400)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:1]
    

    for (i, j) in enumerate(idxs):
            label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
            print(mlb.classes_[j])
            print(label)
            cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Output_image',output)
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI 
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()
