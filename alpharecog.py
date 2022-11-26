import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os,ssl,time

x=np.load("image.npz")["arr_0"]
y=pt.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=9,train_size=7500,test_size=2500)
x_train_scale=x_train/255
x_test_scale=x_test/255
clf=LogisticRegression(solver='saga',multi_class='multinomial')
clf.fit(x_train_scale,y_train)
ypred=clf.predict(x_test_scale)
accuracy=accuracy_score(y_test,ypred)
print(accuracy)
#starting camera
cam=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upperleft=(int(width/2-56),int(height/2-56))
        bottomright=(int(width/2+56),int(height/2+56))
        cv2.rectangle=gray,upperleft,bottomright,(0,255,0),2
        roi = gray[upper_left[1]:bottomright[1], upperleft[0],bottomright[0]]
        im_pil=Image.fromarray
        image_bw=im_pil.convert("L")
        image_bw=resize((28,28),Image.ANTIALIAS)
        image_bw_resized_inverted= PIL.ImageOps.invert(image_bw_resized)
        pixel_filter=20
        minpixel=np.percentile((image_bw_resized,pixel_filter))
        image_bw_resized_inverted_scale=np.clip(image_bw_resized_inverted,0,255)
        maxpixel=np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scale=np.array(image_bw_resized_inverted_scale)/maxpixel
        test_sample= np.array(image_bw_resized_inverted_scale).reshape(1,78)
        test_pred=clf.predict(test_sample)
        print("predicted classes",test_pred)
        cv2.im_show("frame",gray)
        if(cv2.waitKey(1)& 0xFF==ord("q")):
             break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()