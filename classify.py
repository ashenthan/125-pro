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
from PIL import Image
import PIL.ImageOps

if(not os.environ.get("PYTHONHTTPSVERIFY","")and 
getattr(ssl,"_create_unverified_context",None)):
    ssl._create_default_https_context=ssl._create_unverified_context
x,y=fetch_openml("mnist_784",version=1,return_X_y=True)
#print(pd.Series(y).value_counts())
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=9,train_size=7500,test_size=2500)
x_train_scale=x_train/255
x_test_scale=x_test/255
clf=LogisticRegression(solver='saga',multi_class='multinomial')
clf.fit(x_train_scale,y_train)
ypred=clf.predict(x_test_scale)
#accuracy=accuracy_score(y_test,ypred)
#print(accuracy)
#starting camera
def get_prediction(image):
    im_pil=Image.open(image)
    image_bw=im_pil.convert("L")
    image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter=20
    minpixel=np.percentile((image_bw_resized,pixel_filter))
    image_bw_resized_inverted_scale=np.clip(image_bw_resized,0,255)
    maxpixel=np.max(image_bw_resized)
    image_bw_resized_inverted_scale=np.array(image_bw_resized_inverted_scale)/maxpixel
    test_sample= np.array(image_bw_resized_inverted_scale).reshape(1,784)
    test_pred=clf.predict(test_sample)
    return test_pred[0]