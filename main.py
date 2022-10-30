import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
import cv2
import torch
import numpy as np
import os
import time ,sys
from PIL import Image, ImageEnhance
from streamlit_embedcode import github_gist
import sys


model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True) 

def main():
    title = '<p style="font-size: 42px;">Drowsiness Detection tool </p>'
    title = st.markdown(title, unsafe_allow_html=True)
    st.text("Detect drowsiness on truck drivers.\nCreated by Bacem Etteib")
    img2 = input_function()
    placeholder = st.empty()
    with placeholder.form(key="submit-form"):

        input_gender = st.selectbox(
            "Gender:", ["Male",'Female'])
        input_weather = st.selectbox(
            "Weather:", ["Clear",'Cloudy',
            "Foggy","Rainy","Severe Crosswindss",
            "Drizzling","Blowing Sand"])
        input_age = st.number_input("Select your age)", min_value=18, max_value=90, value=40, step=1)
        generate = st.form_submit_button("Submit")
        if generate:
            st.write(risk_assessment(input_gender,input_age,input_weather))
            image_classifying(model,img2)

def image_classifying(model,img2):
            results = model(img2)
            #results.print()
            img3 = np.squeeze(results.render())
            st.image(img3, caption='Proccesed Image.')

def input_function():
    st.subheader("""
    Upload a current picture of you.
    """)
    #upload image using only accepted formats
    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    if file!= None:
        #put image in the same directory as the script
        img = cv2.imread(file.name)
        #img2 = np.array(img1)
        #st.image(img1, caption = "Uploaded Image")
        #my_bar = st.progress(0)

        #st.image(image_classifying(model,img), caption='Proccesed Image.')
        
        #cv2.waitKey(0)
        
        #cv2.destroyAllWindows()
        #my_bar.progress(100)
        return img

def risk_assessment(gender,age,weather):
    X = np.array([int(age),0,0,0,0,0,0,0, 0,0])
    if str(weather) == 'Clear':
      X[2] = 1
    elif str(weather) == 'Cloudy':
      X[3] = 1
    elif str(weather) == 'Foggy':
      X[4] = 1
    elif str(weather) == 'Rainy':
      X[5] = 1
    elif str(weather) == 'Severe Crosswinds':
      X[6] = 1  
    elif str(weather) == 'Drizzling':
      X[7] = 1 
    elif str(weather) == 'Blowing Sand':
      X[1] = 1 

    if str(gender) == 'Male':
      X[9] = 1
    else:
      X[8] = 1
    print(X)
    loaded_model = pickle.load(open('finalized_model_2.sav', 'rb'))
    print(loaded_model)
    result = loaded_model.predict(np.array(X).reshape(-1,1))
    if result == 0:
        return 'Low Risk'
    else:
        return 'Medium to High Risk'

if __name__ == '__main__':
        main()  




