from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px
import platform
from PIL import Image

# title
st.title('Transport classification model')

# upload
file = st.file_uploader('Picture upload', type=['png', 'jpeg', 'gif', 'svg'])

if file is not None:
    img = PILImage.create(file)

    # model
    model = load_learner('transport_model.pkl', pickle_module=pickle)

    # prediction
    prediction = model.predict(img)
    
    # display prediction
    st.success(prediction)


plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

