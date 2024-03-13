import streamlit as st
import pandas as pd
import numpy as np
import pickle
from Predict import show_predict
from Explore import expolre_page


page = st.sidebar.selectbox("Explore or Predict", ("Predict" , "Explore"))


if page == "Predict":
    show_predict()
else:
    expolre_page()
#sda