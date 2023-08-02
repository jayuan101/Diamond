# -*- coding: utf-8 -*-

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

pip install streamlit

import xgboost as xgb
import streamlit as st
import pandas as pd

#Loading up the Regression model we created
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

# Define the prediction function
def predict(carat, cut, color, clarity, depth, table, x, y, z):
    #Predicting the price of the carat
    if cut == 'Fair':
        cut = 0
    elif cut == 'Good':
        cut = 1
    elif cut == 'Very Good':
        cut = 2
    elif cut == 'Premium':
        cut = 3
    elif cut == 'Ideal':
        cut = 4

    if color == 'J':
        color = 0
    elif color == 'I':
        color = 1
    elif color == 'H':
        color = 2
    elif color == 'G':
        color = 3
    elif color == 'F':
        color = 4
    elif color == 'E':
        color = 5
    elif color == 'D':
        color = 6

    if clarity == 'I1':
        clarity = 0
    elif clarity == 'SI2':
        clarity = 1
    elif clarity == 'SI1':
        clarity = 2
    elif clarity == 'VS2':
        clarity = 3
    elif clarity == 'VS1':
        clarity = 4
    elif clarity == 'VVS2':
        clarity = 5
    elif clarity == 'VVS1':
        clarity = 6
    elif clarity == 'IF':
        clarity = 7

prediction = model.predict(pd.DataFrame([[carat,cut,color,clarity,depth,table,x,y,z]], columns=['carat', 'cut','color','clarity','depth','table','x','y','z']))
return prediction

st.title('Diamond Price Predictor')
#st.image("""https://img1.picmix.com/output/stamp/normal/0/4/3/8/1568340_36175.gif""")
st.markdown("I was looking ta dimonds to buy online and I went on Google and searched up its prices, but I didn’t know what metrics drove those prices. Therefore, I decided to apply some machine learning techniques to figure out what drives the price of a flawless diamond ring!")

st.image("""https://giffiles.alphacoders.com/143/14379.gif""")
st.markdown("The data we will be using for this project is the Diamonds dataset, which is publicly available via Kaggle. It contains 53940 observations, and 10 features in the dataset. I performed data preprocessing transformations, and built a regression model to predict the price ($326-$18,823) of the diamond using basic diamond measurement metrics. Each diamond in this dataset is given a price. The price of the diamond is determined by 7 input variables:")
st.write("Carat Weight: 0.2Kg - 5.01Kg  \nCut: Fair, Good, Very Good, Premium, Ideal  \nColor: from J (Worst) to D (Best)  \nClarity: I1 (Worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (Best)  \nPolish: ID (Ideal), EX (Excellent), G (Good), VG (Very Good)  \nSymmetry: ID (Ideal), EX (Excellent), G (Good), VG (Very Good)  \nReport: AGSL (American Gem Society Laboratories), GIA (Gemological Institute of America)")

with st.sidebar:
    st.header('Enter the characteristics of the diamond:')
    carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0)
    cut = st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.selectbox('Color Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
    clarity = st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.number_input('Diamond Depth Percentage:', min_value=0.1, max_value=100.0, value=1.0)
    table = st.number_input('Diamond Table Percentage:', min_value=0.1, max_value=100.0, value=1.0)
    x = st.number_input('Diamond Length (X) in mm:', min_value=0.1, max_value=100.0, value=1.0)
    y = st.number_input('Diamond Width (Y) in mm:', min_value=0.1, max_value=100.0, value=1.0)
    z = st.number_input('Diamond Height (Z) in mm:', min_value=0.1, max_value=100.0, value=1.0)

if st.button('Predict Price'):
    price = predict(carat, cut, color, clarity, depth, table, x, y, z)
    st.success(f'The predicted price of the diamond is ${price[0]:.2f} USD')



