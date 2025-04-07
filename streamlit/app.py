import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


st.title("Fuel economy prediction")
uploaded_file = st.file_uploader("FuelEconomy.csv", type=["csv"])

## this if block says If no file is uploaded, the condition becomes False, and the code inside the block does not run.
if uploaded_file is not None:       
    df = pd.read_csv(uploaded_file)
    st.write("dataset preview:")
    st.write(df.head())

    ## asking for X and y from user, and converting it to list . here ex: horse_power=X and fuel economy=y
    columns = df.columns.tolist()
    feature_cols = st.multiselect("Select Feature Columns (X)", columns) 
    target_col = st.selectbox("Select Target Column (y)", columns)


    if feature_cols and target_col:
       X = df[feature_cols]
       y = df[target_col]

       # Training the Linear Regression model
       model = LinearRegression()
       model.fit(X, y)

       st.write("Model trained successfully!")

       # Sidebar inputs for prediction
       st.sidebar.title("Enter Input Features for Prediction")
       input_data = []

       for col in feature_cols:
         value = st.sidebar.number_input(f"Enter {col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
         input_data.append(value)

         # Convert input list to NumPy array and reshape for prediction
         input_array = np.array(input_data).reshape(1, -1)

         # Make prediction
         if st.sidebar.button("Predict"):
           prediction = model.predict(input_array)
           st.write(f"Predicted Output: {prediction[0]}")
else:
    st.write("Please upload a dataset to proceed.")