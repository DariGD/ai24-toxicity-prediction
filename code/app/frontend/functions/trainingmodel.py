import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error

def train_model(X_train, y_train, X_test, y_test): 
    lr = LinearRegression()  
    lr.fit(X_train, y_train)  
    
    y_pred = lr.predict(X_test)  
    r2_train = r2_score(np.exp(y_train), np.exp(lr.predict(X_train)))  
    r2_test = r2_score(np.exp(y_test), np.exp(y_pred))  
    
    mape_train = mean_absolute_percentage_error(np.exp(y_train), np.exp(lr.predict(X_train)))  
    mape_test = mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred))  
    
    return r2_train, r2_test, mape_train, mape_test, lr.coef_, X_train.columns

def display_model_results(X_train, y_train, X_test, y_test):
    r2_train, r2_test, mape_train, mape_test, coefs, feature_names = train_model(X_train, y_train, X_test, y_test) 
    series_with_coefs = pd.Series(coefs, index=feature_names, name='weights').sort_values(ascending=False).head(10) 
    
    st.success(f"Обучение завершено!") 
    st.markdown(
        f"""
        <div style="background-color:rgb(242, 224, 224); padding: 5px; border-radius: 5px;">
            <strong>R² на трейне:</strong> {r2_train:.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(242, 224, 224); padding: 5px; border-radius: 5px;">
            <strong>R² на тесте:</strong> {r2_test:.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(242, 224, 224); padding: 5px; border-radius: 5px;">
            <strong>MAPE на трейне:</strong> {mape_train:.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(242, 224, 224); padding: 5px; border-radius: 5px;">
            <strong>MAPE на тесте:</strong> {mape_test:.4f}
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown("Коэффициенты модели")
    st.dataframe(series_with_coefs)
