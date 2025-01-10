# grid_search.py

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st
import logging

# Логирование
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def perform_grid_search(X_train, y_train):
    # Параметры для GridSearchCV
    alpha_min = st.number_input("Минимальное значение alpha:", value=1e-4, format="%.6f")
    alpha_max = st.number_input("Максимальное значение alpha:", value=1e-2, format="%.6f")
    num_points = st.slider("Количество точек для alpha:", min_value=1, max_value=100, value=10)

    # Генерация списка значений alpha
    alpha_values = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num_points)

    # Параметры для GridSearchCV
    param_grid = {'alpha': alpha_values}

    # Запуск GridSearchCV
    grid_search = GridSearchCV(Lasso(random_state=42), param_grid, scoring='neg_mean_absolute_error', cv=4, verbose=1)
    grid_search.fit(X_train, y_train)

    # Получение лучших параметров
    best_alpha = grid_search.best_params_['alpha']
    st.success(f"Лучший параметр alpha: {best_alpha}")

    # Обучение модели с лучшим параметром
    best_model = Lasso(alpha=best_alpha, random_state=42)
    best_model.fit(X_train, y_train)
    logger.info(f"Grid Search завершен, лучший alpha: {best_alpha}")

    # Вывод коэффициентов модели
    st.write("Коэффициенты модели:")
    st.write(best_model.coef_)

    # Оценка MAPE на тренировочных данных
    mape_train = mean_absolute_percentage_error(np.exp(y_train), np.exp(best_model.predict(X_train)))
    st.write(f"MAPE на трейне = {mape_train:.4f}")

    # Логирование и вывод коэффициентов
    logger.info(f"MAPE на трейне: {mape_train:.4f}")
    
    return best_model
