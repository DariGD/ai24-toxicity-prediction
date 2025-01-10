from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def perform_svd(X_train, X_test, n_components=600):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)
    
    return X_train_svd, X_test_svd, svd

def train_model_svd(X_train_svd, X_test_svd, y_train, y_test, svd):
    lr_svd = LinearRegression()
    lr_svd.fit(X_train_svd, y_train)

    # Предсказания
    y_train_preds = lr_svd.predict(X_train_svd)
    y_test_preds = lr_svd.predict(X_test_svd)

    # Оценка модели
    r2_train = r2_score(np.exp(y_train), np.exp(y_train_preds))
    r2_test = r2_score(np.exp(y_test), np.exp(y_test_preds))
    mape_train = mean_absolute_percentage_error(np.exp(y_train), np.exp(y_train_preds))
    mape_test = mean_absolute_percentage_error(np.exp(y_test), np.exp(y_test_preds))

    # Оценка с помощью кросс-валидации
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    mape_cv_scores = cross_val_score(lr_svd, np.exp(X_train_svd), np.exp(y_train_preds), cv=5, scoring=mape_scorer)
    
    return lr_svd, r2_train, r2_test, mape_train, mape_test, mape_cv_scores, y_test_preds

def visualize_svd_results(y_test, y_test_preds, cumulative_variance):
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    # График: Предсказанные значения против фактических значений
    x_values = range(100)
    axs.scatter(x_values, np.exp(y_test[:100]), color='blue', label='Фактические значения', alpha=0.6)
    axs.scatter(x_values, np.exp(y_test_preds[:100]), color='orange', label='Предсказанные значения', alpha=0.6)
    axs.plot(x_values, np.exp(y_test[:100]), color='blue', alpha=0.6)
    axs.plot(x_values, np.exp(y_test_preds[:100]), color='orange', alpha=0.6)
    
    axs.set_xlabel('Номер')
    axs.set_ylabel('Значение')
    axs.set_title('Предсказанные и фактические значения')
    axs.legend()
    axs.grid()

    st.plotly_chart(fig)

def run_svd_and_model(X_train, X_test, y_train, y_test):
    """Выполняет SVD, обучает модель и выводит результаты."""
    # Шаг 1: Выполнение SVD
    X_train_svd, X_test_svd, svd = perform_svd(X_train, X_test)
    st.session_state.X_train_svd = X_train_svd
    
    # Шаг 2: Обучение модели
    lr_svd, r2_train, r2_test, mape_train, mape_test, mape_cv_scores, y_test_preds = train_model_svd(X_train_svd, X_test_svd, y_train, y_test, svd)
    
    # Шаг 3: Визуализация результатов
    cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
    st.write(f"Доля объясненной дисперсии: {svd.explained_variance_ratio_.sum() * 100:.2f}%")
    
    visualize_svd_results(y_test, y_test_preds, cumulative_variance)

    # Возвращаем модель для дальнейшего использования
    return lr_svd

def save_model_svd(model, model_name):
    joblib.dump(model, f'{model_name}.pkl')
    st.success(f"Модель успешно сохранена как {model_name}.pkl!")
