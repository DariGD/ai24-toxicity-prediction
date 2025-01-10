from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def perform_pca(X_train, X_test, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    return X_train_pca, X_test_pca, pca

def train_model(X_train_pca, X_test_pca, y_train, y_test, pca):
    lr_pca = LinearRegression()
    lr_pca.fit(X_train_pca, y_train)

    # Предсказания
    y_pred = lr_pca.predict(X_test_pca)

    # Оценка модели
    r2_train = r2_score(np.exp(y_train), np.exp(lr_pca.predict(X_train_pca)))
    r2_test = r2_score(np.exp(y_test), np.exp(y_pred))
    mape_train = mean_absolute_percentage_error(np.exp(y_train), np.exp(lr_pca.predict(X_train_pca)))
    mape_test = mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred))

    # Оценка с помощью кросс-валидации
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    mape_cv_scores = cross_val_score(lr_pca, np.exp(X_train_pca), np.exp(y_train), cv=5, scoring=mape_scorer)
    
    return lr_pca, r2_train, r2_test, mape_train, mape_test, mape_cv_scores, y_pred


def visualize_results(y_test, y_pred, cumulative_variance, X_train_pca, coefs_pca):
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))

    # График 1: Предсказанные значения и фактические
    x_values = range(100)
    axs[0].scatter(x_values, np.exp(y_test[:100]), color='blue', label='Фактические значения', alpha=0.6)
    axs[0].scatter(x_values, np.exp(y_pred[:100]), color='orange', label='Предсказанные значения', alpha=0.6)
    axs[0].plot(x_values, np.exp(y_test[:100]), color='blue', alpha=0.6)
    axs[0].plot(x_values, np.exp(y_pred[:100]), color='orange', alpha=0.6)
    axs[0].set_xlabel('Номер')
    axs[0].set_ylabel('Значение')
    axs[0].set_title('Предсказанные и фактические значения')
    axs[0].legend()
    axs[0].grid()

    # График 2: Кумулятивная доля объяснённой дисперсии
    axs[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    axs[1].set_title('Кумулятивная доля объяснённой дисперсии')
    axs[1].set_xlabel('Число главных компонент')
    axs[1].set_ylabel('Кумулятивная доля дисперсии')
    axs[1].grid()

    plt.tight_layout()
    st.plotly_chart(fig)

    st.dataframe(coefs_pca)

def run_pca_and_model(X_train, X_test, y_train, y_test):
    """Выполняет PCA, обучает модель и выводит результаты."""
    #Выполнение PCA
    X_train_pca, X_test_pca, pca = perform_pca(X_train, X_test)
    st.session_state.X_train_pca = X_train_pca
    
    #Обучение модели
    lr_pca, r2_train, r2_test, mape_train, mape_test, mape_cv_scores, y_pred = train_model(X_train_pca, X_test_pca, y_train, y_test, pca)
    
    # Визуализация результатов
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
    coefs_pca = pd.Series(lr_pca.coef_, index=X_train_pca_df.columns, name='weights').sort_values(ascending=False)

    visualize_results(y_test, y_pred, cumulative_variance, X_train_pca, coefs_pca)

    # Возвращаем модель для дальнейшего использования
    return lr_pca

def save_model(model, model_name):
    joblib.dump(model, f'{model_name}.pkl')
    st.success(f"Модель успешно сохранена как {model_name}.pkl!")

