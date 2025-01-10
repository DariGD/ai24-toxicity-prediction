from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def select_kbest(X_train, y_train, X_test, k=500):
    """Проводит отбор признаков с помощью SelectKBest и возвращает преобразованные данные."""
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = selector.get_support(indices=True)
    
    return X_train_selected, X_test_selected, selector

def train_model_kbest(X_train_selected, X_test_selected, y_train, y_test, selector):
    """Обучает модель линейной регрессии на отобранных признаках и оценивает ее результаты."""
    lr_kbest = LinearRegression()
    lr_kbest.fit(X_train_selected, y_train)

    # Предсказания
    y_pred = lr_kbest.predict(X_test_selected)

    # Оценка модели
    r2_train = r2_score(np.exp(y_train), np.exp(lr_kbest.predict(X_train_selected)))
    r2_test = r2_score(np.exp(y_test), np.exp(lr_kbest.predict(X_test_selected)))
    mape_train = mean_absolute_percentage_error(np.exp(y_train), np.exp(lr_kbest.predict(X_train_selected)))
    mape_test = mean_absolute_percentage_error(np.exp(y_test), np.exp(lr_kbest.predict(X_test_selected)))

    # Оценка с помощью кросс-валидации
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    mape_cv_scores = cross_val_score(lr_kbest, np.exp(X_train_selected), np.exp(lr_kbest.predict(X_train_selected)), cv=5, scoring=mape_scorer)
    
    return lr_kbest, r2_train, r2_test, mape_train, mape_test, mape_cv_scores, y_pred

def visualize_kbest_results(y_test, y_pred, selected_features):
    """Визуализирует результаты после отбора признаков SelectKBest и линейной регрессии."""
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    # График: Предсказанные значения против фактических значений
    x_values = range(100)
    axs.scatter(x_values, np.exp(y_test)[:100], color='blue', label='Фактические значения', alpha=0.6)
    axs.scatter(x_values, np.exp(y_pred)[:100], color='orange', label='Предсказанные значения', alpha=0.6)
    axs.plot(x_values, np.exp(y_test)[:100], color='blue', alpha=0.6)
    axs.plot(x_values, np.exp(y_pred[:100]), color='orange', alpha=0.6)
    
    axs.set_xlabel('Номер')
    axs.set_ylabel('Значение')
    axs.set_title('Предсказанные и фактические значения')
    axs.legend()
    axs.grid()

    st.plotly_chart(fig)

def run_select_kbest_and_model(X_train, X_test, y_train, y_test):
    """Выполняет SelectKBest, обучает модель и выводит результаты."""
    # Шаг 1: Выполнение SelectKBest
    X_train_selected, X_test_selected, selector = select_kbest(X_train, y_train, X_test)
    st.session_state.X_train_selected = X_train_selected
    
    # Шаг 2: Обучение модели
    lr_kbest, r2_train, r2_test, mape_train, mape_test, mape_cv_scores, y_pred = train_model_kbest(X_train_selected, X_test_selected, y_train, y_test, selector)
    
    # Шаг 3: Визуализация результатов
    st.write(f"Количество выбранных признаков: {len(selector.get_support(indices=True))}")
    visualize_kbest_results(y_test, y_pred, selector)

    # Возвращаем модель для дальнейшего использования
    return lr_kbest

def save_model_skb(model, model_name):
    """Сохраняет модель в файл."""
    joblib.dump(model, f'{model_name}.pkl')
    st.success(f"Модель успешно сохранена как {model_name}.pkl!")
