import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_percentage_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import impute
from sklearn import preprocessing
import warnings
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import TruncatedSVD
import joblib
import plotly.express as px
import logging
import os
import streamlit as st
from logging.handlers import RotatingFileHandler
import requests
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename



log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Настройка логирования
log_file = os.path.join(log_directory, "app.log")
logger = logging.getLogger("StreamlitApp")
logger.setLevel(logging.DEBUG)

# Создаем обработчик с ротацией
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)  # 5 MB на файл, 5 резервных копий
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

# Заголовок приложения
st.title("Определение токсичности химических веществ")
st.subheader("Посмотрим на исходный датасет")
st.text("Мы собрали данные с разных источников по токсичности химических веществ. Определили оптимальной метрикой токсичности параметр LD50. Для каждого SMILES (уникальный идентификатор вещества) рассчитали дескрипторы. Сводный датасет, с которым работали на данном этапе предстваляем")


@ st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

st.subheader("Загрузка CSV файла")
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

df = load_data(uploaded_file)
num_rows = st.number_input("Введите количество строк для отображения:", min_value=1, max_value=len(df), value=5)

if st.button("Показать загруженный датасет"): 
            st.write(f"Загруженный датасет (первые {num_rows} строк):")
            logger.info("Датасет загружен и показан!")
            st.dataframe(df.head(num_rows))

st.subheader("Предобработка")
st.text("Избавляемся от дубликатов, оставляем для предсказания только таргет LD50 ")
if (st.checkbox('Сделаем предобработку датасета')):
    df = df.drop_duplicates(subset = ['Smiles', 'Exp. Animal', 'Method of administration'])
    cat_features_mask = (df.dtypes == "object").values
    obj_cols = df.select_dtypes('object').columns
    df[obj_cols] = df[obj_cols].astype('category')
    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
    df = df.drop(['Source', 'Smiles',
        'LD50 (a.u.)', 'Carcinogenicity', 'Carcinogenicity (a.u.)',
        'Hepatoxicity', 'Hepatoxicity (a.u.)', 'Eye_Corrosion',
        'Eye_Corrosion (a.u.)', 'Eye_Irritation', 'Eye_Irritation (a.u.)',
        'Mutagenicity', 'Mutagenicity (a.u.)', 'Respiratory_Toxicity',
        'Respiratory_Toxicity (a.u.)', 'LC50', 'LC50 (a.u.)', 'NOAEL',
        'NOAEL (a.u)'], axis = 1).reset_index(drop = True)

    df = df.dropna(subset=['LD50'])
    Q10 = df['LD50'].quantile(0.10)
    Q90 = df['LD50'].quantile(0.90)
    IQR = Q90 - Q10

    lower_bound = Q10 - 1.5 * IQR
    upper_bound = Q90 + 1.5 * IQR

    df = df[(df['LD50'] >= lower_bound) & (df['LD50'] <= upper_bound)]
    df = df[df['LD50'] != 0]
    logger.info("Датасет успешно предобработан")
    st.success("Датасет успешно предобработан ✅")

st.text("Мы хотим убедиться, что у нас нет признаков, которые были бы статичные во всем датасете, чтобы мимнимзировать риск переобучения на них, поэтому сразу удаляем их ")
st.text("Удаляем столбцы, где 99.6% значений — это одно уникальное значение")

threshold = st.number_input("Введите порог для удаления столбцов (у нас используется 99.6%):", 
                                min_value=0.98, max_value=0.999, value=0.996, step=0.001)

if 'show_ld50' not in st.session_state:
        st.session_state.show_ld50 = False
if 'show_log_ld50' not in st.session_state:
        st.session_state.show_log_ld50 = False
if 'fit_1' not in st.session_state:
        st.session_state.fit_1 = False
if 'delete' not in st.session_state:
        st.session_state.delete = False
if 'pca' not in st.session_state:
        st.session_state.pca = False
if 'svd' not in st.session_state:
        st.session_state.svd = False
if 'skb' not in st.session_state:
        st.session_state.skb = False             

# Функция для удаления столбцов с постоянными значениями
def drop_constant_columns(df, threshold):
    to_drop = []  # Список столбцов для удаления
    for col in df.columns:
        # Рассчитываем долю самого частого значения
        max_freq = df[col].value_counts(normalize=True).max()
        if max_freq >= threshold:
            to_drop.append(col)
    # Удаляем столбцы
    return df.drop(columns=to_drop)

if st.button("Удалить признаки с постоянными значениями"):
        st.session_state.delete = True

if st.session_state.delete:     
        df_cleaned = drop_constant_columns(df, threshold) 
        logger.info("Предварительно почищены признаки")
        st.write(f"Данные очищены. Осталось столбцов: {df_cleaned.shape[1]}") 
        st.dataframe(df_cleaned)
else:
        logger.warning("Нельзя перейти на этот этап.Сначала загрузите и предобработайте датасет.")
        st.warning("Сначала загрузите и предобработайте датасет.")


st.subheader("Анализ данных LD50")
df = df_cleaned
df = df.dropna()

    # Создание нового столбца
df['log_LD50'] = np.log(df['LD50'])

    # Инициализация состояния
if 'show_ld50' not in st.session_state:
        st.session_state.show_ld50 = False
if 'show_log_ld50' not in st.session_state:
        st.session_state.show_log_ld50 = False
if 'fit_1' not in st.session_state:
        st.session_state.fit_1 = False

    # Кнопка для отображения графика LD50
if st.button("Показать график LD50"):
        st.session_state.show_ld50 = True

    # Кнопка для отображения графика log_LD50
if st.button("Показать график log_LD50"):
        st.session_state.show_log_ld50 = True

    # Отображение графиков в зависимости от состояния
if st.session_state.show_ld50:
    plt.figure(figsize=(10, 5)) 
    df['LD50'].hist(bins=30, color='blue', alpha=0.7) 
    plt.title('Гистограмма LD50') 
    plt.xlabel('LD50') 
    plt.ylabel('Частота') 
    st.pyplot(plt)
    logger.info("Показан график")

if st.session_state.show_log_ld50:
    plt.figure(figsize=(10, 5)) 
    df['log_LD50'].hist(bins=30, color='green', alpha=0.7) 
    plt.title('Гистограмма LD50') 
    plt.xlabel('LD50') 
    plt.ylabel('Частота') 
    st.pyplot(plt)
    logger.info("Показан график")

X = df.drop(columns=['LD50', 'log_LD50'])  
y = df['log_LD50']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=123)  
    
num_col = X.select_dtypes(include=['number']).columns.tolist()  
cat_col = X.select_dtypes(include=['category']).columns.tolist()  
    
X_train_cat = X_train[cat_col]  
X_train_num = X_train[num_col]  
    
X_test_cat = X_test[cat_col]  
X_test_num = X_test[num_col]  
    
OHE = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')  
    
X_train_encoded = OHE.fit_transform(X_train_cat)  
X_test_encoded = OHE.transform(X_test_cat)  
    
ohe_columns = OHE.get_feature_names_out(cat_col)  
X_train_cat = pd.DataFrame(X_train_encoded, columns=ohe_columns, index=X_train.index)  
X_test_cat = pd.DataFrame(X_test_encoded, columns=ohe_columns, index=X_test.index)  
    
scaler = MinMaxScaler()  
X_train_scaled = scaler.fit_transform(X_train[num_col])  
X_test_scaled = scaler.transform(X_test[num_col])  
    
X_train_scaled = pd.DataFrame(X_train_scaled, columns=num_col, index=X_train.index)  
X_test_scaled = pd.DataFrame(X_test_scaled, columns=num_col, index=X_test.index)  
X_train = pd.concat([X_train_cat, X_train_scaled], axis=1)  
X_test = pd.concat([X_test_cat, X_test_scaled], axis=1)

@ st.cache_data
def train_model(X_train, y_train, X_test, y_test): 
    lr = LinearRegression()  
    lr.fit(X_train, y_train)  
    
    y_pred = lr.predict(X_test)  
    r2_train = r2_score(np.exp(y_train), np.exp(lr.predict(X_train)))  
    r2_test = r2_score(np.exp(y_test), np.exp(y_pred))  
    
    mape_train = mean_absolute_percentage_error(np.exp(y_train), np.exp(lr.predict(X_train)))  
    mape_test = mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred))  
    
    return r2_train, r2_test, mape_train, mape_test, lr.coef_, X_train.columns

st.subheader("Обучение модели на всех признаках")

if st.button("Обучить модель"):
        st.session_state.fit_1 = True 

if st.session_state.fit_1: 
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
    logger.info("Модель обучена")

# Отображение таблицы с коэффициентами
st.markdown("Коэффициенты модели")
st.dataframe(series_with_coefs)

st.text("Исходя из различий метрик на трейне и тесте, а также полученным весам модели, предполагаем, что модель переобучилась. Необходимо провести отбор признаков для повышения точности модели.")
st.text("Попробуем применить различные методы для уменьшения количества переменных в наборе данных")
st.title("Отбор признаков")

pca = PCA(n_components=0.95)  # 95% объяснённой дисперсии
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

@ st.cache_data
def run_pca_and_model(X_train_pca, X_test_pca, y_train, y_test):
    # Проверка размерности
    st.success(f"PCA проведено. Линейная модель обучилась")
    st.write("Размерность после PCA:")
    st.write(f"X_train_pca: {X_train_pca.shape}, X_test_pca: {X_test_pca.shape}")

    # Обучение модели линейной регрессии
    lr_pca = LinearRegression()
    lr_pca.fit(X_train_pca, y_train)

    # Предсказания
    y_pred = lr_pca.predict(X_test_pca)

    # Оценка модели
    r2_train = r2_score(np.exp(y_train), np.exp(lr_pca.predict(X_train_pca)))
    r2_test = r2_score(np.exp(y_test), np.exp(y_pred))
    mape_train = mean_absolute_percentage_error(np.exp(y_train), np.exp(lr_pca.predict(X_train_pca)))
    mape_test = mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred))
    
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    mape_cv_scores = cross_val_score(lr_pca, np.exp(X_train_pca), np.exp(y_train), cv=5, scoring=mape_scorer)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
    coefs_pca = pd.Series(lr_pca.coef_, index=X_train_pca_df.columns, name='weights').sort_values(ascending=False)

    st.markdown(
        f"""
        <div style="background-color:rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>R² на трейне:</strong> {r2_score(np.exp(y_train), np.exp(lr_pca.predict(X_train_pca)))}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>R² на тесте:</strong> {r2_score(np.exp(y_test), np.exp(y_pred)):.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>MAPE на трейне:</strong> {mean_absolute_percentage_error(np.exp(y_train), np.exp(lr_pca.predict(X_train_pca))):.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>MAPE на тесте:</strong> {mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred)):.4f}
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>MAPE на кросс-валидации:</strong> {-mape_cv_scores.mean():.2f} ± {mape_cv_scores.std():.2f}
        </div>
        """, unsafe_allow_html=True
    )


    fig, axs = plt.subplots(2, 1, figsize=(10, 5))

# График 1: Предсказанные значения против фактических значений
    x_values = range(100)  
    axs[0].scatter(x_values, np.exp(y_test[:100]), color='blue', label='Фактические значения', alpha=0.6)
    axs[0].scatter(x_values, np.exp(y_pred[:100]), color='orange', label='Предсказанные значения', alpha=0.6)
    
    # Соединяем линии
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

    st.plotly_chart(fig)
    st.dataframe(coefs_pca)
    return lr_pca
        
st.subheader("PCA и линейная регрессия")

@ st.cache_data
def save_model(model, model_name):
    joblib.dump(model, f'{model_name}.pkl')
    st.success(f"Модель успешно сохранена как {model_name}.pkl!")

if st.button("PCA и линейная регрессия"):
        st.session_state.pca = True 

if st.session_state.pca:
    lr_pca_model=run_pca_and_model(X_train_pca, X_test_pca, y_train, y_test)
    logger.info("Проведено PCA и обучена модель линейной регрессии")
    st.markdown("**Видим, что коэффициенты модели уменьшились. Наш датасет с признаками представляет собой разреженную матрицу, можно попробовать адаптировать PCA как раз для модели разреженных данных.**")

    if st.checkbox("Сохранить модель"):
        model_name = st.text_input("Введите имя файла для сохранения модели (без расширения):", "linear_regression_model")
        save_model(lr_pca_model, model_name)
        logger.info("Модель успешно сохранена")


svd = TruncatedSVD(n_components=600, random_state=42)  # Сохраняем 600 главных компонент
X_train_pca_upgrade = svd.fit_transform(X_train)          # Применяем SVD на тренировочные данные
X_test_pca_upgrade = svd.transform(X_test)                # Применяем SVD на тестовые данные
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

@ st.cache_data
def svd_(X_train_pca_upgrade, y_train, X_test_pca_upgrade):
    # Обучение линейной регрессии на данных уменьшенной размерности
    lr_pca_upgrade = LinearRegression()
    lr_pca_upgrade.fit(X_train_pca_upgrade, y_train)
    # === Оценка модели ===
    y_train_preds = lr_pca_upgrade.predict(X_train_pca_upgrade)  # Предсказания для тренировки
    y_test_preds = lr_pca_upgrade.predict(X_test_pca_upgrade)    # Предсказания для теста
    mape_pca_upgrade = cross_val_score(lr_pca_upgrade, np.exp(X_train_pca_upgrade), np.exp(y_train_preds), cv=5, scoring=mape_scorer)

    st.write(f"Размер X_train до SVD: {X_train.shape}")
    st.write(f"Размер X_train после SVD: {X_train_pca_upgrade.shape}")
    st.write(f"Размер X_test после SVD: {X_test_pca_upgrade.shape}")
    st.markdown(
        f"""
        <div style="background-color:rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>R² на трейне:</strong> {r2_score(np.exp(y_train), np.exp(y_train_preds)):.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>R² на тесте:</strong> {r2_score(np.exp(y_test), np.exp(y_test_preds)):.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>MAPE на трейне:</strong> {mean_absolute_percentage_error(np.exp(y_train), np.exp(y_train_preds)):.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204; padding: 5px; border-radius: 5px;">
            <strong>MAPE на тесте:</strong> {mean_absolute_percentage_error(np.exp(y_test), np.exp(y_test_preds)):.4f}
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>MAPE на кросс-валидации:</strong> {-mape_pca_upgrade.mean():.2f} ± {mape_pca_upgrade.std():.2f}
        </div>
        """, unsafe_allow_html=True
    )
    st.write(f"Доля объясненной дисперсии: {svd.explained_variance_ratio_.sum() * 100:.2f}%")

    fig, axs = plt.subplots(1, 1, figsize=(10,5))
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

    return lr_pca_upgrade

st.subheader("SVD")
if st.button("SVD"):
    st.session_state.svd = True 

if st.session_state.svd:
    svd_model = svd_(X_train_pca_upgrade, y_train, X_test_pca_upgrade)
    logger.info("Проведено SVD и обучена модель линейной регрессии")
    if st.checkbox("Сохранить модель SVD"):
        model_name = st.text_input("Введите имя файла для сохранения модели (без расширения):", "svd_model")
        save_model(svd_model, model_name)
        logger.info("Модель успешно сохранена")


selector = SelectKBest(score_func=f_regression, k=500)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = selector.get_support(indices=True)

st.subheader("SelectKBest")

@ st.cache_data
def select_(X_train_selected, y_train, X_test_selected):
    lr_kbest = LinearRegression()
    lr_kbest.fit(X_train_selected, y_train)
    y_pred = lr_kbest.predict(X_test_selected)
    mape_kbest = cross_val_score(lr_kbest, np.exp(X_train_selected), np.exp(lr_kbest.predict(X_train_selected)), cv=5, scoring=mape_scorer)

    st.markdown(
        f"""
        <div style="background-color:rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>R² на трейне:</strong> {r2_score(np.exp(y_train), np.exp(lr_kbest.predict(X_train_selected))):.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>R² на тесте:</strong> {r2_score(np.exp(y_test), np.exp(lr_kbest.predict(X_test_selected))):.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>MAPE на трейне:</strong> {mean_absolute_percentage_error(np.exp(y_train), np.exp(lr_kbest.predict(X_train_selected))):.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204; padding: 5px; border-radius: 5px;">
            <strong>MAPE на тесте:</strong> {mean_absolute_percentage_error(np.exp(y_test), np.exp(lr_kbest.predict(X_test_selected))):.4f}
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div style="background-color: rgb(204, 255, 204); padding: 5px; border-radius: 5px;">
            <strong>MAPE на кросс-валидации::</strong> {-mape_kbest.mean():.2f} ± {mape_kbest.std():.2f}
        </div>
        """, unsafe_allow_html=True
    )
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
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

    return lr_kbest

if st.button("SelectKBest"):
    st.session_state.skb = True 

if st.session_state.skb:
    SKB_model = select_(X_train_selected, y_train, X_test_selected)
    logger.info("Проведен отбор признаков по методу SelectKBest и обучена модель линейной регрессии")
    if st.checkbox("Сохранить модель SKB"):
        model_name = st.text_input("Введите имя файла для сохранения модели (без расширения):", "SelectKbest_model")
        save_model(SKB_model, model_name)
        logger.info("Модель успешно сохранена")


st.title("Подбор гиперпараметров")
st.subheader("Grid Search для Lasso Регрессии")

X_train_options = {
    'Без сжатия признакового пространства': X_train,  
    'После PCA': X_train_pca,
    'После SKB': X_train_selected,
    'После SVD':X_train_pca_upgrade 
}
st.text("Здесь мы можем выбрать на каком наборе данных будем проводить GS")
# Выбор параметра X_train из списка
selected_option = st.selectbox("Выберите вариант X_train:", list(X_train_options.keys()))

# Получаем выбранные данные
X_train_pca = X_train_options[selected_option]


# Параметры для GridSearchCV
alpha_min = st.number_input("Минимальное значение alpha:", value=1e-4, format="%.6f")
alpha_max = st.number_input("Максимальное значение alpha:", value=1e-2, format="%.6f")
num_points = st.slider("Количество точек для alpha:", min_value=1, max_value=100, value=10)

# Генерация списка значений alpha
alpha_values = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num_points)

# Параметры для GridSearchCV
param_grid = {'alpha': alpha_values}

# Запуск GridSearchCV
if st.button("Запустить Grid Search"):
    with st.spinner("Поиск лучших параметров..."):
        grid_search = GridSearchCV(Lasso(random_state=42), param_grid, scoring='neg_mean_absolute_error', cv=4, verbose=1)
        grid_search.fit(X_train_pca, y_train)
        
        # Получение лучших параметров
        best_alpha = grid_search.best_params_['alpha']
        st.success(f"Лучший параметр alpha: {best_alpha}")

        # Обучение модели с лучшим параметром
        best_model = Lasso(alpha=best_alpha, random_state=42)
        best_model.fit(X_train_pca, y_train)
        logger.info("GS закончил свою работу по подбору гиперпарметров")
        # Вывод коэффициентов модели
        st.write("Коэффициенты модели:")
        st.write(best_model.coef_)
        st.write(f"MAPE на трейне = {mean_absolute_percentage_error(np.exp(y_train), np.exp(best_model.predict(X_train_pca))):.4f}")


st.title("Модель предсказания LD50")
st.text("Теперь покажем, как наша модель работает. Для этого предварительно запускаем сервер с FastAPI")
API_URL = "http://127.0.0.1:8000"
# Загрузка CSV файла для обучения модели
st.header("Обучение модели")
uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

if uploaded_file is not None:
    if st.button("Обучение модели"):
        files = {"file": uploaded_file}
        response = requests.post(f"{API_URL}/fit", files=files)
        st.success(response.json().get("response"))

# Ввод данных для предсказания
st.header("Предсказание")
smiles_input = st.text_input("Введите SMILES:")
exp_animal = st.selectbox("Выберите экспериментальное животное:", ["mouse", "rat"])
method_of_administration = st.selectbox("Выберите метод введения:", ['intraperitoneal', 'oral', 'intravenous', 'intramusculary', 'skin',
                                    'inhalation', 'percutaneous', 'sc', 'subcutaneous', 'gavage',
                                    'intramuscular', 'diet'])

if st.button("Сделать предсказание"):
    input_data = {
        "Smiles": smiles_input,
        "Exp_animal": exp_animal,
        "Method_of_administration": method_of_administration
    }
    response = requests.post(f"{API_URL}/predict_smile", json=input_data)
    prediction = response.json().get("prediction")
    st.success(f"Предсказанное значение: {prediction}")

# Сохранение модели
st.header("Сохранение модели")

if st.button("Сохранить модель"):
    model_data = {
        "id": "pipeline_model",  
        "Scaler": "MinMaxScaler",
        "Encoder": "OneHotEncoder",
        "Feature_Selector": "PCA"
    }
    response = requests.post(f"{API_URL}/save", json=model_data)

    if response.status_code == 200:
        st.success(response.json().get("response"))
    else:
        st.error(f"Ошибка: {response.json().get('detail')}")

# Добавление в Streamlit интерфейс для установки модели
model_id = st.text_input("Введите ID модели для установки", "Default_model")
if st.button("Установить модель"):
    response = requests.post(f"{API_URL}/set", params={"id": model_id})

    if response.status_code == 200:
        st.success(f"Модель установлена: {response.json().get('response')}")
    else:
        st.error(f"Ошибка: {response.json().get('detail')}")

# Добавление в Streamlit интерфейс для отображения списка моделей
if st.button("Показать список моделей"):
    response = requests.get(f"{API_URL}/list_models")

    if response.status_code == 200:
        models = response.json().get("models")
        st.write(models)
    else:
        st.error(f"Ошибка: {response.json().get('detail')}")

