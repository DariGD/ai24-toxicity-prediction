import streamlit as st
import pandas as pd
from model import *
import seaborn as sns

st.title("Data analysis with streamlit")
st.header("1 step: upload file")

# Загрузка файла
uploaded_file = st.file_uploader("Choose a csv file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head(5))
    st.write(f"Размер таблицы: {df.shape}")
else:
    st.write("Upload a csv file")

# Очистка данных
if uploaded_file is not None:
    st.header("2 step: Clean data")

    if st.checkbox("Избавимся от пропусков в таблице и ненужных колонок"):
        df = drop_columns(df)
        df = df.dropna(subset=['LD50'])
        df = df.dropna()
        st.dataframe(df.head(5))
        st.write(f"Размер таблицы: {df.shape}")

    if st.checkbox("Избавимся от выбросов"):
        df = remove_outliers(df)
        st.bar_chart(df['LD50'])
        st.write(f"Размер таблицы: {df.shape}")
    
    if st.checkbox("Избавимся от колонок с константными значениями"):
        df = drop_constant_columns(df)
        st.dataframe(df.head(5))
        st.write(f"Размер таблицы: {df.shape}")

# EDA
if uploaded_file is not None:
    st.header("EDA:")

    if st.button("Отобразить описательную статистику"):
        st.write(df.describe())

    if st.button("Отобразить график целевой переменной"):
        st.bar_chart(df['LD50'])

    st.subheader("Отобразить категориальные признаки:")
    cat_cols = df.select_dtypes(include=['object']).columns
    column = st.selectbox("Выберите признак для отображения", cat_cols)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    top_values = df[column].value_counts().head(5)
    ax = sns.barplot(x=top_values.index, y=top_values.values)
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.grid()
    st.pyplot(fig)

# Шаг 3
if uploaded_file is not None:
    st.header("Шаг 3: Кодировка и масштабирование признаков")
    X = df.drop(['LD50'], axis=1)
    y = df['LD50']

    if st.checkbox("Применение OHE кодирования"):
        X = OHE_encode(X)
        st.dataframe(X.head(5))
    
    if st.checkbox("Масштабирование данных при помощи MinMaxScaler"):
        X = scale_num(X)
        st.dataframe(X.head(5))

# Обучение модели
if uploaded_file is not None:
    st.header("Шаг 4: Обучение модели")
    X_new = X

     # Уменьшаем признаковое пространство
    st.subheader("Уменьшаем признаковое пространство")
    methods = ['', 'PCA', 'SVD', 'KBest']
    method = st.selectbox("Выберите метод", methods)

    if method == 'PCA':
        pca = PCA(n_components=0.95)  # 95% объяснённой дисперсии
        # Подгоняем PCA на тренировочных данных и трансформируем их
        X_new = pca.fit_transform(X)
        st.write(f"Размер X до PCA: {X.shape}")
        st.write(f"Размер X после PCA: {X_new.shape}")
    elif method == 'SVD':
        # Применение SVD
        svd = TruncatedSVD(n_components=600, random_state=42)  # Сохраняем 500 главных компонент
        X_new = svd.fit_transform(X)          # Применяем SVD на тренировочные данные
        st.write(f"Размер X до SVD: {X.shape}")
        st.write(f"Размер X после SVD: {X_new.shape}")
    elif method == 'KBest':
        selector = SelectKBest(score_func=f_regression, k=500)
        X_new = selector.fit_transform(X, y)
        st.write(f"Размер X до KBest: {X.shape}")
        st.write(f"Размер X после Kbest: {X_new.shape}")

    if st.checkbox("Обучить линейную модель"):
        lr = LinearRegression()
        lr.fit(X_new, y)
        st.write("Модель обучена")

        if st.checkbox("Отобразить веса модели"):
            fig, ax = plt.subplots()
            plt.hist(lr.coef_)
            st.pyplot(fig)

        if st.button("Cохранить модель"):
            st.write('Модель сохранена')
