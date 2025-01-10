import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pages.preprocessdata import preprocess_dataset, drop_constant_columns, analyze_ld50
from pages.logger import setup_logger

def run():
    logger = setup_logger(name="StreamlitApp")

    st.title("Определение токсичности химических веществ")

    # Загрузка файла
    @ st.cache_data
    def load_data(uploaded_file):
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)

    st.subheader("Загрузка CSV файла")
    uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

    df = load_data(uploaded_file)

    if df is not None:
        num_rows = st.number_input("Введите количество строк для отображения:", min_value=1, max_value=len(df), value=5) 
        if st.button("Показать загруженный датасет"):
                    st.write(f"Загруженный датасет (первые {num_rows} строк):")
                    logger.info("Датасет загружен и показан!")
                    st.dataframe(df.head(num_rows))
    # Предобработка
    st.subheader("Предобработка")
    st.text("Избавляемся от дубликатов, оставляем для предсказания только таргет LD50 ")

    if st.checkbox('Сделаем предобработку датасета'):
        try:
            df = preprocess_dataset(df)
            st.success("Датасет успешно предобработан ✅")
        except Exception as e:
            st.error(f"Произошла ошибка при предобработке: {e}")


    st.text("Мы хотим убедиться, что у нас нет признаков, которые были бы статичные во всем датасете, чтобы мимнимзировать риск переобучения на них, поэтому сразу удаляем их ")
    st.text("Удаляем столбцы, где 99.6% значений — это одно уникальное значение")

    #Удаление признаков
    threshold = st.number_input("Введите порог для удаления столбцов (у нас используется 99.6%):", 
                                    min_value=0.98, max_value=0.999, value=0.996, step=0.001)
    if st.button("Удалить признаки с постоянными значениями"):
        try:
            df_cleaned = drop_constant_columns(df, threshold)
            st.session_state.df_cleaned = df_cleaned
            st.success(f"Признаки очищены. Осталось столбцов: {df_cleaned.shape[1]}")
            st.dataframe(df_cleaned)
        except Exception as e:
            st.error(f"Ошибка при удалении признаков: {e}")

    if "df_cleaned" not in st.session_state:
        st.session_state.df_cleaned = None

    st.subheader("Анализ данных LD50")
    if st.session_state.df_cleaned is not None:
        try:
            df_cleaned = st.session_state.df_cleaned

            # Кнопки для отображения графиков
            if st.button("Показать график LD50"):
                st.session_state.show_ld50 = True
            if st.button("Показать график log_LD50"):
                st.session_state.show_log_ld50 = True

            # Добавляем логарифм LD50
            df_cleaned = analyze_ld50(df_cleaned)
            st.session_state.df_cleaned = df_cleaned

            # Отображение графиков
            if st.session_state.get("show_ld50", False):
                plt.figure(figsize=(10, 5))
                df_cleaned['LD50'].hist(bins=30, color='blue', alpha=0.7)
                plt.title('Гистограмма LD50')
                plt.xlabel('LD50')
                plt.ylabel('Частота')
                st.pyplot(plt)

            if st.session_state.get("show_log_ld50", False):
                plt.figure(figsize=(10, 5))
                df_cleaned['log_LD50'].hist(bins=30, color='green', alpha=0.7)
                plt.title('Гистограмма log_LD50')
                plt.xlabel('log_LD50')
                plt.ylabel('Частота')
                st.pyplot(plt)
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
    else:
        st.warning("Сначала удалите признаки с постоянными значениями, чтобы продолжить.")
