import streamlit as st
import importlib

# Словарь страниц
pages = {
    "Предобработка данных": "pages.page1",
    "Обучение моделей":"pages.page2",
    "Клиентский запрос": "pages.page3"
}

# Выбор страницы через боковую панель
selection = st.sidebar.radio("Выберите страницу", list(pages.keys()))

# Импорт выбранной страницы
page_module = importlib.import_module(pages[selection])
page_module.run()