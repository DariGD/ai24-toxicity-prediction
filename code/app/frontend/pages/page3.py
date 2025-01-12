import streamlit as st
from functions.api_client import train_model, predict_smile, save_model, set_model, list_models
from functions.logger import setup_logger

def run():
    logger = setup_logger(name="StreamlitApp")
    st.title("Модель предсказания LD50")
    st.text("Теперь покажем, как наша модель работает. Для этого предварительно запускаем сервер с FastAPI")

    # Загрузка CSV файла для обучения модели
    st.header("Обучение модели")
    uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

    if uploaded_file is not None:
        if st.button("Обучение модели"):
            response = train_model(uploaded_file)
            if response.status_code == 200:
                st.success(response.json().get("response"))
            else:
                st.error(f"Ошибка: {response.json().get('detail')}")

    # Ввод данных для предсказания
    st.header("Предсказание")
    smiles_input = st.text_input("Введите SMILES:")
    exp_animal = st.selectbox("Выберите экспериментальное животное:", ["mouse", "rat"])
    method_of_administration = st.selectbox("Выберите метод введения:", ['intraperitoneal', 'oral', 'intravenous', 'intramusculary', 'skin',
                                        'inhalation', 'percutaneous', 'sc', 'subcutaneous', 'gavage',
                                        'intramuscular', 'diet'])

    if st.button("Сделать предсказание"):
        response = predict_smile(smiles_input, exp_animal, method_of_administration)
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            st.success(f"Предсказанное значение: {prediction}")
        else:
            st.error(f"Ошибка: {response.json().get('detail')}")

    # Сохранение модели
    st.header("Сохранение модели")
    if st.button("Сохранить модель"):
        model_data = {
            "id": "pipeline_model",  
            "Scaler": "MinMaxScaler",
            "Encoder": "OneHotEncoder",
            "Feature_Selector": "PCA"
        }
        response = save_model(model_data)
        if response.status_code == 200:
            st.success(response.json().get("response"))
        else:
            st.error(f"Ошибка: {response.json().get('detail')}")

    # Установка модели
    model_id = st.text_input("Введите ID модели для установки", "Default_model")
    if st.button("Установить модель"):
        response = set_model(model_id)
        if response.status_code == 200:
            st.success(f"Модель установлена: {response.json().get('response')}")
        else:
            st.error(f"Ошибка: {response.json().get('detail')}")

    # Отображение списка моделей
    if st.button("Показать список моделей"):
        response = list_models()
        if response.status_code == 200:
            models = response.json().get("models")
            st.write(models)
        else:
            st.error(f"Ошибка: {response.json().get('detail')}")
