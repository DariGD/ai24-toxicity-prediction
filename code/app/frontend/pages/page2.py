import streamlit as st
from functions.scaler_split import preprocess_data
from functions.trainingmodel import display_model_results
from functions.pca import run_pca_and_model, save_model
from functions.svd import  run_svd_and_model, save_model_svd
from functions.SelectKbest import  run_select_kbest_and_model, save_model_skb
from logger import setup_logger
from functions.grid_search import perform_grid_search

@ st.cache_data
def run():
    logger = setup_logger(name="StreamlitApp")

    # инициализация состояния
    state_vars = ['fit_1', 'pca', 'lr_pca_model', 'svd', 'skb', 'X_train_pca', 'X_train_selected', 'X_train_svd']
    for var in state_vars:
        if var not in st.session_state:
            st.session_state[var] = False if var != 'lr_pca_model' else None

    # Начало работы
    if 'df_cleaned' not in st.session_state or st.session_state.df_cleaned is None:
        st.warning('Сначала пройдите этап предобработки')
        return
    
    df = st.session_state.df_cleaned
    X_train, X_test, y_train, y_test, ohe_columns = preprocess_data(df)

    st.subheader("Обучение модели на всех признаках")
    
    if st.button("Обучить модель"):
        st.session_state.fit_1 = True

    if st.session_state.fit_1: 
        display_model_results(X_train, y_train, X_test, y_test)
        logger.info("Обучена модель линейной регрессии")

    st.subheader("PCA и линейная регрессия")
    if st.button("PCA и линейная регрессия"):
        st.session_state.pca = True

    if st.session_state.pca:
        lr_pca_model = run_pca_and_model(X_train, X_test, y_train, y_test)
        logger.info("Проведено PCA и обучена модель линейной регрессии")
        st.markdown("**Видим, что коэффициенты модели уменьшились. Наш датасет с признаками представляет собой разреженную матрицу, можно попробовать адаптировать PCA как раз для модели разреженных данных.**")

    model_name = st.text_input("Введите имя файла для сохранения модели (без расширения):", "linear_regression_model")
    if st.checkbox("Сохранить модель"):
        save_model(lr_pca_model, model_name)
        logger.info("Модель успешно сохранена")

    st.subheader("SVD")
    if st.button("SVD"):
        st.session_state.svd = True 
    if st.session_state.svd:
        svd_model = run_svd_and_model(X_train, X_test, y_train, y_test)
        logger.info("Проведено SVD и обучена модель линейной регрессии")
        st.markdown("**SVD применен к данным, и модель обучена с использованием уменьшенной размерности.**")

        model_name = st.text_input("Введите имя файла для сохранения модели (без расширения):", "svd_model")
        if st.checkbox("Сохранить модель SVD"):
            model_name = st.text_input("Введите имя файла для сохранения модели (без расширения):", "svd_model")
            save_model_svd(svd_model, model_name)
            logger.info("Модель успешно сохранена")

    st.subheader("SelectKBest")
    if st.button("SelectKBest"):
        st.session_state.skb = True 
    if st.session_state.skb:
        skb_model = run_select_kbest_and_model(X_train, X_test, y_train, y_test)
        logger.info("Проведен отбор признаков по методу SelectKBest и обучена модель линейной регрессии")
        
        st.markdown("**SelectKBest применен к данным, и модель обучена с использованием выбранных признаков.**")
        model_name = st.text_input("Введите имя файла для сохранения модели (без расширения):", "SelectKbest_model")
        if st.checkbox("Сохранить модель SKB"):
            save_model_skb(skb_model, model_name)
            logger.info("Модель успешно сохранена")

    X_train_options = {
        'Без сжатия признакового пространства': X_train,
        'После PCA': st.session_state.X_train_pca,
        'После SKB': st.session_state.X_train_selected,
        'После SVD': st.session_state.X_train_svd
    }

    st.title("Подбор гиперпараметров")
    st.subheader("Grid Search для Lasso Регрессии")
    st.text("Здесь мы можем выбрать, на каком наборе данных будем проводить GS")
    # Выбор параметра X_train из списка
    selected_option = st.selectbox("Выберите вариант X_train:", list(X_train_options.keys()))
    # Запуск Grid Search с выбранным набором данных
    if st.button("Запустить Grid Search"):
        X_train = X_train_options[selected_option]
        with st.spinner("Поиск лучших параметров..."):
            best_model = perform_grid_search(X_train, y_train)
            st.write("Модель успешно обучена с лучшим параметром alpha.")
