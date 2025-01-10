import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_dataset(df):
    try:
        # Удаление дубликатов
        df = df.drop_duplicates(subset=['Smiles', 'Exp. Animal', 'Method of administration'])

        # Преобразование категориальных признаков
        obj_cols = df.select_dtypes('object').columns
        df[obj_cols] = df[obj_cols].astype('category')

        # Downcast числовых признаков
        fcols = df.select_dtypes('float').columns
        icols = df.select_dtypes('integer').columns
        df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
        df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

        # Удаление ненужных колонок
        df = df.drop([
            'Source', 'Smiles', 'LD50 (a.u.)', 'Carcinogenicity', 'Carcinogenicity (a.u.)',
            'Hepatoxicity', 'Hepatoxicity (a.u.)', 'Eye_Corrosion', 'Eye_Corrosion (a.u.)',
            'Eye_Irritation', 'Eye_Irritation (a.u.)', 'Mutagenicity', 'Mutagenicity (a.u.)',
            'Respiratory_Toxicity', 'Respiratory_Toxicity (a.u.)', 'LC50', 'LC50 (a.u.)',
            'NOAEL', 'NOAEL (a.u)'
        ], axis=1).reset_index(drop=True)

        # Удаление записей с NaN в таргетной переменной
        df = df.dropna(subset=['LD50'])

        # Удаление выбросов
        Q10 = df['LD50'].quantile(0.10)
        Q90 = df['LD50'].quantile(0.90)
        IQR = Q90 - Q10
        lower_bound = Q10 - 1.5 * IQR
        upper_bound = Q90 + 1.5 * IQR
        df = df[(df['LD50'] >= lower_bound) & (df['LD50'] <= upper_bound)]

        # Удаление записей с LD50 равным 0
        df = df[df['LD50'] != 0]

        logger.info("Датасет успешно предобработан")
        return df
    except Exception as e:
        logger.error(f"Ошибка при предобработке датасета: {e}")
        raise

# Функция для удаления столбцов с постоянными значениями
def drop_constant_columns(df, threshold):
        to_drop = []  # Список столбцов для удаления
        for col in df.columns:
            # Рассчитываем долю самого частого значения
            max_freq = df[col].value_counts(normalize=True).max()
            if max_freq >= threshold:
                to_drop.append(col)
        # Удаляем столбцы
        logger.info(f"Удалено {len(to_drop)} столбцов с постоянными значениями или высоким порогом частоты")
        return df.drop(columns=to_drop)

def analyze_ld50(df):
    try:
        # Удаление пропущенных значений
        df = df.dropna()
        # Создание нового столбца log_LD50
        df['log_LD50'] = np.log(df['LD50'])
        logger.info("Добавлен столбец log_LD50")
        return df
    except Exception as e:
        logger.error(f"Ошибка при анализе LD50: {e}")
        raise