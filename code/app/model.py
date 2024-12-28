import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import impute
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import TruncatedSVD
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, train_test_split

# file_path = r"D:\Programming\Projects\ai24-toxicity-prediction\data\processed\resultwithfeatures.csv"
# df = pd.read_csv(file_path, low_memory = False)

# Избавимся от ненужных колонок
def drop_columns(data):
    data = data.drop(['Source', 'Smiles',
        'LD50 (a.u.)', 'Carcinogenicity', 'Carcinogenicity (a.u.)',
        'Hepatoxicity', 'Hepatoxicity (a.u.)', 'Eye_Corrosion',
        'Eye_Corrosion (a.u.)', 'Eye_Irritation', 'Eye_Irritation (a.u.)',
        'Mutagenicity', 'Mutagenicity (a.u.)', 'Respiratory_Toxicity',
        'Respiratory_Toxicity (a.u.)', 'LC50', 'LC50 (a.u.)', 'NOAEL',
        'NOAEL (a.u)'], axis = 1).reset_index(drop = True)
    return data

# Избавимся от пропусков в целевой переменной
def dropNA():
    df = df.dropna(subset=['LD50'])

# Избавимся от выбросов
def remove_outliers(df):
    Q10 = df['LD50'].quantile(0.10)
    Q90 = df['LD50'].quantile(0.90)
    IQR = Q90 - Q10

    lower_bound = Q10 - 1.5 * IQR
    upper_bound = Q90 + 1.5 * IQR

    df = df[(df['LD50'] >= lower_bound) & (df['LD50'] <= upper_bound)]
    df = df[df['LD50'] != 0]
    return df

# Удалим колонки с константными значениями
def drop_constant_columns(df, threshold=0.996):
    to_drop = []  # Список столбцов для удаления
    for col in df.columns:
        # Рассчитываем долю самого частого значения
        max_freq = df[col].value_counts(normalize=True).max()
        if max_freq >= threshold:
            to_drop.append(col)
    # Удаляем столбцы
    df = df.drop(columns=to_drop)
    return df


# Логарифмируем таргет
def log_target(df):
    df['LD50'] = np.log(df['LD50'])

# Разобьем на X и y
def X_y_split(df):
    X = df.drop(columns=['LD50', 'log_LD50'])
    y = df['log_LD50']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=123)


def OHE_encode(X):
    num_col = X.select_dtypes(include=['number']).columns.tolist()
    cat_col = X.select_dtypes(include=['object']).columns.tolist()

    X_cat = X[cat_col]
    X_num = X[num_col]

    OHE = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop = 'first')  # handle_unknown='ignore' для новых категорий в тесте

    X_encoded = OHE.fit_transform(X_cat)

    ohe_columns = OHE.get_feature_names_out(cat_col)
    X_cat = pd.DataFrame(X_encoded, columns=ohe_columns, index=X.index)
    
    X = pd.concat([X_cat, X_num], axis=1)
    return X

def scale_num(X):
    num_col = X.select_dtypes(include=['number']).columns.tolist()
    cat_col = X.select_dtypes(include=['object']).columns.tolist()

    X_cat = X[cat_col]
    X_num = X[num_col]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X[num_col])

    X_scaled = pd.DataFrame(X_scaled, columns=num_col, index=X.index)

    X = pd.concat([X_cat, X_scaled], axis=1)
    return X

def fit_model():
    lr = LinearRegression()
    lr.fit(X_train, y_train)

def transform_PCA():
    pca = PCA(n_components=0.95)  # 95% объяснённой дисперсии

    # Подгоняем PCA на тренировочных данных и трансформируем их
    X_train = pca.fit_transform(X_train)

def transform_SVD():
    # Применение SVD
    svd = TruncatedSVD(n_components=600, random_state=42)  # Сохраняем 500 главных компонент
    X_train = svd.fit_transform(X_train)          # Применяем SVD на тренировочные данные

def transform_KBest():
    selector = SelectKBest(score_func=f_regression, k=500)
    X_train = selector.fit_transform(X_train, y_train)
