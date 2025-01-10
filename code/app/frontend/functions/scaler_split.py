import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_data(df):
    X = df.drop(columns=['LD50', 'log_LD50'])  
    y = df['log_LD50']  
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=123)  

    # Определение категориальных и числовых признаков
    num_col = X.select_dtypes(include=['number']).columns.tolist()  
    cat_col = X.select_dtypes(include=['category']).columns.tolist()  

    # Разделение категориальных и числовых данных
    X_train_cat = X_train[cat_col]  
    X_train_num = X_train[num_col]  

    X_test_cat = X_test[cat_col]  
    X_test_num = X_test[num_col]  

    # Кодирование категориальных признаков
    OHE = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')  
    X_train_encoded = OHE.fit_transform(X_train_cat)  
    X_test_encoded = OHE.transform(X_test_cat)  

    ohe_columns = OHE.get_feature_names_out(cat_col)  
    X_train_cat = pd.DataFrame(X_train_encoded, columns=ohe_columns, index=X_train.index)  
    X_test_cat = pd.DataFrame(X_test_encoded, columns=ohe_columns, index=X_test.index)  

    # Масштабирование числовых признаков
    scaler = MinMaxScaler()  
    X_train_scaled = scaler.fit_transform(X_train[num_col])  
    X_test_scaled = scaler.transform(X_test[num_col])  

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=num_col, index=X_train.index)  
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=num_col, index=X_test.index)  

    # Объединение предобработанных данных
    X_train = pd.concat([X_train_cat, X_train_scaled], axis=1)  
    X_test = pd.concat([X_test_cat, X_test_scaled], axis=1)

    return X_train, X_test, y_train, y_test, ohe_columns
