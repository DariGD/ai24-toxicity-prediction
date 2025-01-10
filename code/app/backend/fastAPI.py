from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal, Annotated
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Union
from sklearn.linear_model import LinearRegression
import joblib
from fastapi import FastAPI, UploadFile, Request, File, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import time
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import io

def smiles_to_2d_descriptors(smiles):
    # Преобразование SMILES в молекулу
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError("Неверный SMILES")
    
    # Вычисление дескрипторов
    descriptors = {}
    try:
        for name, function in Descriptors.descList:
            descriptors[name] = function(mol)
    except Exception as e:
        raise ValueError(f"Ошибка при вычислении дескрипторов: {e}")

    return descriptors

app = FastAPI()

set_model = None
models = []
default_model = joblib.load("pipeline_model.pkl")
dd = pd.read_excel("MACCSname.xlsx")

# Добавим дефолтную модель в общий список
model = {}
model['id'] = 'Default_model'
model['Scaler'] = "MinMaxScaler"
model['Encoder'] = "OHE"
model['Feature_Selector'] = "PCA"
model['pipeline'] = default_model 
models.append(model)

class InputData(BaseModel):
    Exp_animal: Literal['mouse', 'rat'] = Field(..., example='mouse')
    Method_of_administration: Literal['intraperitoneal', 'oral', 'intravenous', 'intramusculary', 'skin',
                                      'inhalation', 'percutaneous', 'sc', 'subcutaneous', 'gavage',
                                      'intramuscular', 'diet'] = Field(..., example='oral')
    Smiles: Annotated[str, Field(..., example='CCO')]  # Изменено на Smiles

class ModelData(BaseModel):
    id: Union[str, None] = 'pipeline_model'
    Scaler: Union[str, None] = "MinMaxScaler"
    Encoder: Union[str, None] = "OneHotEncoder"
    Feature_Selector: Union[str, None] = "PCA"

class PredictionResponse(BaseModel):
    prediction: Annotated[float, Field(description="Prediction results")]

class SaveResponse(BaseModel):
    response: Annotated[str, Field(description="Result")]

class SetResponse(BaseModel):
    response: Annotated[str, Field(description="Выбранная модель")]

class ModelListResponse(BaseModel):
    models: List[Dict]

class FitResponse(BaseModel):
    response: Annotated[str, Field(description="Result")]

class InfoResponse(BaseModel):
    length_of_categorical_columns: int = 2
    length_of_numerical_columns: int = 1379

def smiles_to_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Получение дескрипторов 2d
    descriptors_2d = smiles_to_2d_descriptors(smiles)
    descriptors_2d_dict = {}
    for name, value in descriptors_2d.items():
        descriptors_2d_dict[name] = value
    descriptors_2d_df = pd.DataFrame(descriptors_2d_dict, index=[0])
                                     
    # Получение дескрипторов Морган
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    morgan_array = list(morgan_fp)
    morgan_df = pd.DataFrame(morgan_array)
    morgan_df.index = morgan_df.index + 1
    morgan_df = morgan_df.T
    morgan_df = morgan_df.add_prefix("MorganFP_")

    # Получение дескрипторов MACCS
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_array = list(maccs_fp)

    # Создание DataFrame для MACCS
    maccs_df = pd.DataFrame([maccs_array])
    maccs_df.columns = dd['Unnamed: 0']
    del maccs_df['Unnamed']  # Переименование столбцов MACCS

    # Объединяем дескрипторы
    return descriptors_2d_df, maccs_df, morgan_df  # Возвращаем дескрипторы Морган и MACCS, 2d

@app.post("/fit", response_model=FitResponse)
async def fit_model(file: UploadFile = File(...)) -> FitResponse: 
    start_time = time.time()  # Начало отсчета времени
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        X = df.drop(columns=['LD50', 'log_LD50'])
        y = df['log_LD50']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=123)

        # Разделение на числовые и категориальные признаки
        num_col = X.select_dtypes(include=['number']).columns.tolist()
        cat_col = X.select_dtypes(include=['object']).columns.tolist()

        # Создание пайплайнов для преобразования данных
        numerical_transformer = Pipeline(steps=[
            ("scaler", MinMaxScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop='first')),
        ])

        data_transformer = ColumnTransformer(transformers=[
            ("numerical", numerical_transformer, num_col),
            ("categorical", categorical_transformer, cat_col),
        ])

        preprocessor = Pipeline(steps=[("data_transformer", data_transformer)])

        # Создание полного пайплайна модели
        model_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor),
                   ("feature_selector", PCA(n_components=0.95)),
                   ("model", LinearRegression())])

        # Обучение модели
        model_pipeline.fit(X_train, y_train)

        # Cохраняем модель
        n = len(models) + 1
        new_model = {}
        new_model['id'] = f"Model{n}"
        new_model['Scaler'] = "MinMaxScaler"
        new_model['Encoder'] = "OHE"
        new_model['Feature_Selector'] = "PCA"
        new_model['pipeline'] = model_pipeline
        models.append(new_model)

        elapsed_time = time.time() - start_time  # Время выполнения

        if elapsed_time > 10:
            response = "Модель обучается больше 10 секунд"
            return FitResponse(response=response)
            
        response = "Модель обучилась"
        return FitResponse(response=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred during fitting: {str(e)}")


@app.post("/predict_smile", response_model=PredictionResponse)
def predict_smile(input_data: Annotated[InputData, Body(...)]) -> PredictionResponse:
    data = input_data.model_dump()
    df = pd.DataFrame(data, index=[0])
    df['Exp. Animal'] = df['Exp_animal']
    df['Method of administration'] = df['Method_of_administration']
    df = df.drop(['Method_of_administration', 'Exp_animal'], axis=1)

    # Преобразуем SMILES в дескрипторы
    descriptors_2d, maccs_df, morgan_df = smiles_to_descriptors(data['Smiles'])
    X = pd.concat([descriptors_2d, df, morgan_df, maccs_df], axis=1)
    del X['Smiles']

    # Делаем предсказание
    # Используем выбранную модель. Если модель не выбрана, используем дефолтную
    model_pipeline = default_model
    if set_model is not None:
        model_pipeline = set_model
    prediction = model_pipeline.predict(X)

    return PredictionResponse(prediction=prediction[0])

@app.post("/save", response_model=SaveResponse)
async def save(input_data: Annotated[ModelData, Body(...)]) -> SaveResponse:
    data = input_data.model_dump()
    model_pipeline = joblib.load(f"{data['id']}.pkl")
    data['pipeline'] = model_pipeline
    models.append(data)
    response = "Model saved"

    return SaveResponse(response=response)

@app.post("/set", response_model=SetResponse)
async def set(id: str = 'pipeline_model') -> SetResponse:
    global set_model
    response = models[0]['id']
    for model in models:
        if model['id'] == id:
            set_model = model['pipeline']    
            response = id
            
    return SetResponse(response=response)

@app.get("/list_models", response_model=ModelListResponse)
async def list_models():
    models_response = []
    for model in models:
        response = {'id': model['id'],
                    'scaler': model['Scaler'],
                    'encoder': model['Encoder'],
                    'feature_selector': model['Feature_Selector']}
        models_response.append(response)
    return ModelListResponse(models=models_response)