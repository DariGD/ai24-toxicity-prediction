import requests

API_URL = "http://127.0.0.1:8000"

def train_model(file):
    files = {"file": file}
    response = requests.post(f"{API_URL}/fit", files=files)
    return response

def predict_smile(smiles, exp_animal, method):
    input_data = {
        "Smiles": smiles,
        "Exp_animal": exp_animal,
        "Method_of_administration": method
    }
    response = requests.post(f"{API_URL}/predict_smile", json=input_data)
    return response

def save_model(model_data):
    response = requests.post(f"{API_URL}/save", json=model_data)
    return response

def set_model(model_id):
    response = requests.post(f"{API_URL}/set", params={"id": model_id})
    return response

def list_models():
    response = requests.get(f"{API_URL}/list_models")
    return response
