# Module 'pubchem'
# Utilities to collect data from PubChem APIs

import pprint
import json
import boto3.session
import httpx
import boto3
import logging
from botocore.exceptions import ClientError
import os
from tqdm import tqdm
from boto3.s3.transfer import TransferConfig


logger = logging.getLogger()

base_PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
base_PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"


def smiles_to_cid(smiles):
    """
    Convert SMILES to CID by
    making HTTP GET call to PUG_REST API
    :param
        smiles: SMILES string
    :return:
        CID of the compound
    """
    response = httpx.get(f"{base_PUG_REST}/compound/smiles/{smiles}/CIDs/json")
    # Response structure
    # {
    #   "IdentifierList": {
    #     "CID": [
    #       6334
    #     ]
    #   }
    # }
    return response.json()["IdentifierList"]["CID"][0]


def get_toxicity_data(cid, full_toxicity_data=False):
    """
    Get toxicity data from PUG_VIEW API.
    Applies additional filtering to exctact only
    the "Toxicity Data" section from the response.
    :param
        cid: CID of the compound
    :return:
        JSON response from PubChem API
    """
    response = httpx.get(
        f"{base_PUG_VIEW}/data/compound/{cid}/JSON",
        params=f"heading=Toxicity{'' if full_toxicity_data else '+Data'}",
    )
    return response


def get_toxicity_data_value(json_response):
    """
    Extracts value from JSON response.
    Usually the value is located at
    ".Record.Section[0].Section[0].Section[0].Information[0].Value.StringWithMarkup[0].String"
    of the response JSON.
    :param
        json_response: JSON response from PUG_VIEW API
    :return:
        Value (string) from JSON response
    """
    return json_response["Record"]["Section"][0]["Section"][0]["Section"][0][
        "Information"
    ][0]["Value"]["StringWithMarkup"][0]["String"]


def find_string_elements(
        json_data, patterns=["ld50", "ldlo", "lc50", "td50", "bcf"]):
    """
    Находит все элементы с ключом "String",
    значения которых содержат шаблоны поиска.
    Args:
        json_data: JSON-данные для анализа.
        patterns: Список шаблонов для поиска.
            По умолчанию: ["ld50", "ldlo", "lc50", "td50", "bcf"]
    Returns:
        list: Список элементов, удовлетворяющих условию.
    """
    results = []

    def search(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "String" and any(
                    pattern in value.lower() for pattern in patterns
                ):
                    results.append(value)
                elif isinstance(value, (dict, list)):
                    search(value)
        elif isinstance(data, list):
            for item in data:
                search(item)

    search(json_data)
    return results


class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._pbar = tqdm(total=self._size, unit='B', unit_scale=True, desc=filename)

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        self._seen_so_far += bytes_amount
        self._pbar.update(bytes_amount)


def hse_s3_client():
    """
    Создает клиент для работы с S3.
    Используется профиль "hse-ai24-team22" и регион "ru-central1".
    Профиль "hse-ai24-team22" должен быть создан в файле ~/.aws/credentials.
    Пример:
        [hse-ai24-team22]
        aws_access_key_id = ************************
        aws_secret_access_key = **************************************
    """
    session = boto3.session.Session(
        profile_name="hse-ai24-team22", region_name="ru-central1"
    )
    s3 = session.client(
        service_name="s3", endpoint_url="https://storage.yandexcloud.net"
    )
    return s3


def upload_file(file_name, object_name=None, bucket="hse-ai24-team-22-data", progress=False):
    """
    Загружает файл в S3.
    Args:
        file_name: Имя файла.
        object_name: Ключ в S3. Если не задан, то имя файла (без пути) будет ключом.
        bucket: Бакет. По умолчанию: "hse-ai24-team-22-data".
        progress: Отображать прогресс. По умолчанию: False.
    Returns:
        True если успешно, иначе False.
    """

    if object_name is None:
        object_name = os.path.basename(file_name)

    s3 = hse_s3_client()
    try:
        s3.upload_file(file_name, bucket, object_name,
                Callback=ProgressPercentage(file_name) if progress else None)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_file(object_name, file_name=None, bucket="hse-ai24-team-22-data", progress=False):
    """
    Скачивает файл из S3.
    Args:
        object_name: Ключ в S3.
        file_name: Имя файла для сохранения. Если не задано, то будет "./{object_name}"
        bucket: Бакет. По умолчанию: "hse-ai24-team-22-data".
    """

    if file_name is None:
        file_name = os.path.join("./", object_name)

    s3 = hse_s3_client()
    try:
        s3.download_file(bucket, object_name, file_name, Callback=ProgressPercentage(file_name) if progress else None)
    except ClientError as e:
        logging.error(e)
        return False
    return True


if __name__ == "__main__":
    # smiles = input("Smiles: ")
    # cid = smiles_to_cid(smiles)
    # print(f"CID: {cid}")
    # toxicity_data = get_toxicity_data(cid)
    # json.dump(toxicity_data, open("toxicity.json", "w"), indent=4)
    # pprint.pp(toxicity_data)
    # pprint.pp(get_toxicity_data_value(toxicity_data))

    upload_file("toxicity.json", "data/toxicity.json", progress=True)

    download_file("data/toxicity.json", "toxicity_downloaded.json", progress=True)

    # Получить список объектов в бакете
    # for key in s3.list_objects(Bucket='bucket-name')['Contents']:
    #     print(key['Key'])
