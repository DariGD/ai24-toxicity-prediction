import os
import logging
from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError


class ProgressPercentage(object):
    def __init__(self, filename, filesize=None):
        self._filename = filename
        # При скачивании мы не знаем размер, поэтому ждем его как параметр
        if filesize is None:
            self._size = os.path.getsize(filename)
        else:
            self._size = filesize
        self._seen_so_far = 0  # количество байт, которые уже скачаны
        # создаем progress bar с помощью tqdm
        self._pbar = tqdm(
            total=self._size,  # общее количество байт
            unit="B",  # единица измерения (B - байты)
            unit_scale=True,  # автовыбор единиц измерения (B, KB, MB, ...)
            desc=filename,  # отображаемое описание progress bar
        )

    # Этот метод будет вызываться как Callback в функции скачивания/загрузки
    # bytes_amount - количество байт, которые уже скачены/загружены
    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        self._pbar.update(bytes_amount)


def hse_s3_client(profile_name="hse-ai24-team22"):
    """
    Создает клиент для работы с S3.
    По умолчанию берет профиль "hse-ai24-team22" и регион "ru-central1".
    Профиль должен быть создан в файле ~/.aws/credentials.
    Пример:
        [hse-ai24-team22]
        aws_access_key_id = ************************
        aws_secret_access_key = **************************************
    """
    # Сессия для boto3 - считает ключи из файла ~/.aws/credentials
    session = boto3.session.Session(
        profile_name=profile_name,  # имя профиля в конфиге
        region_name="ru-central1",  # регион "ru-central1" - так надо ;)
    )
    s3 = session.client(
        service_name="s3",  # название сервиса (boto3 может разные сервисы)
        endpoint_url="https://storage.yandexcloud.net",  # точка входа
    )
    return s3


def upload_file(
    file_name, object_name=None, bucket="hse-ai24-team-22-data", progress=False
):
    """
    Загружает файл в S3.
    Args:
        file_name: Имя файла.
        object_name: Ключ в S3. По умолчанию использует имя файла.
        bucket: Бакет. По умолчанию: "hse-ai24-team-22-data".
        progress: Отображать прогресс. По умолчанию: False.
    Returns:
        True если успешно, иначе False.
    """
    # Если ключ не задан, то используем имя файла
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3 = hse_s3_client()  # Создаем клиент для работы с S3
    try:
        s3.upload_file(
            file_name,  # путь к файлу на локальной машине
            bucket,  # бакет
            object_name,  # ключ - т.е. имя записываемого объекта в S3
            # объект, обслуживающий обратный вызов для показа прогресса
            Callback=ProgressPercentage(file_name) if progress else None,
        )
    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_file(
    object_name, file_name=None, bucket="hse-ai24-team-22-data", progress=False
):
    """
    Скачивает файл из S3.
    Args:
        object_name: Ключ в S3 (т.е. имя записываемого объекта в S3)
        file_name: Имя файла для сохранения. По умолчанию использует ключ.
        bucket: Бакет. По умолчанию: "hse-ai24-team-22-data".
    """

    # Если имя файла не задано, то используем ключ для имени
    if file_name is None:
        file_name = os.path.join("./", object_name)

    s3 = hse_s3_client()  # Создаем клиент для работы с S3
    try:
        # Проверим существует ли нужный объект в бакете
        response = s3.list_objects(Bucket=bucket, Prefix=object_name)
        logging.debug(f"Response: {response}")
        # Если в ответе (dict) нет ключа "Contents", то объект не найден
        if len(response.get("Contents", [])) == 0:
            logging.error(f"Object {object_name} not found in bucket {bucket}")
            return False
        # Получим размер файла для отображения прогресса
        filesize = response["Contents"][0]["Size"]
        print(f"File size: {filesize} bytes")
        s3.download_file(
            bucket,  # бакет, по умолчанию: "hse-ai24-team-22-data"
            object_name,  # ключ - т.е. имя считываемого объекта в S3
            file_name,  # путь к файлу на локальной машине
            # объект, обслуживающий обратный вызов для показа прогресса
            Callback=(
                ProgressPercentage(file_name, filesize) if progress else None
            ),
        )
    except ClientError as e:
        logging.error(e)
        return False
    return True
