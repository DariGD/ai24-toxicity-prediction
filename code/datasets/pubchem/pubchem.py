# Module 'pubchem'
# Utilities to collect data from PubChem APIs

import httpx
from datasets.utils import upload_file, download_file


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
        json_data, patterns=["ld50", "ldlo", "lc50", "td50", "bcf"]
):
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


# Тут разные эксперименты для отладки.
# При импорте как модуль, код ниже не исполняется
if __name__ == "__main__":
    # smiles = input("Smiles: ")
    # cid = smiles_to_cid(smiles)
    # print(f"CID: {cid}")
    # toxicity_data = get_toxicity_data(cid)
    # json.dump(toxicity_data, open("toxicity.json", "w"), indent=4)
    # pprint.pp(toxicity_data)
    # pprint.pp(get_toxicity_data_value(toxicity_data))

    upload_file(
        "toxicity.json",  # путь к файлу на локальной машине
        "data/toxicity.json",  # ключ - т.е. имя записываемого объекта в S3
        progress=True  # отображать прогресс
    )

    download_file(
        "data/toxicity.json",  # ключ - т.е. имя считываемого объекта в S3
        "toxicity_downloaded.json",  # путь к файлу на локальной машине
        progress=True  # отображать прогресс
    )

    # Получить список объектов в бакете
    # for key in s3.list_objects(Bucket='bucket-name')['Contents']:
    #     print(key['Key'])
