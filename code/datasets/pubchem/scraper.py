import random
import time
import os
from datasets.pubchem import pubchem as pc
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-b", "--begin", help="First index to process.", type=int, default=0
)
# 118040 это максимальный номер в скачанном файле
parser.add_argument(
    "-e", "--end", help="Last index to process.", type=int, default=118040
)
parser.add_argument("-f", "--filename", help="Input file name.", required=True)
parser.add_argument("-p", "--save-path", help="Path to save files.")
parser.add_argument(
    "-u", "--upload", help="Enable upload results to S3.", action="store_true"
)


args = parser.parse_args()

start = args.begin
stop = args.end
file_name = args.filename

if args.save_path:
    save_path = args.save_path
else:
    save_path = os.path.dirname(file_name)

try:
    toxicity_data = pd.read_csv(file_name)
except IOError as e:
    print(f"File {file_name} not found.\n{e}")
    exit(1)

# version 0.1.1:    если было 404 по версии 0.1.0, то повторный запрос всей
#                   секции Toxicity и сохранение всего отвеа
# version 0.1.2:    разбр всей секции Toxicity и сохранение найденного
#                   списка строк
index = 0
for index, row in toxicity_data.loc[start:stop].iterrows():
    # Если создать файл с именем `stop` в текущем каталоге, то выйти
    if os.path.exists("./stop"):
        print(f"Stop file detected at index {index}. Exiting loop.")
        break

    print(
        f"{index}/{stop}: cid={row['cid']}, code={row['status_code']}, \
            version={row['version']} =========="
    )
    if row["status_code"] == 200:  # повторные проходы
        if row["version"] in ["0.1.0", "0.1.2"]:
            print("  Повтор. Code=200, v0.1.0/0.1.2 ... continue")
            continue
        elif row["version"] == "0.1.1":
            # разбор поля Toxicity_data_info и повышение версии строки
            print("  Повтор. Code=200, v0.1.1 ... upgrading to 0.1.2")
            r = eval(toxicity_data.at[index, "toxicity_data_info"])
            toxicity_data.at[index, "toxicity_data_value"] = \
                pc.find_string_elements(r)
            toxicity_data.at[index, "version"] = "0.1.2"
    else:
        try:
            print(f"  Loading new data for CID {row['cid']}")
            res = pc.get_toxicity_data(row["cid"], full_toxicity_data=True)
            time.sleep(random.random() / 4 + 0.2)
            if res.is_success:
                toxicity_data.at[index, "status_code"] = int(res.status_code)
                toxicity_data.at[index, "toxicity_data_info"] = res.json()
                toxicity_data.at[index, "toxicity_data_value"] = (
                    pc.find_string_elements(res.json())
                )
                toxicity_data.at[index, "version"] = "0.1.2"
            print(f"DONE: {index}/{stop}: cid={row['cid']}, {res.status_code}")
        except Exception as e:
            toxicity_data.at[index, "status_code"] = 0
            toxicity_data.at[index, "toxicity_data_info"] = None
            toxicity_data.at[index, "toxicity_data_value"] = None
            print(f"ERROR: {index}/{stop}: cid={row['cid']}: {e}")
    if index % 5000 == 0:
        # сохраняем промежуточные файлы через каждые 5000 строк
        toxicity_data.to_csv(
            f"{save_path}toxicity_data_index-0.1.2-{index}.csv",
            index=False
        )
        print(f"SAVED: {index}/{stop}: cid={row['cid']}")

# сохраняем последний файл
toxicity_data.to_csv(
    f"{save_path}toxicity_data_index-0.1.2-{index}.csv",
    index=False
)

if args.upload:
    print("Uploading files to S3...")
    pc.upload_file(
        f"{save_path}toxicity_data_index-0.1.2-{index}.csv",
        f"pubchem/toxicity_data_index-0.1.2-{index}.csv",
        progress=True
    )
