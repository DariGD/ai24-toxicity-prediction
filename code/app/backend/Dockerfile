FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы приложения в рабочую директорию
COPY fastAPI.py ./
COPY requirements.txt ./
COPY pipeline_model.pkl /app/pipeline_model.pkl
COPY "MACCSname.xlsx" /app/"MACCSname.xlsx"

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
# Указываем команду для запуска приложения
CMD ["uvicorn", "fastAPI:app", "--host", "0.0.0.0", "--port", "8000"]