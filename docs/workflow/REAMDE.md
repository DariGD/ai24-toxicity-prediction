# Процесс работы Команды 22

> Отжатая и адаптированная версия [The Data Science Lifecycle Process](https://github.com/dslp/dslp/blob/main/README.md)



## Главные темы

- [Шаги процесса](steps.md)
- [Шаблоны задач (issue templates)](issue-types/0-overview-issue-types.md)
- [Стратегия ветвления (Branching Strategy)](branch-types.md)
- [Семантическое версионирование ассетов (Semantic Version of Data Science Assets)](semantic-versioning.md)
- [Метки (Labels)](labels.md)
- [Структура репозитория (Standard Repo Structure)](repo-structure.md)

## Допущения, соглашения
- в качестве основной используется только ветка `main` (нет ветки `develop` и т.п.)
- коммиты в `main` запрещены

## Переиспользование кода
Для удобства переиспользования в вашем коде любых наработок из директории `./code`, выполните в корневой папке проекта
```sh
pip install .
```
Это приведет к запуску срипта `setup.py`, который обнаружит и обновит ссылки на все модули внутри `./code`, таким образом, что они будут доступны для импорта без указания путей. Вот так 
```python
import datasets
import deployment
import features
import models
```
или
```python
from datasets import ...
```