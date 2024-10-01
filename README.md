# ПРОГНОЗИРОВАНИЕ ТОКСИЧНОСТИ ХИМИЧЕСКИХ СОЕДИНЕНИЙ

## Введение
На сегодняшний день химическое пространство оценивается уже в 10^60 малых молекул. Поиск химических структур, обладающих целевыми показателями, в таком огромном массиве довольно затруднителен. Химикам необходимы новые методы навигации и фильтрации химического пространства.

## Цель	работы
Разработка группы моделей классификации и регрессии для прогнозирования различных метрик токсичности химических соединений.

## Основные этапы для достижения цели

1) Сбор известных данных от токсичности химических соединений (до 20.10.2024)
2) Генерация физико-химических дескрипторов и EDA по ним (до 11.11.2024)
3) Построение первых ML моделей (до 01.12.2024)
4) Дополнительная генерация признаков и улучшение текущих моделей (до 01.02.2024)
5) Сервис над готовыми моделями (до 01.04.2024)
6) Эксперименты с DL (до 15.05.2024)

## Сбор данных
Сбор данных будет осуществляться по 30 основным источникам:

<details>
  <summary>Список источников</summary>
  
1. PubChem 
2. ChemSpider
3. ChemBL 
4. CompTox Chemicals Dashboard
5. COMPTOX
6. TOXRIC  
7. Open Food Tox
8. OpenTox 
9. Acute Toxicity Test Database Query
10. Exploring ToxCost Data
11. TOX 21
12. Comparative Toxicogenomics Database (CTD)
13. ECOTOX 
14. European Chemicals Agency (ECHA)  
15. EMBL-EBI (European Bioinformatics Institute) 
16. Chemical Effects in Biological Systems (CEBS)  
17. UK Chemical Hazards Compendum 
18. Pharmaceuticals in the Environment, Information for Assessing Risk website 
19. Human Metabolome Database (HMDB) 
20. PA Integrated Risk Information System (IRIS)
21. NORMAN Suspect List Exchange
22. SIDER 
23. The Carcinogenic Potency Database
24. Chemical Carcinogenesis Research Information System 
25. Life Science Database Archive
26. TOXICO DB
27. KEGG: Kyoto Encyclopedia of Genes and Genomes
28. RepDose
29. Публикация описывающая построение CATMoS: Collaborative Acute Toxicity Modeling Suite 
30. Публикация об анализе баз данных по токсичности

</details>

Каждый источник будет проработан на предмет "полезности" - то есть сколько в нём данных и как легко их достать. Качественно-количественное описание источников будет [тут](https://docs.google.com/spreadsheets/d/1R5E_jR72js-ya6yBi1ozjt458giZ-A_6EqPiU1vfdLg/edit?gid=0#gid=0)
Итоговый датасет будет сформирован на базе доступной информации из этих источников. 

Для того чтобы дальше было удобно работать с данными, нужно привести собранные в единую структуру.
Для этого потребуется:
- все индификаторы перевести в Smiles
- определеть на каком животном производились замеры
- способ введения вещества
- в рамках одной метрики привести все значения к одним единицам измерения

Итоговый формат данных будет иметь следующую структуру:
|Поле|Описание|
|---|---|
|Smiles|Смайлс химического вещества|
|the experimental animal|подопытное животное|
|the input method|метод введения вещества|
|metric_1|найденная метрика №1|
|metric_2|найденная метрика №2|
|...|...|
|metric_N|найденная метрика №N|


### Метрики токсичности
Существует огромное множество метрик токсичности:

<details>
  <summary>Список метрик</summary>
  
1. Developmental toxicity
2. Skin Sensitization
3. Blood Brain Barrier Penetration
4. BBB-CHT mediated BBB permeation
5. Hepatotoxicity
6. Cardiotoxicity/hERG inhibition
7. Carcinogenicity
8. Endocrine system disruption
9. Eye Irritation
10. Eye Corrosion
11. Mouse / Rat / Rabbit / Guinea Pig Intraperitoneal LD50
12. Mouse / Rat / Rabbit / Guinea Pig Intraperitoneal LDLo
13. Mouse / Rat / Rabbit / Guinea Pig Intravenous LD50
14. Mouse / Rat / Rabbit / Guinea Pig Intravenous LDLo
15. Mouse / Rat / Rabbit / Guinea Pig Oral LD50 16. Mouse / Rat / Rabbit / Guinea Pig Oral LDLo
 17. Mouse / Rat / Rabbit / Guinea Pig Subcutaneous LD50
 18. Mouse / Rat / Rabbit / Guinea Pig Subcutaneous LDLo
 19. Mouse / Rat / Rabbit / Guinea Pig Skin LD50
 20. Mouse / Rat / Rabbit / Guinea Pig Skin LDLo
 21. Ames test /
 22. Bioconcentration factor
 23. 40 hour Tetrahymena pyriformis IGC50
 24. 48 hour Daphnia magna LC50
 25. 96 hour Fathead Minnow LC50
 26. Mouse / Rat Carcinogenic potency TD50
 27. Mouse / Rat
 28. Класс опасности по острой токсичности для водной среды / при
 попадании на кожу / при вдыхании / при проглатывании
 29. Класс опасности по хронической токсичности для водной среды
 30. Класс опасности по репродуктивной токсичности
 31. Классы опасности по раздражению глаз/кожи
 32. Класс опасности по мутагенности
 33. Класс опасности по канцерогенности
 34. Класс опасности по избирательной токсичности при
 однократном/многократном введении

</details>

Из этих метрик мы выберем наиболее частовстречающиеся и будем использовать их в качестве таргета

## Генерация признаков

Способов описания химических молекул очень много, но изначально мы рассмотрим 6 основных доменов признаков:
- RDKit2D (физико-химические дискрипторы)
- MACCS
- MorganFingerprints
- Mol2Vec
- Graph2Vec
- RoBERTa

Первые 3 стандарт индустрии и реализованы в библиотеке RDKIT. Вторые 3 это нейросетевые подходы для генерации эмбедингов описания структуры молекулы.

В первых итерациях будем пользваться первым доменом признаков (RDKit2D) так как:
- они прямо интерпритируемы
- их нет так много (порядка 200), что лешит нас проклятия размерности
- они относительно легко считаются из коробки

По мере накопления и улучшения знаний о наших данных мы будем утяжелять модели добавлением новых доменов признаков






