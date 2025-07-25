# Описание
Проект по обучению CNN модели на базе mobilenetV3 на задаче регрессии ключевых точек лица.
Так же добавлен экспорт в формат onnx для более удобного инференса

# Содержание
- [Функции](#Функции)
- [Датасет](#Датасет)
- [Установка](#Установка)
- [Обучение](#Обучение)
- [Результат работы](#Результат-работы-модели)
- [Демо](#Демо)
- [Экспорт ONNX](#Экспорт-ONNX)

# Функции
- Обучение модели регрессии на базе mobilenetV3 68 ключевых точек лица
- Подготовка датасета
- Экспорт в формат ONNX

# Датасет
В проекте используется [датасет](https://www.kaggle.com/competitions/facial-keypoints-detection/data) 
Функции подготовки датасета находятся в файле datasets.py 

# Установка
1. Клонируйте репозиторий
   ```Powershell
   git clone https://github.com/AntoxaZ18/face-keypoints.git
   cd lpr_detect
   ```
2. Установите вирутальную среду и зависимости через Poetry
   Если хотите чтобы вирутальная среда создалась в папке с проектом
   ```Powershell
   poetry config settings.virtualenvs.in-project true
   ```
   Создайте преднастроенную виртуальную среду
   ```Powershell
   poetry install
   ```
   Активируйте среду при помощи poetry (опционально)
   ```Powershell
   poetry env activate
   ```

# Обучение

Ноутбук обучения находится в train.ipynb


# Результат работы модели

![plot](result.png)

# Демо

Демо [проект](https://github.com/AntoxaZ18/face_keypoints_onnx)  использующий модель

# Экспорт в ONNX
Для экспорта в формат ONNX всех моделей, получившихся в результате ablation study необходимо выполнить
--models путь к обученным моделям в формате torch
--onnx  путь куда будут сохранены экспортированные в ONNX модели

```Powershell
python export_to_onnx.py --model best_model.pth --onnx mobilenet_relu.onnx
```

