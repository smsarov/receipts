# ML: извлечение позиций из российских чеков

Основная цель — извлекать по каждой позиции название, количество, цену за единицу и сумму строки:

```json
{"items":[{"name":"Молоко","quantity":"1","price":"89.99","sum":"89.99"}]}
```

Первое дообучение завершено: [результаты и примеры предсказаний](reports/finetune_2026-10-18/summary.md), [отчёт за неделю](../course/12/12.10-18.10.md).

Проверка validation и эксперимент продолжения с learning rate `1e-5` находятся в [отчёте эксперимента](reports/finetune_2026-10-18/low_lr_2026-10-25/). Низкая скорость снизила validation loss, но не улучшила генерацию названий; этот checkpoint не заменяет основной. Модель пока не готова к интеграции: см. [оценку извлечения](reports/item_extraction_status_2026-10-25.md) и [отчёт за 19.10–25.10](../course/12/19.10-25.10.md).

## Окружение

Из корня репозитория:

```bash
python3 -m venv ml/.venv
ml/.venv/bin/python -m pip install -r ml/requirements.txt
```

Модель и исходные данные должны находиться в `ml/models/donut-base/` и `ml/data/raw/receipt-ocr-ru/`:

```bash
ml/.venv/bin/hf download naver-clova-ix/donut-base --local-dir ml/models/donut-base
ml/.venv/bin/hf download cdek-ocr/receipt-ocr-ru --repo-type dataset --local-dir ml/data/raw/receipt-ocr-ru --include 'annotations/**' --include 'images/**'
```

Скрипты автоматически используют Apple MPS, если он доступен, иначе CPU. Для явного выбора устройства у эксперимента и предсказания есть `--device cpu` или `--device mps`. Используются локальные веса; платные API не нужны.

## Подготовка и эксперимент

```bash
# Докачать недостающие изображения, очистить разметку и сформировать JSON-цели.
ml/.venv/bin/python ml/scripts/prepare_receipts.py --download
ml/.venv/bin/python ml/scripts/inspect_receipt_dataset.py

# Сохранить признаки замороженного визуального энкодера.
ml/.venv/bin/python ml/scripts/finetune_receipts.py cache

# Проверить исходный checkpoint на 24 фиксированных тестовых чеках.
ml/.venv/bin/python ml/scripts/finetune_receipts.py baseline

# Дообучить декодер; лучший checkpoint выбирается по validation loss.
ml/.venv/bin/python ml/scripts/finetune_receipts.py train --epochs 5

# Оценить выбранный checkpoint на всём подготовленном test.
ml/.venv/bin/python ml/scripts/finetune_receipts.py evaluate
```

`cache` пропускает готовые файлы и проверяет конфигурацию кеша. Для продолжения обучения после сохранённой эпохи используйте `train --epochs 5 --resume`. Для продолжения генерации с уже записанных результатов — `baseline --resume` или `evaluate --resume` с теми же параметрами. Не меняйте данные и параметры в середине возобновляемого эксперимента.

Для отдельного эксперимента задайте `--run-name`. Флаг `--train-encoder` включает вычислительно дорогую адаптацию визуального энкодера; используйте отдельный запуск и сперва оцените скорость на MPS. Не запускайте такой вариант на большом наборе без оценки времени. Можно продолжить с decoder весов другого checkpoint параметром `--init-checkpoint`; оценивать его следует через `evaluate --checkpoint PATH --run-name NAME --split validation`. Веса и результаты такого запуска сохраняются отдельно от основного checkpoint.

Первый вариант обучения замораживает визуальный энкодер и обучает весь декодер. Изображения вписываются в 960 × 1280 без поворота текста. Исходный Unigram-токенизатор расширяется символами для русского текста с сохранением прежних ID; при загрузке обученного checkpoint нужен `AutoTokenizer(..., use_fast=True)`, поскольку расширенный словарь сохранён в `tokenizer.json`. Скрипты уже учитывают это.

## Предсказание для изображения

После обучения:

```bash
ml/.venv/bin/python ml/scripts/predict_receipt.py path/to/receipt.jpg
```

При корректном JSON скрипт печатает `{"items": [...]}`. Если генерация не соответствует схеме, он выводит сырой текст с `valid_json: false` и завершается с кодом 2. Эталонные данные в предсказании не используются.

## Артефакты

- `data/raw/receipt-ocr-ru/` — исходные JSONL и изображения.
- `data/processed/receipt-ocr-ru/` — очищенные `train`, `validation`, `test` с JSON-целями.
- `data/features/donut-base-1280x960-upright/` — признаки изображений и конфигурация кеша.
- `models/donut-base/` — неизменённый исходный checkpoint.
- `models/donut-receipts/best/` — обученный checkpoint и токенизатор в формате Hugging Face.
- `models/donut-receipts/last.pt` — состояние для возобновления обучения.
- `reports/data_preparation.json` — исключённые записи, хеши файлов и ID фиксированной тестовой подвыборки.
- `reports/finetune_2026-10-18/` — протокол, конфигурация, журнал обучения, сырые предсказания и метрики.
- `reports/item_extraction_status_2026-10-25.md` — Donut и OCR для названия, количества, цены и суммы.
- `reports/finetune_2026-10-18/logs/` — журналы загрузки, подготовки, обучения и проверки.

Данные, кеши и веса исключены из Git через `.gitignore`. Скрипты не изменяют исходную разметку и исходные веса.

Подробности формирования выборок и определения метрик — в [протоколе](reports/finetune_2026-10-18/protocol.md).
