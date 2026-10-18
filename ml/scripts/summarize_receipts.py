"""Create the experiment summary and weekly report from completed run artifacts."""

from __future__ import annotations

import json

from receipt_common import ML, REPORTS, read_jsonl, write_json


def percent(value: float) -> str:
    return f"{100 * value:.1f}%".replace(".", ",")


def main() -> None:
    def load(name: str):
        return json.loads((REPORTS / name).read_text(encoding="utf-8"))
    config = load("config.json")
    complete = load("training_complete.json")
    best = load("best_checkpoint.json")
    baseline = load("baseline_fixed_test_metrics.json")
    fixed = load("finetuned_fixed_test_metrics.json")
    test = load("finetuned_test_metrics.json")
    initial = load("initial_validation.json")
    tokenizer = load("tokenizer.json")
    preparation = json.loads((ML / "reports/data_preparation.json").read_text(encoding="utf-8"))
    full_rows = read_jsonl(REPORTS / "finetuned_test.jsonl")
    base_rows = read_jsonl(REPORTS / "baseline_fixed_test.jsonl")
    expected_fixed = set(preparation["fixed_test_ids"])
    assert complete["epochs_completed"] == config["epochs"]
    assert {row["image_id"] for row in base_rows} == expected_fixed
    assert {row["image_id"] for row in full_rows if row["fixed_comparison_subset"]} == expected_fixed
    assert test["receipts"] == preparation["splits"]["test"]["prepared"] == len(full_rows)
    summary = {"training": complete, "best_checkpoint": best, "initial_validation_loss": initial["loss"],
               "baseline_fixed_test": baseline, "finetuned_fixed_test": fixed, "finetuned_full_test": test}
    write_json(REPORTS / "summary.json", summary)
    rows = []
    for label, metrics in (("Исходная Donut: фиксированная выборка", baseline),
                           ("После дообучения: та же выборка", fixed),
                           ("После дообучения: весь подготовленный test", test)):
        rows.append(f"| {label} | {metrics['receipts']} | {percent(metrics['valid_json_rate'])} | "
                    f"{percent(metrics['name']['f1'])} | {percent(metrics['sum']['f1'])} | {percent(metrics['pair']['f1'])} |")
    table = "\n".join([
        "| Модель и выборка | Чеков | Валидный JSON | F1 названий | F1 сумм | F1 пар |",
        "|---|---:|---:|---:|---:|---:|", *rows])
    detail = (f"# Результаты эксперимента Donut\n\n{table}\n\n"
              f"Обучено эпох: {complete['epochs_completed']}; выбран checkpoint эпохи {best['epoch']}. "
              f"Validation loss: {initial['loss']:.4f} → {best['validation_loss']:.4f}.\n\n"
              f"Полное совпадение чека на test: {percent(test['receipt_exact_match'])}; "
              f"CER извлечённых названий: {test['product_names_cer']:.4f}.\n\n"
              "F1 считается по мультимножествам внутри чека; невалидный JSON даёт пустой список извлечённых позиций. "
              "Определения метрик и ограничения — в [протоколе](protocol.md).\n\n"
              "## Примеры из тестовой выборки\n")
    for result in full_rows[:3]:
        detail += (f"\n### {result['image_id']}\n\nЭталон:\n\n```json\n"
                   f"{json.dumps(result['target'], ensure_ascii=False, indent=2)}\n```\n\n"
                   f"Выход модели (валидный JSON: {result['valid_json']}):\n\n```text\n"
                   f"{result['generated_text']}\n```\n")
    (REPORTS / "summary.md").write_text(detail, encoding="utf-8")
    report = f"""# Отчёт за 12.10–18.10.2026

**Проект:** «Receipts»  
**Команда:** 12  
**Авторы:** Мартынов Александр, Сороколетов Денис

## 1. Цель работы за период

Завершить подготовку данных, дообучить выбранную Donut на русской разметке и оценить распознавание названий и сумм товаров на отложенных чеках.

## 2. Что было сделано

- Завершили загрузку изображений `validation`. Убрали дубликаты и сформировали JSON-цели только с названиями и суммами товаров. Для обучения подготовили {config['train_examples']} чеков, для validation — {config['validation_examples']}, для test — {test['receipts']}.
- Расширили исходный токенизатор для русского текста: добавили {tokenizer['added_character_tokens']} символа и префикс задачи. Проверили восстановление обучающих целевых строк без неизвестных токенов.
- Проверили исходную `naver-clova-ix/donut-base` на фиксированной выборке из {baseline['receipts']} тестовых чеков, затем провели {complete['epochs_completed']} эпох дообучения её декодера. Визуальный энкодер оставили замороженным. Лучший checkpoint выбрали по validation loss; тестовые данные в обучении и выборе checkpoint не использовались.
- Оценили обученную модель на той же фиксированной выборке и на всём подготовленном test. Сохранили checkpoint, скрипт распознавания отдельного чека, сырые предсказания и метрики в `ml/`.

## 3. Результаты проверки модели

{table}

F1 пар учитывает одновременное совпадение названия и суммы товара. Выбран checkpoint эпохи {best['epoch']}; validation loss снизилась с {initial['loss']:.4f} до {best['validation_loss']:.4f}.

Получен первый обученный checkpoint для целевой задачи. Суммы распознаются лучше названий; основная задача дальнейшей работы — повысить точность названий товаров.

## 4. Текущий статус

| Компонент | Статус |
|---|---|
| Подготовка данных | Завершена для первого эксперимента |
| Дообучение основной Donut | Выполнено, лучший checkpoint сохранён в `ml/models/donut-receipts/best/` |
| Оценка на отложенных данных | Выполнена; предсказания и метрики в `ml/reports/finetune_2026-10-18/` |
| Запуск на отдельном чеке | Скрипт `ml/scripts/predict_receipt.py` проверен на тестовом изображении |

## 5. План работы на следующий период

- Разобрать ошибки распознавания названий и сумм на `validation`.
- Подобрать параметры дальнейшего дообучения Donut.
- Повторить оценку качества и проверить готовность модели к подключению к приложению.
"""
    destination = ML.parent / "course/12/12.10-18.10.md"
    destination.write_text(report, encoding="utf-8")
    print(table)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
