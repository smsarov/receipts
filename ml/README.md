# ML experiments

The experiments for the project live here. The current probe checks only the original, un-fine-tuned `naver-clova-ix/donut-base` checkpoint and passes its pretraining task prompts to generation.

## Local environment

Create an environment and install the dependencies:

```bash
python3 -m venv ml/.venv
ml/.venv/bin/python -m pip install -r ml/requirements.txt
```

Download the public dataset annotations and the selected checkpoint into the paths expected by the scripts:

```bash
hf download cdek-ocr/receipt-ocr-ru --repo-type dataset --local-dir ml/data/raw/receipt-ocr-ru --include 'annotations/**' --include 'images/**'
hf download naver-clova-ix/donut-base --local-dir ml/models/donut-base
```

Download the image files referenced in the JSONL annotations under `ml/data/raw/receipt-ocr-ru/` before running inference. Hugging Face may rate-limit large multi-file downloads; the smoke script reports samples whose images are missing.

Inspect the local annotations and run the selected model:

```bash
ml/.venv/bin/python ml/scripts/inspect_receipt_dataset.py
ml/.venv/bin/python ml/scripts/run_donut_base.py
```

The scripts write the dataset summary and prompted raw predictions to `ml/reports/`. The file `donut_base_unprompted_pipeline_diagnostic.jsonl` is retained only to document an earlier pipeline call made without a task prompt; those empty strings are not a valid quality result. Dataset files and model weights are excluded from Git by `ml/.gitignore`.
