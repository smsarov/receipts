"""Local HTTP adapter for the receipt recognition model."""
from contextlib import asynccontextmanager
import asyncio
import logging
from pathlib import Path
import sys
import tempfile
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ml' / 'scripts'))
MAX_IMAGE_BYTES = 15 * 1024 * 1024
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    app.state.model_lock = asyncio.Lock()
    yield


app = FastAPI(title='Receipts local API', lifespan=lifespan)


@app.get('/api/health')
def health():
    return {'status': 'ok'}


def run_model(path: Path, app: FastAPI):
    if app.state.predictor is None:
        from receipt_predictor import ReceiptPredictor
        app.state.predictor = ReceiptPredictor()
    return app.state.predictor(path)


@app.post('/api/recognize')
async def recognize(file: Annotated[UploadFile, File()]):
    try:
        data = await file.read(MAX_IMAGE_BYTES + 1)
    finally:
        await file.close()
    if not data:
        raise HTTPException(status_code=400, detail='Файл пустой.')
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail='Размер фотографии не должен превышать 15 МБ.')
    try:
        from io import BytesIO
        with Image.open(BytesIO(data)) as image:
            image.verify()
            if image.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise ValueError('unsupported image format')
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail='Загрузите корректное изображение JPEG, PNG или WebP.')
    try:
        with tempfile.TemporaryDirectory(prefix='receipts-') as directory:
            path = Path(directory) / 'receipt.png'
            with Image.open(BytesIO(data)) as image:
                ImageOps.exif_transpose(image).convert('RGB').save(path, format='PNG')
            async with app.state.model_lock:
                return await run_in_threadpool(run_model, path, app)
    except Exception:
        log.exception('Receipt recognition failed')
        raise HTTPException(status_code=500, detail='Не удалось распознать чек. Проверьте, что модель и её зависимости установлены.')
