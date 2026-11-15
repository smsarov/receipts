import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArrowRight, CircleAlert, FileImage, ImagePlus, LoaderCircle, Maximize2, ReceiptText, UploadCloud, X } from 'lucide-react';
import './style.css';

const MAX_SIZE = 15 * 1024 * 1024;
const acceptTypes = ['image/jpeg', 'image/png', 'image/webp'];
const money = (value) => value == null || value === '' ? '—' : `${String(value).replace('.', ',')} ₽`;

function App() {
  const input = useRef(null);
  const request = useRef(null);
  const dialog = useRef(null);
  const [photoOpen, setPhotoOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (photoOpen && !element.open) element.showModal();
    if (!photoOpen && element.open) element.close();
  }, [photoOpen]);

  function selectFile(next) {
    if (!next) return;
    if (!acceptTypes.includes(next.type)) { setError('Подойдёт изображение в формате JPG, PNG или WebP.'); return; }
    if (next.size > MAX_SIZE) { setError('Размер фотографии не должен превышать 15 МБ.'); return; }
    request.current?.abort();
    setPhotoOpen(false); setFile(next); setResult(null); setError(''); setLoading(false);
  }
  function reset() {
    request.current?.abort();
    setPhotoOpen(false); setFile(null); setResult(null); setError(''); setLoading(false);
    if (input.current) input.current.value = '';
  }
  async function recognize() {
    if (!file || loading) return;
    const controller = new AbortController();
    request.current = controller;
    setLoading(true); setError(''); setResult(null);
    const form = new FormData(); form.append('file', file);
    try {
      const response = await fetch('/api/recognize', { method: 'POST', body: form, signal: controller.signal });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Не удалось обработать фотографию.');
      if (!Array.isArray(data.items)) throw new Error('Сервер вернул неожиданный ответ.');
      setResult(data);
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message === 'Failed to fetch' ? 'Не удалось связаться с сервером. Проверьте, что бэкенд запущен.' : err.message);
    } finally {
      if (request.current === controller) { request.current = null; setLoading(false); }
    }
  }
  const items = result?.items ?? [];
  return <div className="app-shell">
    <main className="main">
      <section className="heading"><h1>Распознавание чека</h1></section>
      <div className="workspace">
        <section className="panel upload-panel"><div className="panel-heading"><div><h2>Загрузите чек</h2></div><span className="small-icon"><ImagePlus size={19}/></span></div>
          <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(e) => {selectFile(e.target.files?.[0]); e.target.value = '';}} />
          {!file ? <button type="button" className={`dropzone ${dragging ? 'dragging' : ''}`} onClick={() => input.current?.click()} onDragOver={e => {e.preventDefault(); setDragging(true);}} onDragLeave={() => setDragging(false)} onDrop={e => {e.preventDefault(); setDragging(false); selectFile(e.dataTransfer.files?.[0]);}}><span className="upload-illustration"><UploadCloud size={28} strokeWidth={1.5}/></span><strong>Перетащите фото сюда</strong><span>или <u>выберите файл</u> на устройстве</span><span className="file-types">JPG, PNG, WEBP <i/> ДО 15 МБ</span></button> : <div className="preview-wrap"><div className="preview-image"><button type="button" className="preview-zoom" onClick={() => setPhotoOpen(true)} aria-label="Увеличить фотографию чека"><img src={preview} alt="Выбранная фотография чека"/><span className="zoom-label"><Maximize2 size={14}/> Увеличить</span></button><button className="remove" type="button" onClick={reset} aria-label="Удалить фотографию"><X size={17}/></button></div><div className="file-info"><span className="file-icon"><FileImage size={17}/></span><span className="filename"><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toFixed(2)} МБ · Фото загружено</small></span><button type="button" className="change" onClick={() => input.current?.click()}>Заменить</button></div></div>}
          {error && <div className="error" role="alert"><CircleAlert size={17}/><span>{error}</span></div>}
          <button className="primary" type="button" disabled={!file || loading} onClick={recognize}>{loading ? <><LoaderCircle className="spin" size={18}/> Распознаём чек...</> : <>Распознать чек <ArrowRight size={18}/></>}</button>
        </section>
        <section className="panel result-panel"><div className="panel-heading result-heading"><div><h2>Позиции в чеке</h2></div></div>
          {!result ? <div className="empty-state"><span className="empty-circle"><ReceiptText size={29}/></span><h3>{loading ? 'Читаем ваш чек' : 'Здесь появится результат'}</h3><p>{loading ? 'Распознавание может занять некоторое время, особенно при первом запуске.' : 'Загрузите фотографию и нажмите «Распознать чек». Найденные позиции появятся здесь.'}</p></div> : items.length === 0 ? <div className="empty-state no-items"><span className="empty-circle"><ReceiptText size={29}/></span><h3>Позиции не найдены</h3><p>Попробуйте другую фотографию: расположите чек ровно и убедитесь, что текст хорошо виден.</p></div> : <div className="results"><div className="table-scroll"><table><thead><tr><th>ТОВАР</th><th>КОЛ-ВО</th><th>ЦЕНА</th><th>СУММА</th></tr></thead><tbody>{items.map((item, index) => <tr key={index}><td><span className="item-name">{item.name || 'Без названия'}</span>{item.name_source === 'catalog' && item.raw_name && item.raw_name.toLocaleLowerCase() !== item.name?.toLocaleLowerCase() && <span className="item-raw">В чеке: {item.raw_name}</span>}</td><td>{item.quantity ?? '—'}{item.quantity_inferred && <sup title="Количество предположено">*</sup>}</td><td>{money(item.price)}</td><td className="sum-cell">{money(item.sum)}{item.sum_inferred && <sup title="Сумма рассчитана">*</sup>}</td></tr>)}</tbody></table></div></div>}
        </section>
      </div>
    </main>
    <dialog ref={dialog} className="photo-dialog" aria-labelledby="photo-title" onClose={() => setPhotoOpen(false)} onClick={(event) => { if (event.target === event.currentTarget) setPhotoOpen(false); }}>
      <div className="photo-modal">
        <div className="photo-modal-header"><h2 id="photo-title">Фотография чека</h2><button type="button" onClick={() => setPhotoOpen(false)} aria-label="Закрыть просмотр"><X size={21}/></button></div>
        {preview && <img src={preview} alt="Увеличенная фотография чека" />}
      </div>
    </dialog>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
