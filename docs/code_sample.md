# Извлечение и прикрепление номеров страниц при парсинге PDF

Номер страницы проставляется в три шага.

## 1. Читаем PDF постранично

[src/rag_assistant/ingest/loader.py:98](../src/rag_assistant/ingest/loader.py#L98)

```python
pages = pymupdf4llm.to_markdown(
    str(pdf_path),
    page_chunks = True,
    show_progress = False,
    use_ocr = OCRMode.NEVER,
)

return [
    Document(
        doc_id = f"{pdf_path.name}:{page['metadata']['page_number']}",
        text = normalize_page(page["text"]),
        metadata = {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "page_label": str(page["metadata"]["page_number"]),
            **DOCUMENT_METADATA.get(pdf_path.name, {}),
        },
    )
    for page in pages
]
```

## 2. Склеиваем страницы в файл, запоминая границы в символах

[src/rag_assistant/ingest/page_map.py:35](../src/rag_assistant/ingest/page_map.py#L35)

```python
for page in pages:
    text = page.text
    spans.append(
        PageSpan(
            number = int(page.metadata["page_label"]),
            start = offset,
            end = offset + len(text),
        )
    )
    parts.append(text)
    offset += len(text) + len(PAGE_JOIN)
```

## 3. После нарезки возвращаем узлу страницу по его смещению

[src/rag_assistant/ingest/page_attribution.py:42](../src/rag_assistant/ingest/page_attribution.py#L42)

```python
content = node.get_content(metadata_mode = MetadataMode.NONE)
bounds = locate(text = text, content = content, cursor = cursors.get(file_name, 0))
start, end = bounds

first = page_at(spans = spans[file_name], offset = start)
last = page_at(spans = spans[file_name], offset = end - 1)
node.metadata[PAGE_LABEL_KEY] = str(first)

if last > first:
    node.metadata[PAGE_END_KEY] = str(last)
```

Узел на двух листах получает диапазон: `page_label` — начало, `page_end` — конец.
Так печатается ссылка `стр. 60-61` в прогонах.

## 4. Прячем номер от эмбеддера, оставляя модели

[src/rag_assistant/ingest/loader.py:37](../src/rag_assistant/ingest/loader.py#L37)

```python
EXCLUDED_FROM_EMBEDDING = ("file_path", "page_label", "page_end", "year")
EXCLUDED_FROM_PROMPT = ("file_path",)
```

Номер страницы в векторе — шум: у всех узлов это одинаковое по форме поле со случайным
для смысла числом. В промпт он идёт, по нему модель ставит ссылку.
