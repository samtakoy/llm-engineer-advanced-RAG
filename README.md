# RAG по годовым отчётам

Ассистент отвечает на вопросы по PDF из папки `documents`, подставляя в ответ имя файла и номер страницы.

Стек: LlamaIndex + ChromaDB, модель отвечает через локальный OpenAI-совместимый сервер (LM Studio).

## Что реализовано (коротко)

- Задать вопрос из терминала в разных режимах
- Прогнать список контрольных вопросов на стенде в разных режимах и снять метрики

Техники:
- Наивный RAG
- Hybrid RAG
- Реранкер (кросс-энкодер)
- MMR реранкинг

## Отчет

[docs/report.md](docs/report.md)

## Документы

Отчёты берутся с сервера раскрытия информации, страница МКПАО «ЮМГ»:
https://www.e-disclosure.ru/portal/files.aspx?id=39022&type=5

Прямые ссылки на отчёты эмитента:

| Период | Ссылка |
|---|---|
| 12 мес. 2025 | https://www.e-disclosure.ru/portal/FileLoad.ashx?Fileid=1928001 |
| 12 мес. 2024 | https://www.e-disclosure.ru/portal/FileLoad.ashx?Fileid=1878741 |

По ссылке скачивается zip с одним PDF внутри. Скачивает и распаковывает в `documents/umg/` скрипт:

```bash
uv run src/scripts/fetch_reports.py
```

```
12 мес. 2025: Отчет эмитента МКПАО ЮМГ 12м2025.pdf (1422 КБ)
12 мес. 2024: FY2024_Issuer's report.pdf (638 КБ)
```

Новый отчёт добавляется одной строкой в словарь `REPORT_FILE_IDS` — идентификатор виден
в ссылке на файл на странице эмитента.

После добавления отчётов нужен `uv run main.py reindex`.

Pdf отчетов добавлены для наглядности в репозиторий:

- [FY2024_Issuer's report.pdf](documents/umg/FY2024_Issuer's%20report.pdf)
- [Отчет эмитента МКПАО ЮМГ 12м2025.pdf](documents/umg/Отчет%20эмитента%20МКПАО%20ЮМГ%2012м2025.pdf)

## Запуск

```bash
uv sync
cp .env.example .env      # при необходимости поправить модель и адрес сервера
uv run main.py ask "Какая выручка была в 2024 году?"
```

Перестроить индекс после смены документов или настроек нарезки:

```bash
uv run main.py reindex
```

## Фильтрация по метаданным

Поиск можно ограничить разметкой документов — сейчас это год отчёта.
Отбор идёт до поиска соседей, поэтому выдача набирается уже внутри отобранных файлов.

```bash
uv run main.py ask "Сколько было визитов?" --tag "year:2025"
uv run main.py ask "Сколько было визитов?" --tag "year:2024|year:2025"
```

Тег — это «поле:значение» из метаданных документа. Одинаковые ключи расширяют выбор
по полю, разные ключи сужают выдачу каждый своим условием: `"year:2025|section:риски"`.
Разметка задаётся в `DOCUMENT_METADATA` в `src/rag_assistant/ingest/loader.py`, её правка
требует `reindex`: метаданные лежат в Chroma рядом с векторами.

В контрольных вопросах теги проставлены полем `tags` и применяются только с флагом `--filters`.

## Режимы 

```
Наивный раг:
uv run main.py ask "Сколько было визитов?"

Подключить гибридный поиск:
uv run main.py ask "Сколько было визитов?" --hybrid

Подключить реранкинг:
uv run main.py ask "Сколько было визитов?" --hybrid --rerank

Подключить mmr после реранкинга:
uv run main.py ask "Сколько было визитов?" --hybrid --rerank --mmr2
```

## Прогон контрольных вопросов со снятием метрик

Режима два, задаются первым аргументом:

- `retrieval` (по умолчанию) — только поиск. Меряет, попали ли в выдачу эталонные страницы
  и подстроки. Модель не вызывается, поэтому прогон быстрый.
- `full` — то же плюс ответ модели: добавляются метрики по тексту ответа и ссылкам,
  с `--judge` оценка ответов моделью-судьёй.

```bash
uv run python -m eval.run retrieval                              # только поиск, без модели
uv run python -m eval.run retrieval --filters --name filters     # с фильтром по тегам вопроса
uv run python -m eval.run retrieval --rerank --name rerank       # с cross-encoder из RERANK_MODEL
uv run python -m eval.run retrieval --hybrid --name hybrid       # вектор плюс лексический BM25
uv run python -m eval.run full --judge --name full               # ответы модели и оценка судьёй
```

Флаги складываются, снимки и метрики пишутся в `docs/eval` под именем из `--name`

## Что уходит в индекс

Посмотреть нарезку до векторизации:

```bash
uv run main.py parse
```

В `parsed/` появится по markdown-файлу на документ: узлы в порядке чтения, у каждого
все метаданные и текст целиком.

```markdown
# FY2024_Issuer's report.pdf

Узлов: 168 · страниц: 69 · символов: 276143

В эмбеддинг уходят метаданные: page_label, file_name, file_type
В промпт модели уходят: page_label, file_name, file_type

## Узел 13 · стр. 8 · 1951 симв.

- page_label: 8
- file_name: FY2024_Issuer's report.pdf
...
```

Также появится очищенный маркдаун файл на каждый отчет до нарезки:
```
FY2024_Issuer's report_markdown.md
```

Пример:

- [FY2024_Issuer's report.md](parsed/FY2024_Issuer's%20report.md)
- [FY2024_Issuer's report_markdown.md](parsed/FY2024_Issuer's%20report_markdown.md)

## Как устроено

```
documents/           исходные PDF
parsed/              нарезка в markdown, создаётся командой parse
chroma/              векторы, создаётся автоматически
main.py              точка входа и сборка пайплайна
src/rag_assistant/
    config.py        настройки из .env
    models.py        LLM, эмбеддер, сплиттер
    ingest/
        loader.py    чтение папки в Document
        normalize.py починка markdown после разбора PDF
        filters.py   отсев страниц без содержания
        chained_parser.py  нарезка по заголовкам с пределом длины
    dump.py          выгрузка нарезки в markdown
    index.py         построение и открытие индекса
    metadata_filters.py  теги «ключ:значение» -> фильтр поиска
    lexical.py       поиск по словам, индекс BM25 в памяти
    index_signature.py  отпечаток настроек индекса, сверка при открытии
    engine.py        вопрос -> ответ со ссылками
    prompts.py       промпт ответа по контексту
src/eval/            прогон контрольных вопросов и метрики
src/scripts/
    fetch_reports.py скачивание отчётов с портала раскрытия
    heading_report.py  сводка по заголовкам после разбора PDF
```

Пайплайн: `ingest.load_documents` разбирает PDF в markdown через `pymupdf4llm` и отдаёт по одному `Document` на страницу, `Settings.node_parser` режет их на чанки, `index.open_index` векторизует и складывает в ChromaDB, `engine.RagEngine` ищет и просит модель ответить.

Техники поиска включаются флагами.

```bash
uv run main.py ask "вопрос" --rerank --hybrid
uv run python -m eval.run retrieval --rerank --hybrid --name flat_advanced
```
## Настройки env

<details>
<summary>Список параметров</summary>

| Параметр | Что это |
|---|---|
| `LOCAL_BASE_URL` | адрес OpenAI-совместимого сервера |
| `LOCAL_API_KEY` | ключ сервера |
| `LOCAL_MODEL` | модель, которая отвечает |
| `LOCAL_REASONING_EFFORT` | уровень reasoning: low, medium, high. Пусто — выключен |
| `JUDGE_MODEL` | модель-судья в прогоне вопросов. Пусто — та же, что отвечает |
| `LLM_TEMPERATURE` | температура генерации |
| `LLM_MAX_TOKENS` | лимит токенов ответа |
| `LLM_CONTEXT_WINDOW` | размер контекстного окна модели |
| `LLM_TIMEOUT_SECONDS` | таймаут запроса к модели |
| `EMBEDDING_MODEL` | модель эмбеддингов из HuggingFace |
| `RERANK_MODEL` | cross-encoder для флага `--rerank`. Пусто — реранкинга нет |
| `DOCUMENTS_DIR` | папка с исходными PDF |
| `PARSED_DIR` | папка выгрузки нарезки |
| `CHROMA_DIR` | папка векторов ChromaDB |
| `CHROMA_COLLECTION` | имя коллекции векторов |
| `CHUNK_SIZE` | размер чанка в токенах эмбеддера |
| `CHUNK_OVERLAP` | перекрытие соседних чанков в токенах |
| `TOP_K` | сколько фрагментов уходит модели в контекст |
| `CANDIDATE_TOP_K` | сколько фрагментов поиск приносит реранкеру |
| `MMR_THRESHOLD` | вес флагов `--mmr` и `--mmr2`: 1.0 — близость, 0.0 — разнообразие. Пусто — выключен |

</details>
