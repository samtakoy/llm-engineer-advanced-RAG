"""Выгрузка решений по заголовкам: что снято, что оставлено.

Разбор PDF помечает заголовком всё, что набрано крупнее основного текста, и правила
из `normalize.retag_headings` делят такие строки на настоящие заголовки и ложные.

Запуск:
    uv run python -m scripts.heading_report

Пишет два файла в папку нарезки: снятые заголовки и оставленные, в каждом строка
в том виде, в котором она уходит в нарезку.
"""
from pathlib import Path
from typing import List, Tuple

import pymupdf4llm
from pymupdf4llm.ocr import OCRMode

from rag_assistant.config import AppConfig
from rag_assistant.ingest.normalize import HEADING_PATTERN, heading_level

DEMOTED_FILE_NAME = "headings_demoted.txt"
KEPT_FILE_NAME = "headings_kept.txt"


def read_headings(pdf_path: Path) -> List[Tuple[int, str]]:
    """Собирает строки, которые разбор PDF пометил заголовком.

    Аргументы:
        pdf_path: путь к файлу отчёта.

    Возвращает:
        Пары «номер страницы, строка заголовка как её выдал разбор».
    """
    pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks = True,
        show_progress = False,
        use_ocr = OCRMode.NEVER,
    )

    return [
        (page["metadata"]["page_number"], line.strip())
        for page in pages
        for line in page["text"].splitlines()
        if HEADING_PATTERN.match(line.strip())
    ]


def render_decision(file_name: str, page_number: int, line: str) -> str:
    """Описывает одно решение по заголовку.

    Аргументы:
        file_name: имя файла отчёта.
        page_number: номер страницы.
        line: строка заголовка как её выдал разбор.

    Возвращает:
        Запись из адреса строки и строки в том виде, в котором она уходит
        в нарезку.
    """
    heading_text = HEADING_PATTERN.match(line).group("text").strip()
    level = heading_level(heading_text)
    result = heading_text if level is None else f"{'#' * level} {heading_text}"

    return f"# {file_name}, стр. {page_number}\n{result}\n"


def write_report(config: AppConfig) -> Tuple[int, int]:
    """Пишет решения по заголовкам в два файла.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Сколько заголовков снято и сколько оставлено.
    """
    demoted: List[str] = []
    kept: List[str] = []

    for pdf_path in sorted(config.documents_dir.rglob("*.pdf")):
        for page_number, line in read_headings(pdf_path):
            heading_text = HEADING_PATTERN.match(line).group("text").strip()
            target = kept if heading_level(heading_text) is not None else demoted
            target.append(
                render_decision(
                    file_name = pdf_path.name,
                    page_number = page_number,
                    line = line,
                )
            )

    config.parsed_dir.mkdir(parents = True, exist_ok = True)
    (config.parsed_dir / DEMOTED_FILE_NAME).write_text("\n".join(demoted), encoding = "utf-8")
    (config.parsed_dir / KEPT_FILE_NAME).write_text("\n".join(kept), encoding = "utf-8")

    return len(demoted), len(kept)


def main() -> None:
    """Выгружает решения по заголовкам.

    Возвращает:
        None.
    """
    config = AppConfig.from_env()
    demoted, kept = write_report(config)

    print(f"Снято заголовков: {demoted} — {config.parsed_dir / DEMOTED_FILE_NAME}")
    print(f"Оставлено заголовков: {kept} — {config.parsed_dir / KEPT_FILE_NAME}")


if __name__ == "__main__":
    main()
