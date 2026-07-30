"""Скачивает отчёты эмитента с сервера раскрытия информации в documents/umg.

Запуск: uv run src/scripts/fetch_reports.py
"""
import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

# Страница эмитента МКПАО «ЮМГ». Портал отдаёт файл только с этим Referer.
PORTAL_PAGE = "https://www.e-disclosure.ru/portal/files.aspx?id=39022&type=5"
FILE_URL = "https://www.e-disclosure.ru/portal/FileLoad.ashx?Fileid={file_id}"

TARGET_DIR = Path(__file__).resolve().parents[2] / "documents" / "umg"

# Отчёты эмитента: период и идентификатор файла на портале.
REPORT_FILE_IDS = {
    "12 мес. 2025": 1928001,
    "12 мес. 2024": 1878741,
}


def download_archive(file_id: int) -> bytes:
    """Скачивает архив с отчётом.

    Аргументы:
        file_id: идентификатор файла на портале.

    Возвращает:
        Содержимое zip-архива.
    """
    request = Request(
        url = FILE_URL.format(file_id = file_id),
        headers = {"Referer": PORTAL_PAGE},
    )

    with urlopen(request, timeout = 60) as response:
        return response.read()


def decode_entry_name(entry: zipfile.ZipInfo) -> str:
    """Читает имя файла внутри архива.

    Кириллические имена портал записывает в cp866 без флага UTF-8,
    и zipfile отдаёт их как cp437.

    Аргументы:
        entry: запись архива.

    Возвращает:
        Имя файла.
    """
    is_utf8 = bool(entry.flag_bits & 0x800)

    return entry.filename if is_utf8 else entry.filename.encode("cp437").decode("cp866")


def extract_pdf(archive: bytes, target_dir: Path) -> Path:
    """Распаковывает PDF из архива.

    Аргументы:
        archive: содержимое zip-архива.
        target_dir: папка назначения.

    Возвращает:
        Путь к распакованному файлу.
    """
    with zipfile.ZipFile(io.BytesIO(archive)) as archive_file:
        entry = archive_file.infolist()[0]
        target_path = target_dir / decode_entry_name(entry)
        target_path.write_bytes(archive_file.read(entry))

    return target_path


def main() -> None:
    """Скачивает и распаковывает все отчёты.

    Возвращает:
        None.
    """
    TARGET_DIR.mkdir(parents = True, exist_ok = True)

    for period, file_id in REPORT_FILE_IDS.items():
        path = extract_pdf(
            archive = download_archive(file_id),
            target_dir = TARGET_DIR,
        )
        print(f"{period}: {path.name} ({path.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
