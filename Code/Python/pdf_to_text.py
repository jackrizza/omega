"""Extract embedded text from each PDF in a folder; no OCR is performed."""

import argparse
from importlib import import_module
from pathlib import Path
import sys


def extract_text(source: Path) -> str:
    """Read all pages before creating an output file."""
    from pypdf import PdfReader

    with source.open("rb") as stream:
        reader = PdfReader(stream)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF is password-protected; decrypt it before conversion")
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    if not any(pages):
        raise ValueError("No extractable text; scanned/image-only PDFs require OCR")
    return "\n\n".join(pages).strip() + "\n"


def convert_folder(input_folder: Path, output_folder: Path) -> int:
    """Convert top-level PDFs, returning 1 if any conversion fails, otherwise 0."""
    try:
        if not input_folder.is_dir():
            raise ValueError(f"Input folder does not exist or is not a directory: {input_folder}")
        sources = sorted(
            (path for path in input_folder.iterdir()
             if path.is_file() and path.suffix.lower() == ".pdf"),
            key=lambda path: path.name,
        )
        if not sources:
            raise ValueError(f"No PDF files found in: {input_folder}")
        output_folder.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    converted = 0
    for source in sources:
        destination = output_folder / f"{source.stem}.txt"
        try:
            # This early check avoids parsing a PDF whose output already exists;
            # exclusive creation below also protects against concurrent writers.
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Output already exists: {destination}")
            text = extract_text(source)
            output = destination.open("x", encoding="utf-8", newline="\n")
            try:
                with output:
                    output.write(text)
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            converted += 1
            print(f"Converted: {source.name} -> {destination}")
        except Exception as error:
            # A malformed PDF should not prevent other documents from converting.
            print(f"Failed: {source.name}: {error}", file=sys.stderr)

    failures = len(sources) - converted
    print(f"Finished: {converted} converted, {failures} failed.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, type=Path,
                        help="Folder containing PDFs (non-recursive)")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        help="Folder for UTF-8 .txt files; created if missing")
    args = parser.parse_args(argv)
    try:
        import_module("pypdf")
    except ImportError:
        print("Missing dependency: install with python -m pip install pypdf", file=sys.stderr)
        return 1
    return convert_folder(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
