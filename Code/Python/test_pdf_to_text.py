"""Run with: python -m unittest discover -s Code/Python -v"""


from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pdf_to_text


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "output"
        self.addCleanup(patch.stopall)
        patch("sys.stdout", new_callable=StringIO).start()
        patch("sys.stderr", new_callable=StringIO).start()

    def test_multiple_pdfs_and_output_creation(self):
        for name in ["one.pdf", "two.PDF", "ignore.txt"]:
            (self.source / name).touch()
        (self.source / "nested").mkdir()
        (self.source / "nested" / "ignored.pdf").touch()
        with patch.object(pdf_to_text, "extract_text", return_value="Hello café\n") as extract:
            self.assertEqual(pdf_to_text.convert_folder(self.source, self.output), 0)
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), ["one.txt", "two.txt"])
        self.assertEqual((self.output / "one.txt").read_text(encoding="utf-8"), "Hello café\n")

    def test_existing_output_is_preserved(self):
        (self.source / "one.pdf").touch()
        self.output.mkdir()
        (self.output / "one.txt").write_text("keep", encoding="utf-8")
        with patch.object(pdf_to_text, "extract_text") as extract:
            self.assertEqual(pdf_to_text.convert_folder(self.source, self.output), 1)
            extract.assert_not_called()
        self.assertEqual((self.output / "one.txt").read_text(), "keep")

    def test_failure_does_not_stop_other_files(self):
        (self.source / "a.pdf").touch()
        (self.source / "b.pdf").touch()
        with patch.object(pdf_to_text, "extract_text", side_effect=[ValueError("bad PDF"), "valid\n"]):
            self.assertEqual(pdf_to_text.convert_folder(self.source, self.output), 1)
        self.assertFalse((self.output / "a.txt").exists())
        self.assertEqual((self.output / "b.txt").read_text(), "valid\n")

    def test_missing_or_empty_input(self):
        self.assertEqual(pdf_to_text.convert_folder(self.root / "missing", self.output), 1)
        self.assertEqual(pdf_to_text.convert_folder(self.source, self.output), 1)
        self.assertFalse(self.output.exists())

    def test_cli_requires_both_folders(self):
        with self.assertRaises(SystemExit) as error:
            pdf_to_text.main(["--input", str(self.source)])
        self.assertEqual(error.exception.code, 2)


class ExtractionTests(unittest.TestCase):
    def test_real_pdf_text_and_blank_pdf(self):
        # These fixtures are generated locally; no downloaded PDFs are needed.
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "text.pdf"
            writer = PdfWriter()
            page = writer.add_blank_page(width=200, height=200)
            font = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
            })
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 20 100 Td (Hello world) Tj ET")
            page[NameObject("/Contents")] = stream
            writer.write(source)
            self.assertEqual(pdf_to_text.extract_text(source).strip(), "Hello world")

            blank = PdfWriter()
            blank.add_blank_page(width=200, height=200)
            blank.write(source)
            with self.assertRaisesRegex(ValueError, "OCR"):
                pdf_to_text.extract_text(source)

    def test_password_protected_pdf(self):
        from pypdf import PdfWriter

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "encrypted.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            writer.encrypt("secret")
            writer.write(source)
            with self.assertRaisesRegex(ValueError, "password-protected"):
                pdf_to_text.extract_text(source)


if __name__ == "__main__":
    unittest.main()
