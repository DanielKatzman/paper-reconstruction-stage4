from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_reconstruction_stage2.sources import load_source_page


@unittest.skipUnless(importlib.util.find_spec("fitz"), "PyMuPDF is not installed")
class PdfSourceTests(unittest.TestCase):
    def test_pdf_page_is_rendered_to_configured_size(self) -> None:
        import fitz

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pdf"
            document = fitz.open()
            page = document.new_page(width=595, height=842)
            page.insert_text((72, 72), "Stage 2 PDF source test")
            document.save(path)
            document.close()

            source = load_source_page(
                [path],
                width=240,
                height=320,
                rng=np.random.default_rng(3),
                puzzle_index=0,
            )
            self.assertEqual(source.image.size, (240, 320))
            self.assertEqual(source.page_index, 0)
            self.assertEqual(source.origin, str(path.resolve()))


if __name__ == "__main__":
    unittest.main()

