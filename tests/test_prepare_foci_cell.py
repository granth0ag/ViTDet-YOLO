import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from prepare_foci import convert
from prepare_foci_cell import prepare


class CellPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.output = self.root / "prepared"
        (self.raw / "cell_Pic1").mkdir(parents=True)
        (self.raw / "Xml_cell").mkdir()
        for i in range(6):
            Image.new("RGB", (100, 80), "white").save(self.raw / "cell_Pic1" / f"{i}.jpg")
            (self.raw / "Xml_cell" / f"{i}.xml").write_text(
                '<annotation><size><width>100</width><height>80</height></size>'
                '<object><name>cell</name><difficult>1</difficult><bndbox>'
                '<xmin>1</xmin><ymin>1</ymin><xmax>100</xmax><ymax>80</ymax>'
                '</bndbox></object></annotation>')

    def run_prepare(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(self.raw, self.output, val_size=1, **kwargs)

    def test_conversion_keeps_difficult_boxes_and_reuses_identical_split(self):
        manifest = self.run_prepare()
        self.assertEqual(manifest["counts"], {"train": {"images": 5, "boxes": 5},
                                            "val": {"images": 1, "boxes": 1}})
        self.assertFalse(set(manifest["splits"]["train"]) & set(manifest["splits"]["val"]))
        for record in manifest["files"].values():
            self.assertEqual(record["labels"], "0 0.500000 0.500000 1.000000 1.000000\n")
        self.assertEqual(self.run_prepare(), manifest)
        marker = (self.output / ".done").read_bytes()
        with self.assertRaises(ValueError):
            self.run_prepare(seed=43)
        self.assertEqual((self.output / ".done").read_bytes(), marker)

    def test_bad_pairing_dimensions_and_boxes_fail_before_output(self):
        xml = self.raw / "Xml_cell/0.xml"
        original = xml.read_text()
        xml.unlink()
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.assertFalse(self.output.exists())
        for bad in (original.replace('<width>100', '<width>99'),
                    original.replace('<xmax>100', '<xmax>101'),
                    original.replace('<xmin>1', '<xmin>nan')):
            xml.write_text(bad)
            with self.assertRaises(ValueError):
                self.run_prepare()
            self.assertFalse(self.output.exists())

    def test_tampered_prepared_labels_are_detected(self):
        self.run_prepare()
        label = next((self.output / "labels/train").glob('*.txt'))
        label.write_text('')
        with self.assertRaises(ValueError):
            self.run_prepare()

    def test_default_virus_conversion_retains_class_validation(self):
        image, xml = self.raw / "cell_Pic1/0.jpg", self.raw / "Xml_cell/0.xml"
        with self.assertRaises(SystemExit):
            convert(image, xml)
        xml.write_text(xml.read_text().replace('<name>cell', '<name>virus'))
        self.assertEqual(convert(image, xml)[0], ['0 0.500000 0.500000 1.000000 1.000000'])


if __name__ == '__main__':
    unittest.main()
