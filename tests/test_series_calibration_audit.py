import json
from pathlib import Path
import tempfile
import unittest

from chimera.calibration import file_sha256
from chimera.cli import audit_series_calibration


class SeriesCalibrationAuditTests(unittest.TestCase):
    def test_reference_overlap_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, head, temperature, output = (root / name for name in ('cal.jsonl', 'head.jsonl', 'temp.jsonl', 'output.json'))
            calibration.write_text(json.dumps({'series': [[8]], 'class_label': 0}) + '\n')
            head.write_text(json.dumps({'series': [[1]], 'class_label': 0}) + '\n')
            temperature.write_text(json.dumps({'series': [[1.0]], 'class_label': 1}) + '\n')
            refs = {'head': (head, file_sha256(head)), 'temperature': (temperature, file_sha256(temperature))}
            with self.assertRaisesRegex(ValueError, 'head et temperature'):
                audit_series_calibration(calibration, file_sha256(calibration), refs, output=output)
            with self.assertRaisesRegex(ValueError, 'empreinte'):
                audit_series_calibration(calibration, file_sha256(calibration), {'head': (head, 'a'*64)}, output=output)
            with self.assertRaisesRegex(ValueError, 'absente'):
                audit_series_calibration(calibration, file_sha256(calibration), {'head': (head, None)}, output=output)
            with self.assertRaisesRegex(ValueError, 'a changé'):
                audit_series_calibration(calibration, 'a'*64, {'head': (head, file_sha256(head))}, output=output)
            audit_series_calibration(calibration, file_sha256(calibration), {'head': (head, file_sha256(head))}, output=output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
