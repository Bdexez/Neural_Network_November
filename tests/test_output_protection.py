from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from chimera import Chimera, ChimeraConfig
from chimera.cli import main
from chimera.training import save_checkpoint


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Installer l'extra vision pour tester PNG/JPEG")
class OutputProtectionTests(unittest.TestCase):
    def test_commands_preserve_referenced_images_and_input_hardlinks(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "red").save(image)
            original_image = image.read_bytes()
            hardlink, symlink = root / "hardlink.png", root / "symlink.png"
            os.link(image, hardlink)
            symlink.symlink_to(image)
            data = root / "data.jsonl"
            data.write_text(json.dumps(dict(image_path=image.name, class_label=1)))
            checkpoint = root / "model.pt"
            save_checkpoint(checkpoint, Chimera(ChimeraConfig.preset("tiny")))
            for command in ("predict", "embed", "evaluate", "calibrate", "refine-forecast"):
                for destination in (image, hardlink, symlink):
                    with self.subTest(command=command, destination=destination), self.assertRaises(SystemExit):
                        main(["--threads", "2", command, "--checkpoint", str(checkpoint), "--data", str(data),
                              "--output", str(destination)])
                    self.assertEqual(image.read_bytes(), original_image)
                    self.assertTrue(symlink.is_symlink())
            with self.assertRaises(SystemExit):
                main(["--threads", "2", "refine-forecast", "--checkpoint", str(checkpoint), "--data", str(data),
                      "--output", str(root / "out.pt"), "--report", str(hardlink)])
            self.assertEqual(image.read_bytes(), original_image)
            data_link = root / "data-link"
            os.link(data, data_link)
            original_data = data.read_bytes()
            with self.assertRaises(SystemExit):
                main(["--threads", "2", "evaluate", "--checkpoint", str(checkpoint), "--data", str(data),
                      "--output", str(data_link)])
            self.assertEqual(data.read_bytes(), original_data)


if __name__ == "__main__":
    unittest.main()
