from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from face_similarity import clip_runtime
from face_similarity.clip_embedding import get_clip_model


class ClipRuntimeTests(unittest.TestCase):
    def test_configure_clip_cache_sets_d_drive_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            clip_runtime.configure_clip_cache()

            self.assertEqual(os.environ["HF_HOME"], str(clip_runtime.HF_CACHE_DIR))
            self.assertEqual(os.environ["HUGGINGFACE_HUB_CACHE"], str(clip_runtime.HF_CACHE_DIR / "hub"))
            self.assertEqual(os.environ["TRANSFORMERS_CACHE"], str(clip_runtime.HF_CACHE_DIR / "transformers"))
            self.assertEqual(os.environ["TORCH_HOME"], str(clip_runtime.HF_CACHE_DIR / "torch"))

    def test_knn_method_does_not_reexec(self) -> None:
        with patch("os.execv") as execv:
            clip_runtime.ensure_clip_runtime("knn")

        execv.assert_not_called()

    def test_multimodal_method_reexecs_into_clip_venv(self) -> None:
        with patch.object(sys, "executable", r"C:\repo\.venv\Scripts\python.exe"):
            with patch.object(sys, "argv", ["bumble_phone_auto.py", "--loop", "--delay", "3"]):
                with patch("face_similarity.clip_runtime._same_path", return_value=False):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("os.execv") as execv:
                            clip_runtime.ensure_clip_runtime("face_biased")

        execv.assert_called_once_with(
            str(clip_runtime.CLIP_VENV_PYTHON),
            [str(clip_runtime.CLIP_VENV_PYTHON), "bumble_phone_auto.py", "--loop", "--delay", "3"],
        )

    def test_clip_model_uses_local_cache_by_default(self) -> None:
        torch = SimpleNamespace()
        processor_class = SimpleNamespace(from_pretrained=Mock(return_value=object()))
        model = Mock()
        model.to.return_value = model
        model_class = SimpleNamespace(from_pretrained=Mock(return_value=model))

        with (
            patch(
                "face_similarity.clip_embedding.import_clip_dependencies",
                return_value=(torch, model_class, processor_class),
            ),
            patch("face_similarity.clip_embedding.select_device", return_value="cpu"),
            patch.dict(os.environ, {}, clear=True),
        ):
            get_clip_model.cache_clear()
            get_clip_model("test-model", "cpu")

        processor_class.from_pretrained.assert_called_once_with("test-model", local_files_only=True)
        model_class.from_pretrained.assert_called_once_with("test-model", local_files_only=True)


if __name__ == "__main__":
    unittest.main()
