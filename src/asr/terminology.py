"""CTC context-graph terminology recognition for arbitrary audio."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import soundfile as sf

from src.catalog import ProductCatalog
from src.context_graph import ContextGraphRecognizer
from src.models.asr import ProductMention
from src.models.configs import ContextGraphConfig, TerminologyConfig
from src.nemo_runtime import quiet_nemo_transcribe, silence_nemo_configuration_logs


class CtcTerminologyRecognizer:
    """Detect catalog terms acoustically without ASR fine-tuning."""

    def __init__(self, config: TerminologyConfig):
        self.config = config
        self._model = None
        self._recognizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.config.catalog_path.is_file():
            raise FileNotFoundError(self.config.catalog_path)
        if not self.config.model_path.is_file():
            raise FileNotFoundError(self.config.model_path)
        silence_nemo_configuration_logs()
        import nemo.collections.asr as nemo_asr

        self._model = nemo_asr.models.ASRModel.restore_from(
            str(self.config.model_path), map_location=self.config.device
        ).eval()
        self._recognizer = ContextGraphRecognizer(
            self._model,
            ProductCatalog.load(self.config.catalog_path),
            ContextGraphConfig(
                context_weight=self.config.context_weight,
                keyword_threshold=self.config.keyword_threshold,
                ctc_alignment_weight=self.config.ctc_alignment_weight,
            ),
        )

    def recognize(self, audio_path: Path) -> list[ProductMention]:
        self._ensure_loaded()
        with quiet_nemo_transcribe():
            hypothesis = self._model.transcribe(
                [str(audio_path)], batch_size=1, return_hypotheses=True, verbose=False
            )[0]
        alignment = hypothesis.alignments
        if hasattr(alignment, "cpu"):
            alignment = alignment.cpu().numpy()
        return self._recognizer.recognize(
            np.asarray(alignment), 0.0, float(sf.info(audio_path).duration)
        )

    def unload(self) -> None:
        self._model = None
        self._recognizer = None
        gc.collect()
