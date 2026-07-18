from pathlib import Path
from tempfile import TemporaryDirectory

import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from src.catalog import ProductCatalog
from src.context_graph import ContextGraphRecognizer
from src.models.configs import ContextGraphConfig
from src.nemo_runtime import quiet_nemo_transcribe, silence_nemo_configuration_logs
from src.runtime.resources import release_accelerator_memory


class CtcTerminologyRecognizer:
    """Detect catalog terms acoustically without ASR fine-tuning."""

    def __init__(self, config):
        self.config = config
        self._model = None
        self._recognizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if not self.config.catalog_path.is_file():
            raise FileNotFoundError(self.config.catalog_path)
        if not self.config.model_path.is_file():
            raise FileNotFoundError(self.config.model_path)
        silence_nemo_configuration_logs()
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

    def recognize(self, audio_path, speech_segments=None):
        """Spot terms per VAD segment, preserving offsets in the source audio."""
        self._ensure_loaded()
        if speech_segments is None:
            return self._recognize_file(
                audio_path, 0.0, float(sf.info(audio_path).duration)
            )

        audio = AudioSegment.from_file(audio_path)
        mentions = []
        with TemporaryDirectory(prefix="terminology_segments_") as temporary_directory:
            for index, segment in enumerate(speech_segments):
                start_ms = max(0, round(segment.start * 1_000))
                end_ms = min(len(audio), round(segment.end * 1_000))
                if end_ms <= start_ms:
                    continue
                segment_path = Path(temporary_directory) / f"segment_{index:04d}.wav"
                audio[start_ms:end_ms].export(segment_path, format="wav")
                mentions.extend(
                    self._recognize_file(segment_path, segment.start, segment.end)
                )
        return mentions

    def _recognize_file(self, audio_path, segment_start, segment_end):
        with quiet_nemo_transcribe():
            hypothesis = self._model.transcribe(
                [str(audio_path)], batch_size=1, return_hypotheses=True, verbose=False
            )[0]
        alignment = hypothesis.alignments
        if hasattr(alignment, "cpu"):
            alignment = alignment.cpu().numpy()
        return self._recognizer.recognize(
            np.asarray(alignment), segment_start, segment_end
        )

    def unload(self):
        self._model = None
        self._recognizer = None
        release_accelerator_memory()
