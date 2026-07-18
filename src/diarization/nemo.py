import json
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf

from src.diarization.base import BaseDiarizer
from src.models.configs import NemoDiarizationConfig
from src.models.diarization import DiarizationResult, SpeakerTurn


class NemoDiarizer(BaseDiarizer):
    def __init__(self, cfg=None):
        self.cfg = cfg or NemoDiarizationConfig()
        self._temporary_dir = TemporaryDirectory()
        self._work_dir = Path(self._temporary_dir.name)
        self._manifest_path = self._work_dir / "manifest.json"
        self._output_dir = self._work_dir / "output"
        self._clusterer = ClusteringDiarizer(
            self._build_config(self._manifest_path, self._output_dir)
        )

    def _write_manifest(self, audio_path, manifest_path):
        """Write the one-record manifest expected by NeMo diarization."""
        record = {
            "audio_filepath": str(audio_path.resolve()),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": self.cfg.num_speakers,
        }
        manifest_path.write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _build_config(self, manifest_path, output_dir):
        """Build the NeMo configuration for a single input manifest."""

        return OmegaConf.create(
            {
                "device": self.cfg.device,
                "sample_rate": 16_000,
                "batch_size": 1,
                "num_workers": self.cfg.num_workers,
                "verbose": False,
                "diarizer": {
                    "manifest_filepath": str(manifest_path),
                    "out_dir": str(output_dir),
                    "oracle_vad": False,
                    "collar": 0.25,
                    "ignore_overlap": True,
                    "vad": {
                        "model_path": (
                            None
                            if self.cfg.external_vad_manifest is not None
                            else self.cfg.vad_model
                        ),
                        "external_vad_manifest": self.cfg.external_vad_manifest,
                        "parameters": {
                            "window_length_in_sec": 0.15,
                            "shift_length_in_sec": 0.01,
                            "smoothing": "median",
                            "overlap": 0.875,
                            "onset": self.cfg.onset,
                            "offset": self.cfg.offset,
                            "pad_onset": self.cfg.pad_onset,
                            "pad_offset": self.cfg.pad_offset,
                            "min_duration_on": self.cfg.min_duration_on,
                            "min_duration_off": self.cfg.min_duration_off,
                            "filter_speech_first": True,
                        },
                    },
                    "speaker_embeddings": {
                        "model_path": self.cfg.speaker_model,
                        "parameters": {
                            "window_length_in_sec": 1.5,
                            "shift_length_in_sec": 0.75,
                            "multiscale_weights": None,
                            "save_embeddings": False,
                        },
                    },
                    "clustering": {
                        "parameters": {
                            "oracle_num_speakers": self.cfg.num_speakers is not None,
                            "max_num_speakers": self.cfg.max_num_speakers,
                            "enhanced_count_thres": 80,
                            "max_rp_threshold": self.cfg.max_rp_threshold,
                            "sparse_search_volume": 30,
                        }
                    },
                },
            }
        )

    def _parse_rttm(self, rttm_path):
        """Convert NeMo RTTM output to the application's diarization model."""
        raw_turns = []
        for line in rttm_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            raw_turns.append((start, start + float(parts[4]), parts[7]))

        speaker_map = {
            label: f"SPEAKER_{index:02d}"
            for index, label in enumerate(sorted({label for _, _, label in raw_turns}))
        }
        turns = sorted(
            (
                SpeakerTurn(start=start, end=end, speaker_id=speaker_map[label])
                for start, end, label in raw_turns
            ),
            key=lambda turn: (turn.start, turn.end, turn.speaker_id),
        )
        return DiarizationResult(turns=turns, num_speakers=len(speaker_map))

    def diarize(self, audio_path):
        self._write_manifest(audio_path, self._manifest_path)
        self._clusterer.multiscale_embeddings_and_timestamps = {}
        self._clusterer.diarize()

        rttm_path = self._output_dir / "pred_rttms" / f"{audio_path.stem}.rttm"
        return self._parse_rttm(rttm_path)
