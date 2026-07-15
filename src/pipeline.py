from __future__ import annotations

import json
from pathlib import Path

from src.asr.base import BaseAsr
from src.asr.terminology import CtcTerminologyRecognizer
from src.diarization.base import BaseDiarizer
from src.diarization.transcript import assign_speakers, speaker_for_interval
from src.models.diarization import (
    DiarizedTranscript,
    SpeakerAttributedProductMention,
    TimedTranscriptSegment,
)
from src.vad.base import BaseVoiceDetection
class DiarizedSpeechPipeline:
    """Run VAD, ASR and diarization and produce one speaker-attributed transcript."""

    def __init__(
        self,
        asr: BaseAsr,
        vad: BaseVoiceDetection,
        diarizer: BaseDiarizer,
        terminology: CtcTerminologyRecognizer | None = None,
    ):
        self.asr = asr
        self.vad = vad
        self.diarizer = diarizer
        self.terminology = terminology

    def transcribe(self, audio_path: Path) -> DiarizedTranscript:
        speech_segments = self.vad.detect(audio_path)
        transcription = self.asr.transcribe(audio_path, speech_segments)
        diarization = self.diarizer.diarize(audio_path)
        product_mentions = []
        if self.terminology is not None:
            product_mentions = [
                SpeakerAttributedProductMention(
                    product_id=mention.product_id,
                    canonical_name=mention.canonical_name,
                    score=mention.score,
                    start=mention.start,
                    end=mention.end,
                    speaker_id=speaker_for_interval(
                        mention.start, mention.end, diarization.turns
                    ),
                )
                for mention in self.terminology.recognize(audio_path)
            ]
        segments = [
            TimedTranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
            )
            for segment in transcription.segments
            if segment.text.strip()
        ]
        return DiarizedTranscript(
            audio_path=audio_path,
            text=transcription.text,
            language=transcription.language,
            num_speakers=diarization.num_speakers,
            speaker_turns=diarization.turns,
            segments=assign_speakers(segments, diarization.turns),
            product_mentions=product_mentions,
        )

    def transcribe_to_json(self, audio_path: Path, output_path: Path) -> DiarizedTranscript:
        result = self.transcribe(audio_path)
        output_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
