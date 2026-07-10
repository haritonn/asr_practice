from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextGraphConfig:
    context_weight: float
    keyword_threshold: float
    ctc_alignment_weight: float
