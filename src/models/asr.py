from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProductMention:
    product_id: str
    canonical_name: str
    score: float
    start: float
    end: float
