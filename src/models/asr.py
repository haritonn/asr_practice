from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProductMention:
    product_id: str
    canonical_name: str
    manufacturer: str
    source_catalog: str
    score: float
    start: float
    end: float
