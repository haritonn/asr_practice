"""NeMo CTC context graph built from a versioned product catalog.

The graph is an acoustic decoder constraint, not post-ASR string matching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from nemo.collections.asr.parts import context_biasing

from .catalog import Product, ProductCatalog
from .models.asr import ProductMention
from .models.configs import ContextGraphConfig


@dataclass(frozen=True, slots=True)
class _SpottedItem:
    label: str
    score: float
    start_frame: int
    end_frame: int


class ContextGraphRecognizer:
    def __init__(self, model, catalog: ProductCatalog, config: ContextGraphConfig):
        self.model = model
        self.catalog = catalog
        self.config = config
        self.context_biasing = context_biasing
        self.blank_id = model.decoding.blank_id
        self._graphs = {}
        self._global_product_ids = frozenset(catalog.products)

    def recognize(
        self, logprobs: np.ndarray, segment_start: float, segment_end: float
    ) -> list[ProductMention]:
        return self._mentions_from_hits(
            self._spot(logprobs, self._product_graph(self._global_product_ids)),
            logprobs,
            segment_start,
            segment_end,
        )

    def _mentions_from_hits(
        self,
        product_hits: list[_SpottedItem],
        logprobs: np.ndarray,
        segment_start: float,
        segment_end: float,
    ) -> list[ProductMention]:
        frame_seconds = (segment_end - segment_start) / max(logprobs.shape[0], 1)
        best_by_id = {}
        for hit in product_hits:
            previous = best_by_id.get(hit.label)
            if previous is None or hit.score > previous.score:
                best_by_id[hit.label] = hit

        mentions = []
        for product_id, hit in best_by_id.items():
            product = self.catalog.products[product_id]
            mentions.append(
                ProductMention(
                    product_id=product.id,
                    canonical_name=product.canonical_name,
                    score=hit.score,
                    start=max(
                        segment_start, segment_start + hit.start_frame * frame_seconds
                    ),
                    end=min(
                        segment_end, segment_start + (hit.end_frame + 1) * frame_seconds
                    ),
                )
            )
        return sorted(mentions, key=lambda item: (item.start, -item.score))

    def _product_graph(self, product_ids: frozenset[str]):
        graph = self._graphs.get(product_ids)
        if graph is None:
            graph = self._build_graph(
                [
                    (product.id, product.graph_forms)
                    for product in self._products(product_ids)
                ]
            )
            self._graphs[product_ids] = graph
        return graph

    def _products(self, product_ids: frozenset[str]) -> list[Product]:
        return [self.catalog.products[product_id] for product_id in sorted(product_ids)]

    def _build_graph(self, items: list[tuple[str, tuple[str, ...]]]):
        graph = self.context_biasing.ContextGraphCTC(blank_id=self.blank_id)
        graph.add_to_graph(
            [
                (label, [self.model.tokenizer.text_to_ids(form) for form in forms])
                for label, forms in items
            ]
        )
        return graph

    def _spot(self, logprobs: np.ndarray, graph) -> list[_SpottedItem]:
        hypotheses = self.context_biasing.run_word_spotter(
            logprobs,
            graph,
            self.model,
            blank_idx=self.blank_id,
            cb_weight=self.config.context_weight,
            keyword_threshold=self.config.keyword_threshold,
            ctc_ali_token_weight=self.config.ctc_alignment_weight,
        )
        return [
            _SpottedItem(
                label=item.word,
                score=float(item.score),
                start_frame=int(item.start_frame),
                end_frame=int(item.end_frame),
            )
            for item in hypotheses
        ]
