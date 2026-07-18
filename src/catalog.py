import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Product:
    id: str
    canonical_name: str
    spoken_forms: tuple[str, ...]
    ctc_forms: tuple[str, ...]

    @property
    def graph_forms(self):
        return tuple(dict.fromkeys((*self.spoken_forms, *self.ctc_forms)))


@dataclass(frozen=True, slots=True)
class ProductCatalog:
    version: str
    products: dict[str, Product]

    @classmethod
    def load(cls, path):
        with path.open(encoding="utf-8") as source:
            data = json.load(source)

        products = {}
        canonical_names = set()
        for item in data["products"]:
            product = Product(
                id=item["id"],
                canonical_name=item["canonical_name"],
                spoken_forms=tuple(item["spoken_forms"]),
                ctc_forms=tuple(item.get("ctc_forms", ())),
            )
            if product.id in products:
                raise ValueError(f"Duplicate product id: {product.id}")
            if product.canonical_name in canonical_names:
                raise ValueError(f"Duplicate canonical name: {product.canonical_name}")
            if not product.spoken_forms:
                raise ValueError(f"Product {product.id} must have spoken forms")
            products[product.id] = product
            canonical_names.add(product.canonical_name)

        return cls(version=data["catalog_version"], products=products)
