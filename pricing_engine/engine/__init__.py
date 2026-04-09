"""
Módulo de engine de pricing da Allu Pricing Engine.

Exporta a função principal price_asset e utilitários de suporte.
"""

from pricing_engine.engine.pricing import price_asset
from pricing_engine.engine.optimizer import annual_irr, calculate_payback, round_to_90
from pricing_engine.engine.depreciation import (
    build_dep_schedule,
    calendar_to_economic,
    get_dep_params,
    get_floor_pct,
    get_sale_book_value,
)
from pricing_engine.engine.cashflow import build_cashflows

__all__ = [
    "price_asset",
    "annual_irr",
    "calculate_payback",
    "round_to_90",
    "build_dep_schedule",
    "calendar_to_economic",
    "get_dep_params",
    "get_floor_pct",
    "get_sale_book_value",
    "build_cashflows",
]
