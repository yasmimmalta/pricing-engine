"""
Modelo de dados para o resultado do cálculo de pricing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pricing_engine.models.asset import Asset
from pricing_engine.models.contract import ContractParams, PricingParams


@dataclass
class PricingResult:
    """
    Resultado completo do cálculo de pricing para um ativo e contrato específicos.

    Atributos:
        asset: O ativo precificado.
        contract: Os parâmetros do contrato.
        params: Os parâmetros de pricing utilizados.
        renewal_price: Preço da renovação (None se não houver renovação).
        breakeven_price: Preço mínimo para TIR desalavancada = 0% (ponto de equilíbrio).
        suggested_price_raw: Preço sugerido antes do arredondamento (TIR = min_unlevered_irr).
        suggested_price: Preço sugerido arredondado para X9,90 (ex: 299,90).
        unlevered_irr: TIR desalavancada anual calculada com o suggested_price.
        levered_irr: TIR alavancada anual calculada com o suggested_price.
        payback_months: Mês em que o fluxo de caixa acumulado desalavancado fica positivo.
        cf_unlev: Array com o fluxo de caixa desalavancado mensal (índice 0 = t=0).
        cf_lev: Array com o fluxo de caixa alavancado mensal (índice 0 = t=0).
        monthly_details: Lista de dicts com o detalhamento de cada mês do fluxo.
    """

    asset: Asset
    contract: ContractParams
    params: PricingParams
    renewal_price: Optional[float]
    breakeven_price: Optional[float]
    suggested_price_raw: Optional[float]
    suggested_price: Optional[float]       # arredondado para X9,90
    unlevered_irr: Optional[float]         # TIR anual desalavancada
    levered_irr: Optional[float]           # TIR anual alavancada
    payback_months: Optional[int]
    payback_lev_months: Optional[int] = None  # payback alavancado
    margem_ebitda: Optional[float] = None  # margem EBITDA do contrato (ex: 0.15 = 15%)
    cf_unlev: Optional[np.ndarray] = None  # fluxo mensal desalavancado
    cf_lev: Optional[np.ndarray] = None    # fluxo mensal alavancado
    monthly_details: List[Dict] = field(default_factory=list)
    annual_cashflows: List[Dict] = field(default_factory=list)  # FCO/FCI/FCF consolidado por ano
    # Restrições individuais (preço mínimo exigido por cada uma)
    price_for_irr_constraint: Optional[float] = None
    price_for_payback_constraint: Optional[float] = None
    price_for_margin_constraint: Optional[float] = None
    price_for_payback_lev_constraint: Optional[float] = None
    binding_constraint: Optional[str] = None  # qual restrição fixou o preço

    def summary_dict(self) -> dict:
        """
        Retorna um dicionário com os principais outputs formatados.

        Útil para exibição em tabelas e exportação.
        """
        # Formata valores monetários de forma segura
        def fmt_brl(val: Optional[float]) -> str:
            if val is None:
                return "N/A"
            return f"R$ {val:,.2f}"

        def fmt_pct(val: Optional[float]) -> str:
            if val is None:
                return "N/A"
            return f"{val:.2%}"

        return {
            # Identificação
            "ativo_id": self.asset.id,
            "ativo_nome": self.asset.name,
            "cliente": self.contract.client_type.value,
            "prazo_meses": self.contract.term_months,
            "ciclo_economico_meses": self.contract.eco_total,
            "tem_renovacao": self.contract.has_renewal,
            "meses_renovacao": self.contract.renewal_months,
            # Preços calculados
            "preco_sugerido": self.suggested_price,
            "preco_sugerido_fmt": fmt_brl(self.suggested_price),
            "preco_bruto": self.suggested_price_raw,
            "preco_bruto_fmt": fmt_brl(self.suggested_price_raw),
            "preco_breakeven": self.breakeven_price,
            "preco_breakeven_fmt": fmt_brl(self.breakeven_price),
            "preco_renovacao": self.renewal_price,
            "preco_renovacao_fmt": fmt_brl(self.renewal_price),
            # Métricas financeiras
            "tir_desalavancada": self.unlevered_irr,
            "tir_desalavancada_fmt": fmt_pct(self.unlevered_irr),
            "tir_alavancada": self.levered_irr,
            "tir_alavancada_fmt": fmt_pct(self.levered_irr),
            "payback_meses": self.payback_months,
            "margem_ebitda": self.margem_ebitda,
            "margem_ebitda_fmt": fmt_pct(self.margem_ebitda),
            # Parâmetros de referência
            "preco_compra": self.asset.purchase_price,
            "preco_mercado": self.asset.market_price,
            "tir_minima_alvo": self.params.min_unlevered_irr,
            "tir_minima_fmt": fmt_pct(self.params.min_unlevered_irr),
        }

    def __repr__(self) -> str:
        preco = f"R$ {self.suggested_price:.2f}" if self.suggested_price else "N/A"
        tir = f"{self.unlevered_irr:.2%}" if self.unlevered_irr else "N/A"
        return (
            f"PricingResult({self.asset.name}, {self.contract.client_type.value} "
            f"{self.contract.term_months}m → preço={preco}, TIR={tir})"
        )
