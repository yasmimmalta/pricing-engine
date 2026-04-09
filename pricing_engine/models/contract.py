"""
Modelos de dados para parâmetros de contrato e pricing.

Define as regras de ciclo econômico, renovação implícita e parâmetros tributários/operacionais.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ClientType(str, Enum):
    """Tipo de cliente: B2C (consumidor final) ou B2B (corporativo)."""
    B2C = "B2C"
    B2B = "B2B"


# ============================================================
# MAPEAMENTO DE RENOVAÇÃO IMPLÍCITA
# ============================================================
# Estrutura: {(client_type, term_months): renewal_months}
# renewal_months = 0 significa sem renovação (ciclo encerra no term)
#
# Lógica:
#   B2C: ciclo econômico total = 36 meses
#     - 12m → renova por 24m  (12 + 24 = 36)
#     - 24m → renova por 12m  (24 + 12 = 36)
#     - 36m → sem renovação   (36 = 36)
#
#   B2B: ciclo econômico total = 48 meses
#     - 12m → renova por 36m  (12 + 36 = 48)
#     - 24m → renova por 24m  (24 + 24 = 48)
#     - 36m → renova por 12m  (36 + 12 = 48)
#     - 48m → sem renovação   (48 = 48)
# ============================================================

RENEWAL_MAP: dict = {
    (ClientType.B2C, 12): 24,
    (ClientType.B2C, 24): 12,
    (ClientType.B2C, 36): 0,
    (ClientType.B2B, 12): 36,
    (ClientType.B2B, 24): 24,
    (ClientType.B2B, 36): 12,
    (ClientType.B2B, 48): 0,
}

# Prazos válidos por tipo de cliente
VALID_TERMS = {
    ClientType.B2C: [12, 24, 36],
    ClientType.B2B: [12, 24, 36, 48],
}


@dataclass
class ContractParams:
    """
    Parâmetros do contrato de assinatura.

    Atributos:
        client_type: Tipo do cliente (B2C ou B2B).
        term_months: Prazo do contrato inicial em meses (12, 24, 36 ou 48 para B2B).
        with_final_sale: Se True, inclui a venda do device ao final do ciclo econômico.
        is_standalone: Se True, trata como contrato sem renovação interna (usado no cálculo
                       do preço de renovação para evitar recursão infinita).
    """

    client_type: ClientType
    term_months: int        # 12, 24, 36 (ou 48 para B2B)
    with_final_sale: bool = True
    is_standalone: bool = False   # True = sem renovação interna (preço de renovação)

    def __post_init__(self):
        """Valida os parâmetros do contrato."""
        # Normaliza client_type se vier como string
        if isinstance(self.client_type, str):
            self.client_type = ClientType(self.client_type.upper())

        # Valida prazo
        valid = VALID_TERMS.get(self.client_type, [])
        if self.term_months not in valid:
            raise ValueError(
                f"Prazo {self.term_months}m inválido para {self.client_type.value}. "
                f"Prazos válidos: {valid}"
            )

    @property
    def eco_total(self) -> int:
        """
        Duração total do ciclo econômico em meses.

        Se is_standalone=True, o ciclo é apenas o term_months (sem renovação).
        Caso contrário:
          - B2C: 36 meses
          - B2B: 48 meses
        """
        if self.is_standalone:
            return self.term_months
        return 36 if self.client_type == ClientType.B2C else 48

    @property
    def renewal_months(self) -> int:
        """
        Quantidade de meses da renovação.

        Se is_standalone=True, retorna 0 (sem renovação).
        Caso contrário, consulta o RENEWAL_MAP.
        """
        if self.is_standalone:
            return 0
        return RENEWAL_MAP.get((self.client_type, self.term_months), 0)

    @property
    def has_renewal(self) -> bool:
        """Retorna True se houver período de renovação."""
        return self.renewal_months > 0

    @property
    def total_calendar_months(self) -> int:
        """
        Total de meses do ciclo calendário.

        Sem gap: renovação começa imediatamente após o contrato inicial.
        Fórmula: term_months + renewal_months  se houver renovação
                 term_months                    se não houver renovação
        """
        if self.has_renewal:
            return self.term_months + self.renewal_months
        return self.term_months

    @property
    def gap_calendar_month(self) -> Optional[int]:
        """Sem gap — retorna sempre None."""
        return None

    @property
    def renewal_start_calendar_month(self) -> Optional[int]:
        """
        Mês calendário em que começa a renovação.

        Retorna None se não houver renovação.
        Sem gap: renovação começa no mês seguinte ao último do contrato inicial.
        """
        return self.term_months + 1 if self.has_renewal else None

    def __repr__(self) -> str:
        renova = f", renova {self.renewal_months}m" if self.has_renewal else ""
        standalone = " [standalone]" if self.is_standalone else ""
        return (
            f"ContractParams({self.client_type.value}, {self.term_months}m"
            f"{renova}, eco_total={self.eco_total}m{standalone})"
        )


@dataclass
class PricingParams:
    """
    Parâmetros do modelo de precificação da Allu.

    Contém todos os percentuais tributários, custos operacionais e parâmetros
    de risco usados no cálculo do fluxo de caixa e otimização de preço.
    """

    # ----------------------------------------------------------
    # FISCAL E TRIBUTÁRIO
    # ----------------------------------------------------------

    # ICMS sobre preço de compra do device (14%)
    icms_pct: float = 0.14

    # Custo de captação sobre o funding total (upfront, t=0) — 4%
    funding_captacao_pct: float = 0.04

    # Taxa de juros anual da dívida (COMPOSTA) — 23,72% a.a.
    debt_annual_rate: float = 0.2372

    # PIS + COFINS sobre receita bruta (9,25%)
    pis_cofins_pct: float = 0.0925

    # Crédito de PIS/COFINS sobre o preço de aquisição (9,25%)
    pis_cofins_credit_pct: float = 0.0925

    # ISS sobre receita (2,5%)
    iss_pct: float = 0.025

    # MDR (taxa da maquininha/gateway) sobre receita (2%)
    mdr_pct: float = 0.02

    # ----------------------------------------------------------
    # CUSTOS COMERCIAIS
    # ----------------------------------------------------------

    # CAC de venda como % do valor de venda do device ao final — 10%
    cac_venda_pct: float = 0.10

    # Custo de preparação do device para venda — R$ 80
    prep_venda: float = 80.0

    # Logística para envio do device vendido — R$ 40
    logistics_venda: float = 40.0

    # Logística de assinatura: envio no início e na renovação — R$ 50
    logistics_assinatura: float = 50.0

    # Consulta de risco de crédito (B2C only) — R$ 15
    risk_query_value: float = 15.0

    # Benefícios mensais ao cliente (seguro, suporte, etc.) — R$ 20/mês
    customer_benefits: float = 20.0

    # Logistica AT: percentual anual para provisão de logística de troca por estrago
    # Separado do maintenance_annual_pct do ativo. Padrão: 1,42% a.a.
    logistics_at_pct: float = 0.0142

    # ----------------------------------------------------------
    # RISCO E INADIMPLÊNCIA
    # ----------------------------------------------------------

    # PDD (Provisão para Devedores Duvidosos) — 8% da mensalidade
    pdd_pct: float = 0.08

    # Default anual sobre o valor líquido contábil do ativo — 3% a.a.
    # Fórmula: default_mensal = default_pct * (valor_liquido_m / prazo_meses)
    default_pct: float = 0.03

    # ----------------------------------------------------------
    # DEPRECIAÇÃO CONTÁBIL (base do cálculo de default M12)
    # ----------------------------------------------------------

    # Taxa de depreciação contábil anual — fallback universal quando categoria
    # não está em dep_contabil.json. O arquivo por categoria tem prioridade.
    # Valor padrão: 20% a.a. (linear sobre purchase_price)
    dep_contabil_pct: float = 0.20

    # ----------------------------------------------------------
    # OTIMIZAÇÃO
    # ----------------------------------------------------------

    # TIR mínima desalavancada (hurdle rate) — 30% a.a.
    min_unlevered_irr: float = 0.30

    # ----------------------------------------------------------
    # IMPOSTO DE RENDA E CSLL
    # ----------------------------------------------------------

    # Alíquota combinada IR + CSLL sobre o EBT — incide quando EBT > 0 (23,80%)
    ir_csll_pct: float = 0.238

    # ----------------------------------------------------------
    # PRAZO DA DÍVIDA
    # ----------------------------------------------------------

    # Prazo médio da dívida em meses — juros do mês 1 ao mês prazo_divida_meses,
    # amortização do principal no mês seguinte (prazo_divida_meses + 1).
    # Fixo em 30 meses, independente do ciclo econômico do ativo.
    prazo_divida_meses: int = 30

    @property
    def debt_monthly_rate(self) -> float:
        """
        Taxa de juros mensal — juros simples: debt_annual_rate / 12.

        O prazo do contrato NÃO altera a taxa mensal.
        O prazo apenas determina por quantos meses os juros incidem.
        Fórmula: debt_annual_rate / 12
        Para 23,72% a.a.: 23,72% / 12 = 1,9767%/mês (constante)

        Juros totais = principal × (debt_annual_rate / 12) × term_months
        """
        return self.debt_annual_rate / 12

    def __repr__(self) -> str:
        return (
            f"PricingParams(irr_alvo={self.min_unlevered_irr:.1%}, "
            f"juros={self.debt_annual_rate:.2%}a.a., "
            f"pis_cofins={self.pis_cofins_pct:.2%})"
        )
