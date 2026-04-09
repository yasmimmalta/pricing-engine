"""
Módulo de otimização de preços da Allu Pricing Engine.

Implementa:
- Cálculo de TIR (Taxa Interna de Retorno) anual a partir de fluxos mensais
- Busca do preço que atinge uma TIR alvo (usando scipy.optimize.brentq)
- Arredondamento para o padrão X9,90 da Allu
- Cálculo do payback
"""

import math
from typing import Callable, Optional, Tuple

import numpy as np
import numpy_financial as npf
from scipy.optimize import brentq


def annual_irr(cashflows: np.ndarray) -> Optional[float]:
    """
    Calcula a TIR anual diretamente a partir de um vetor de fluxos ANUAIS.

    Usa numpy_financial.irr sobre o vetor anual. Cada posição = 1 ano.
    Equivalente a =TIR() do Excel sobre fluxos anuais.

    Args:
        cashflows: Array numpy com os fluxos de caixa anuais.
                   O índice 0 deve ser o fluxo no Ano 0 (geralmente negativo).

    Retorna:
        TIR anual como float (ex: 0.15 para 15% a.a.).
        Retorna None se não for possível calcular.
    """
    if cashflows is None or len(cashflows) == 0:
        return None

    try:
        irr = npf.irr(cashflows)

        if irr is None or np.isnan(irr):
            return None

        return float(irr)

    except Exception:
        return None


def build_irr_vectors(monthly_details: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monta os vetores anuais para cálculo da TIR desalavancada e alavancada.

    TIR desalavancada (FCFF):
      Ano 0 = FCI[0]
      Ano N = FCO[N] + FCI[N]   (N >= 1)

    TIR alavancada (FCFE):
      Ano 0 = FCI[0]
      Ano 1 = FCO[1] + FCI[1] + FCF[0] + FCF[1]
      Ano N = FCO[N] + FCI[N] + FCF[N]   (N >= 2)

    Retorna:
        (vetor_desalavancado, vetor_alavancado) — ambos np.ndarray
    """
    # Agrupa FCO, FCI, FCF por ano
    fco_by_year: dict = {}
    fci_by_year: dict = {}
    fcf_by_year: dict = {}

    for m in monthly_details:
        cal_m = m.get("cal_m", 0)
        ano = 0 if cal_m == 0 else math.ceil(cal_m / 12)

        fco_by_year[ano] = fco_by_year.get(ano, 0.0) + m.get("fco_total", 0.0)
        fci_by_year[ano] = fci_by_year.get(ano, 0.0) + m.get("fci_total", 0.0)
        fcf_by_year[ano] = fcf_by_year.get(ano, 0.0) + m.get("fcf_total", 0.0)

    if not fco_by_year:
        return np.array([]), np.array([])

    max_ano = max(max(fco_by_year), max(fci_by_year), max(fcf_by_year))

    # Monta vetor desalavancado: Ano 0 = FCO[0] + FCI[0] (purchase + ICMS), Ano N = FCO[N] + FCI[N]
    unlev = []
    for ano in range(max_ano + 1):
        unlev.append(fco_by_year.get(ano, 0.0) + fci_by_year.get(ano, 0.0))

    # Monta vetor alavancado: Ano 0 = FCO[0] + FCI[0], Ano 1 = FCO[1]+FCI[1]+FCF[0]+FCF[1], Ano N = FCO+FCI+FCF
    lev = []
    for ano in range(max_ano + 1):
        if ano == 0:
            lev.append(fco_by_year.get(0, 0.0) + fci_by_year.get(0, 0.0))
        elif ano == 1:
            lev.append(
                fco_by_year.get(1, 0.0)
                + fci_by_year.get(1, 0.0)
                + fcf_by_year.get(0, 0.0)
                + fcf_by_year.get(1, 0.0)
            )
        else:
            lev.append(
                fco_by_year.get(ano, 0.0)
                + fci_by_year.get(ano, 0.0)
                + fcf_by_year.get(ano, 0.0)
            )

    return np.array(unlev), np.array(lev)


def find_price_for_irr(
    cashflow_builder: Callable[[float], np.ndarray],
    target_annual_irr: float,
    bounds: Tuple[float, float] = (1.0, 50000.0),
) -> Optional[float]:
    """
    Encontra o preço P que faz a TIR desalavancada atingir o valor alvo.

    Usa o método brentq (bisseção robusta) da scipy para resolver:
        annual_irr(cashflow_builder(P)) = target_annual_irr

    Para target_annual_irr = 0, encontra o breakeven (preço mínimo).
    Para target_annual_irr = min_unlevered_irr, encontra o preço sugerido.

    Args:
        cashflow_builder: Função que recebe um preço P e retorna o array de
                          fluxos de caixa desalavancados (cf_unlev).
        target_annual_irr: TIR anual alvo (ex: 0.15 para 15% a.a., 0.0 para breakeven).
        bounds: Intervalo (min, max) de busca do preço. Default: (1, 50000).

    Retorna:
        Preço P (float) que atinge a TIR alvo.
        Retorna None se não for possível encontrar (ex: a função não muda de sinal
        no intervalo, ou ocorre erro numérico).

    Nota:
        A função de objetivo é: f(P) = annual_irr(cashflows(P)) - target_annual_irr
        O brentq requer que f(bounds[0]) e f(bounds[1]) tenham sinais opostos.
    """
    lo, hi = bounds

    def objective(P: float) -> float:
        """Diferença entre TIR calculada e TIR alvo."""
        cf = cashflow_builder(P)
        irr = annual_irr(cf)
        if irr is None:
            # Se não conseguiu calcular, retorna valor extremo para orientar o solver
            return -999.0
        return irr - target_annual_irr

    try:
        # Verifica se a função muda de sinal no intervalo (pré-requisito do brentq)
        f_lo = objective(lo)
        f_hi = objective(hi)

        if f_lo * f_hi > 0:
            # Mesmos sinais — o brentq não conseguirá encontrar raiz
            # Tenta expandir o intervalo superior como fallback
            for hi_test in [100000.0, 500000.0]:
                f_hi_test = objective(hi_test)
                if f_lo * f_hi_test <= 0:
                    hi = hi_test
                    f_hi = f_hi_test
                    break
            else:
                return None

        # Executa o brentq com tolerância razoável
        price = brentq(objective, lo, hi, xtol=0.01, rtol=1e-6, maxiter=200)
        return float(price)

    except (ValueError, RuntimeError):
        return None


def round_to_90(price: float) -> float:
    """
    Arredonda o preço para o padrão X9,90 da Allu.

    O padrão X9,90 significa que o preço termina em 9,90:
    - 299,90 — 399,90 — 499,90 — etc.

    Fórmula: ceil((price + 0.10) / 10) * 10 - 0.10

    Esta fórmula garante que:
    - O preço arredondado seja SEMPRE >= ao preço original (nunca arredonda para baixo).
    - O resultado termina sempre em 9,90.

    Args:
        price: Preço a ser arredondado (R$).

    Retorna:
        Preço arredondado no padrão X9,90.

    Exemplos:
        >>> round_to_90(297.45)
        299.9
        >>> round_to_90(299.90)
        299.9
        >>> round_to_90(300.00)
        309.9
        >>> round_to_90(350.00)
        359.9
        >>> round_to_90(289.91)
        299.9
    """
    if price is None or math.isnan(price):
        return price

    # Adiciona 0.10 para garantir arredondamento para cima quando já está em X9,90
    # Divide por 10, arredonda para cima (ceil), multiplica por 10, subtrai 0.10
    rounded = math.ceil((price + 0.10) / 10) * 10 - 0.10

    return round(rounded, 2)


def calculate_payback(cf_unlev: np.ndarray) -> Optional[int]:
    """
    Calcula o mês de payback do fluxo de caixa desalavancado.

    O payback é o primeiro mês em que o fluxo de caixa acumulado se torna positivo.

    Args:
        cf_unlev: Array com os fluxos de caixa mensais desalavancados.
                  O índice 0 corresponde a t=0 (desembolso inicial).

    Retorna:
        int com o índice do mês do payback (1-indexed a partir de t=1).
        Retorna None se o fluxo acumulado nunca ficar positivo.

    Exemplo:
        >>> cf = np.array([-1000, -200, 300, 300, 300, 300, 300])
        >>> calculate_payback(cf)
        4  # no mês 4, o acumulado fica positivo
    """
    if cf_unlev is None or len(cf_unlev) == 0:
        return None

    cumsum = np.cumsum(cf_unlev)

    # Procura o primeiro índice onde o acumulado fica > 0
    positive_months = np.where(cumsum > 0)[0]

    if len(positive_months) == 0:
        return None

    return int(positive_months[0])


def find_price_for_payback(
    full_cashflow_builder: Callable[[float], Tuple[np.ndarray, np.ndarray, list]],
    max_payback: int = 24,
    bounds: Tuple[float, float] = (1.0, 50000.0),
) -> Optional[float]:
    """
    Encontra o menor preço P cujo payback desalavancado <= max_payback meses.

    Usa bisseção: preço sobe → payback cai.
    """
    lo, hi = bounds

    def get_payback(P: float) -> int:
        cf_u, _, _ = full_cashflow_builder(P)
        pb = calculate_payback(cf_u)
        return pb if pb is not None else 9999

    # Se já atende no limite inferior, retorna
    if get_payback(hi) > max_payback:
        return None  # impossível mesmo com preço altíssimo

    if get_payback(lo) <= max_payback:
        return lo

    # Bisseção
    for _ in range(100):
        mid = (lo + hi) / 2
        if hi - lo < 0.01:
            break
        if get_payback(mid) <= max_payback:
            hi = mid
        else:
            lo = mid

    return float(hi)


def find_price_for_payback_lev(
    full_cashflow_builder: Callable[[float], Tuple[np.ndarray, np.ndarray, list]],
    max_payback: int = 30,
    bounds: Tuple[float, float] = (1.0, 50000.0),
) -> Optional[float]:
    """
    Encontra o menor preço P cujo payback alavancado <= max_payback meses.

    Usa bisseção sobre cf_lev.
    """
    lo, hi = bounds

    def get_payback(P: float) -> int:
        _, cf_l, _ = full_cashflow_builder(P)
        pb = calculate_payback(cf_l)
        return pb if pb is not None else 9999

    if get_payback(hi) > max_payback:
        return None

    if get_payback(lo) <= max_payback:
        return lo

    for _ in range(100):
        mid = (lo + hi) / 2
        if hi - lo < 0.01:
            break
        if get_payback(mid) <= max_payback:
            hi = mid
        else:
            lo = mid

    return float(hi)


def calculate_ebitda_margin(monthly_details: list) -> Optional[float]:
    """
    Calcula a margem EBITDA do contrato.

    margem = (EBITDA_ass + EBITDA_venda) / Receita_líq_ass

    Retorna float (ex: 0.15 para 15%) ou None se receita = 0.
    """
    total_ebitda = sum(
        m.get("dre_ass_ebitda", 0) + m.get("dre_venda_ebitda", 0)
        for m in monthly_details
    )
    total_rec_liq = sum(m.get("dre_ass_receita_liq", 0) for m in monthly_details)

    if total_rec_liq == 0:
        return None
    return total_ebitda / total_rec_liq


def calculate_contribution_margin(monthly_details: list) -> Optional[float]:
    """
    Calcula a margem de contribuição do contrato.

    Margem de contribuição = Receita líq - Custos - CAC - Risco
                           = Lucro bruto - CAC - Risco

    Diferença do EBITDA: NÃO desconta PDD e Default.

    Retorna float (ex: 0.15 para 15%) ou None se receita = 0.
    """
    total_lucro_bruto = sum(m.get("dre_ass_lucro_bruto", 0) for m in monthly_details)
    total_cac = sum(m.get("dre_ass_cac", 0) for m in monthly_details)
    total_risco = sum(m.get("dre_ass_risk_query", 0) for m in monthly_details)
    total_rec_liq = sum(m.get("dre_ass_receita_liq", 0) for m in monthly_details)

    if total_rec_liq == 0:
        return None

    margem_abs = total_lucro_bruto - total_cac - total_risco
    return margem_abs / total_rec_liq


def find_price_for_margin(
    full_cashflow_builder: Callable[[float], Tuple[np.ndarray, np.ndarray, list]],
    min_margin: float = 0.13,
    bounds: Tuple[float, float] = (1.0, 50000.0),
) -> Optional[float]:
    """
    Encontra o menor preço P cuja margem EBITDA >= min_margin.

    Usa bisseção: preço sobe → margem sobe.
    """
    lo, hi = bounds

    def get_margin(P: float) -> float:
        _, _, md = full_cashflow_builder(P)
        m = calculate_ebitda_margin(md)
        return m if m is not None else -999.0

    if get_margin(hi) < min_margin:
        return None

    if get_margin(lo) >= min_margin:
        return lo

    for _ in range(100):
        mid = (lo + hi) / 2
        if hi - lo < 0.01:
            break
        if get_margin(mid) >= min_margin:
            hi = mid
        else:
            lo = mid

    return float(hi)
