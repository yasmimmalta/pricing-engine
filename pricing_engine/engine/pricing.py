"""
Módulo principal de orquestração do pricing da Allu Pricing Engine.

Ponto de entrada: função price_asset().
Coordena depreciação, fluxo de caixa, otimização e empacota o resultado.
"""

import math
from typing import Dict, List, Optional

import numpy as np

import dataclasses

from pricing_engine.config import get_dep_contabil_curves, get_depreciation_curves
from pricing_engine.engine.cashflow import build_cashflows
from pricing_engine.engine.depreciation import (
    build_dep_schedule,
    get_dep_params,
    get_sale_book_value,
)
from pricing_engine.engine.optimizer import (
    annual_irr,
    build_irr_vectors,
    calculate_ebitda_margin,
    calculate_payback,
    find_price_for_irr,
    find_price_for_margin,
    find_price_for_payback,
    find_price_for_payback_lev,
    round_to_90,
)
from pricing_engine.models.asset import Asset
from pricing_engine.models.contract import ClientType, ContractParams, PricingParams
from pricing_engine.models.result import PricingResult


def price_asset(
    asset: Asset,
    contract: ContractParams,
    params: PricingParams,
    renewal_price_override: Optional[float] = None,
) -> PricingResult:
    """
    Ponto de entrada principal do motor de precificação.

    Orquestra todo o cálculo:
    1. Carrega as curvas de depreciação
    2. Constrói o schedule de depreciação do ativo
    3. Define o preço de renovação (override externo ou cálculo interno)
    4. Monta o cashflow_builder(P) que retorna cf_unlev para qualquer P
    5. Encontra o preço sugerido (TIR = min_unlevered_irr) via brentq
    6. Encontra o preço breakeven (TIR = 0%)
    7. Calcula as métricas financeiras com o preço sugerido
    8. Retorna o PricingResult completo

    Args:
        asset: O ativo (device) a ser precificado.
        contract: Parâmetros do contrato (tipo de cliente, prazo, renovação).
        params: Parâmetros de pricing (tributário, operacional, risco, hurdle rate).
        renewal_price_override: Preço de renovação externo. Quando fornecido, substitui o
            cálculo interno (_calculate_renewal_price). Usado para garantir que o preço
            de renovação seja idêntico ao preço da assinatura do prazo correspondente.

    Retorna:
        PricingResult com todos os outputs calculados.
    """
    # ----------------------------------------------------------
    # 1. Carrega curvas de depreciação
    # ----------------------------------------------------------
    # dep_venda: curvas econômicas/comerciais → valor de venda final (M5)
    curves = get_depreciation_curves()

    # dep_contabil: taxas contábeis por categoria → base do default (M12)
    # Prioridade: taxa do arquivo por categoria > params.dep_contabil_pct (fallback)
    dep_contabil_curves = get_dep_contabil_curves()
    cat_key = asset.category.lower().strip()
    cat_contabil = dep_contabil_curves.get(cat_key) or dep_contabil_curves.get("default", {})
    cat_contabil_rate = cat_contabil.get("annual_rate", params.dep_contabil_pct)

    # Se a taxa por categoria difere do fallback universal, aplica override
    if cat_contabil_rate != params.dep_contabil_pct:
        params = dataclasses.replace(params, dep_contabil_pct=cat_contabil_rate)

    # ----------------------------------------------------------
    # 2. Constrói o schedule de depreciação
    # ----------------------------------------------------------
    dep_params   = get_dep_params(asset.category, curves)
    dep_schedule = build_dep_schedule(
        market_price=asset.market_price,
        eco_total=contract.eco_total,
        floor_pct=dep_params["floor_pct"],
        dep_method=dep_params.get("dep_method", "linear"),
        annual_rate=dep_params.get("annual_rate"),
    )
    # Valor de venda final: stepped (demais categorias) ou exponencial (iPhone)
    # Separado do dep_schedule, que é usado exclusivamente para o default mensal
    sale_value = get_sale_book_value(
        market_price=asset.market_price,
        eco_total=contract.eco_total,
        dep_params=dep_params,
    )

    # ----------------------------------------------------------
    # 3. Define o preço de renovação
    # ----------------------------------------------------------
    # Regra de negócio: o preço de renovação deve ser idêntico ao preço da
    # assinatura do prazo correspondente (passado via renewal_price_override).
    # Caso não fornecido (chamadas legacy ou renovação autônoma), usa o cálculo
    # interno sobre o ativo depreciado.
    renewal_price = 0.0
    if contract.has_renewal:
        if renewal_price_override is not None:
            renewal_price = renewal_price_override
        else:
            renewal_price = _calculate_renewal_price(
                asset=asset,
                contract=contract,
                params=params,
                dep_schedule=dep_schedule,
                curves=curves,
            )

    # ----------------------------------------------------------
    # 4. Monta o cashflow_builder(P) -> vetor anual desalavancado
    #    Ano 0 = FCI[0], Ano N = FCO[N] + FCI[N]
    # ----------------------------------------------------------
    def cashflow_builder(P: float) -> np.ndarray:
        """Constrói o vetor anual desalavancado para o preço P."""
        _, _, md = build_cashflows(
            asset=asset,
            contract=contract,
            params=params,
            P=P,
            renewal_price=renewal_price,
            dep_schedule=dep_schedule,
            curves=curves,
            sale_book_value=sale_value,
        )
        vec_unlev, _ = build_irr_vectors(md)
        return vec_unlev

    # ----------------------------------------------------------
    # 5. Builder completo (retorna cf_unlev, cf_lev, monthly_details)
    # ----------------------------------------------------------
    def full_cashflow_builder(P: float):
        return build_cashflows(
            asset=asset, contract=contract, params=params, P=P,
            renewal_price=renewal_price, dep_schedule=dep_schedule,
            curves=curves, sale_book_value=sale_value,
        )

    # ----------------------------------------------------------
    # 6. Preço mínimo para cada restrição
    # ----------------------------------------------------------
    # 6a. TIR desalavancada >= 30%
    price_for_irr = find_price_for_irr(
        cashflow_builder=cashflow_builder,
        target_annual_irr=params.min_unlevered_irr,
    )

    # 6b. Payback desalavancado — calculado apenas para output informativo, NÃO entra no solver
    price_for_payback = find_price_for_payback(
        full_cashflow_builder=full_cashflow_builder,
        max_payback=24,
    )

    # 6c. Margem EBITDA >= 13%
    price_for_margin = find_price_for_margin(
        full_cashflow_builder=full_cashflow_builder,
        min_margin=0.13,
    )

    # 6d. Payback alavancado <= 30 meses
    price_for_payback_lev = find_price_for_payback_lev(
        full_cashflow_builder=full_cashflow_builder,
        max_payback=30,
    )

    # Preço final = max das 2 restrições ativas (ambos os paybacks são apenas informativos)
    candidates = [p for p in [price_for_irr, price_for_margin] if p is not None]
    suggested_price_raw = max(candidates) if candidates else None
    suggested_price = round_to_90(suggested_price_raw) if suggested_price_raw else None

    # Identifica qual restrição é dominante entre as 2 ativas
    _constraint_map = [
        (price_for_irr,    "TIR Desalavancada"),
        (price_for_margin, "Margem EBITDA"),
    ]
    binding_constraint = None
    if suggested_price_raw is not None:
        for price, name in _constraint_map:
            if price is not None and abs(price - suggested_price_raw) < 0.01:
                binding_constraint = name
                break

    # ----------------------------------------------------------
    # 7. Preço breakeven: TIR = 0%
    # ----------------------------------------------------------
    breakeven_price_raw = find_price_for_irr(
        cashflow_builder=cashflow_builder,
        target_annual_irr=0.0,
    )
    breakeven_price = round_to_90(breakeven_price_raw) if breakeven_price_raw else None

    # ----------------------------------------------------------
    # 8. Calcula métricas
    #    - Cashflows/DRE: preço arredondado (suggested_price) → o que o cliente paga
    #    - TIR desalavancada: preço raw → exatamente 30% (hurdle rate por construção)
    #    - TIR alavancada: preço arredondado → reflete o retorno real do acionista
    # ----------------------------------------------------------
    calc_price = suggested_price or suggested_price_raw

    cf_unlev_final = None
    cf_lev_final   = None
    monthly_details = []
    unlevered_irr   = None
    levered_irr     = None
    payback_months  = None
    payback_lev_months = None
    margem_ebitda   = None

    if calc_price is not None:
        # Cashflows no preço arredondado (para DFC, DRE, display)
        cf_unlev_final, cf_lev_final, monthly_details = build_cashflows(
            asset=asset,
            contract=contract,
            params=params,
            P=calc_price,
            renewal_price=renewal_price,
            dep_schedule=dep_schedule,
            curves=curves,
            sale_book_value=sale_value,
        )

        # TIR desalavancada: usa preço raw para dar exatamente 30%
        if suggested_price_raw is not None:
            _, _, md_raw = build_cashflows(
                asset=asset,
                contract=contract,
                params=params,
                P=suggested_price_raw,
                renewal_price=renewal_price,
                dep_schedule=dep_schedule,
                curves=curves,
                sale_book_value=sale_value,
            )
            vec_unlev_raw, _ = build_irr_vectors(md_raw)
            unlevered_irr = annual_irr(vec_unlev_raw)
        else:
            vec_unlev, _ = build_irr_vectors(monthly_details)
            unlevered_irr = annual_irr(vec_unlev)

        # TIR alavancada: usa preço arredondado (retorno real do acionista)
        _, vec_lev = build_irr_vectors(monthly_details)
        levered_irr = annual_irr(vec_lev)

        payback_months = calculate_payback(cf_unlev_final)
        payback_lev_months = calculate_payback(cf_lev_final)
        margem_ebitda  = calculate_ebitda_margin(monthly_details)

    # ----------------------------------------------------------
    # 9. Consolida FCO/FCI/FCF por ano
    # ----------------------------------------------------------
    annual_cashflows = _compute_annual_cashflows(monthly_details)

    # ----------------------------------------------------------
    # 9. Monta e retorna o resultado
    # ----------------------------------------------------------
    return PricingResult(
        asset=asset,
        contract=contract,
        params=params,
        renewal_price=renewal_price if contract.has_renewal else None,
        breakeven_price=breakeven_price,
        suggested_price_raw=suggested_price_raw,
        suggested_price=suggested_price,
        unlevered_irr=unlevered_irr,
        levered_irr=levered_irr,
        payback_months=payback_months,
        payback_lev_months=payback_lev_months,
        margem_ebitda=margem_ebitda,
        cf_unlev=cf_unlev_final,
        cf_lev=cf_lev_final,
        monthly_details=monthly_details,
        annual_cashflows=annual_cashflows,
        price_for_irr_constraint=price_for_irr,
        price_for_payback_constraint=price_for_payback,
        price_for_margin_constraint=price_for_margin,
        price_for_payback_lev_constraint=price_for_payback_lev,
        binding_constraint=binding_constraint,
    )


def _compute_annual_cashflows(monthly_details: List[Dict]) -> List[Dict]:
    """
    Consolida os campos FCO/FCI/FCF mensais em totais anuais.

    Regra de atribuição:
      - t=0  → Ano 0
      - m=1..12  → Ano 1
      - m=13..24 → Ano 2  etc.

    Retorna lista de dicts (um por ano + linha "Total").
    Itens individuais = valores absolutos; totais fco/fci/fcf/cf_total = líquidos com sinal.
    """
    DFC_KEYS = [
        # FCO
        "fco_receita_bruta", "fco_pis_cofins", "fco_iss", "fco_mdr",
        "fco_manutencao", "fco_pdd", "fco_default", "fco_logistics_troca",
        "fco_cac", "fco_risk_query", "fco_logistics_ass", "fco_customer_benefits",
        "fco_logistics_devolucao", "fco_ir_csll", "fco_icms", "ir_fcff", "ir_fcfe", "fco_total",
        # FCI
        "fci_compra", "fci_sale_val", "fci_cac_venda", "fci_prep_venda",
        "fci_logistics_venda", "fci_total",
        # FCF
        "fcf_funding", "fcf_captacao_fee", "fcf_juros", "fcf_principal",
        "fcf_total", "cf_total",
        # DRE Assinatura
        "dre_ass_receita", "dre_ass_impostos", "dre_ass_receita_liq",
        "dre_ass_customer_benefits", "dre_ass_manutencao", "dre_ass_logistica",
        "dre_ass_credito_pis", "dre_ass_icms", "dre_ass_lucro_bruto",
        "dre_ass_risk_query", "dre_ass_cac", "dre_ass_pdd", "dre_ass_default",
        "dre_ass_ebitda", "dre_ass_ebit",
        # DRE Venda
        "dre_venda_receita", "dre_venda_dep_acum", "dre_venda_custo_venda",
        "dre_venda_lucro_bruto", "dre_venda_cac", "dre_venda_logistica",
        "dre_venda_prep", "dre_venda_ebitda", "dre_venda_dep_contabil",
        "dre_venda_ebit",
        # Consolidação DRE
        "dre_ebit_consolidado", "dre_juros", "dre_mdr",
        "dre_ebt", "dre_ir_csll", "dre_lucro_liq",
    ]

    annual: Dict[int, Dict] = {}
    for detail in monthly_details:
        cal_m = detail.get("cal_m", 0)
        ano   = 0 if cal_m == 0 else math.ceil(cal_m / 12)

        if ano not in annual:
            annual[ano] = {"ano": ano, **{k: 0.0 for k in DFC_KEYS}}

        for k in DFC_KEYS:
            annual[ano][k] += detail.get(k, 0.0)

    rows = sorted(annual.values(), key=lambda x: x["ano"])

    # Linha de total acumulado do ciclo
    total_row: Dict = {"ano": "Total"}
    for k in DFC_KEYS:
        total_row[k] = sum(r[k] for r in rows)
    rows.append(total_row)

    return rows


def _calculate_renewal_price(
    asset: Asset,
    contract: ContractParams,
    params: PricingParams,
    dep_schedule: List[float],
    curves: dict,
) -> float:
    """
    Calcula o preço da renovação como contrato standalone sobre o ativo depreciado.

    A lógica de renovação é:
    - O ativo já depreciou por term_months meses durante o contrato inicial
    - O "novo valor de mercado" do ativo é seu book value em dep_schedule[term_months]
    - O contrato de renovação é tratado como standalone (sem nova renovação dentro dele)
    - O PIS/COFINS credit_saldo começa em 0 (conservador — já foi consumido no ciclo anterior)
    - O investimento inicial (funding) usa o book value como "custo de oportunidade"
      (é o valor que se abriria mão de vender o device ao invés de renovar)

    Args:
        asset: O ativo original.
        contract: Parâmetros do contrato original (usado para obter term_months e renewal_months).
        params: Parâmetros de pricing.
        dep_schedule: Schedule de depreciação do ativo (calculado com eco_total completo).
        curves: Dicionário de curvas de depreciação.

    Retorna:
        float com o preço mensal de renovação calculado.
        Retorna 0.0 se não for possível calcular.
    """
    term = contract.term_months
    renewal_months = contract.renewal_months

    # Valor do ativo no início da renovação (mês econômico = term_months)
    book_at_renewal_start = dep_schedule[min(term, len(dep_schedule) - 1)]

    # Cria um asset "virtual" com o valor depreciado como market_price
    # purchase_price é mantido pois reflete o custo histórico de manutenção
    # mas para o cálculo de renovação, usamos o book value como base de funding
    renewal_asset = Asset(
        id=asset.id + "_renewal",
        name=asset.name + " (Renovação)",
        category=asset.category,
        purchase_price=book_at_renewal_start,  # "custo" do ativo depreciado
        market_price=book_at_renewal_start,    # valor de mercado atual
        maintenance_annual_pct=asset.maintenance_annual_pct,
        storage=asset.storage,
        generation=asset.generation,
    )

    # Contrato de renovação: standalone (sem nova renovação), pelo período de renewal_months
    renewal_contract = ContractParams(
        client_type=contract.client_type,
        term_months=renewal_months,
        with_final_sale=contract.with_final_sale,
        is_standalone=True,  # IMPORTANTE: evita recursão e zera crédito PIS/COFINS
    )

    # Calcula o schedule de depreciação para o período de renovação
    # Partindo do book value no início da renovação até o fim do ciclo econômico
    dep_params = get_dep_params(asset.category, curves)
    renewal_dep_schedule = build_dep_schedule(
        market_price=book_at_renewal_start,
        eco_total=renewal_months,
        floor_pct=dep_params["floor_pct"],
        dep_method=dep_params.get("dep_method", "linear"),
        annual_rate=dep_params.get("annual_rate"),
    )
    # Valor de venda ao final da renovação (stepped ou exponencial sobre o valor depreciado)
    renewal_sale_value = get_sale_book_value(
        market_price=book_at_renewal_start,
        eco_total=renewal_months,
        dep_params=dep_params,
    )

    # Builder do vetor anual desalavancado de renovação
    def renewal_cf_builder(P: float) -> np.ndarray:
        _, _, md = build_cashflows(
            asset=renewal_asset,
            contract=renewal_contract,
            params=params,
            P=P,
            renewal_price=0.0,           # standalone: sem nova renovação
            dep_schedule=renewal_dep_schedule,
            curves=curves,
            sale_book_value=renewal_sale_value,
        )
        vec_unlev, _ = build_irr_vectors(md)
        return vec_unlev

    # Encontra o preço de renovação que atinge a TIR mínima
    renewal_price_raw = find_price_for_irr(
        cashflow_builder=renewal_cf_builder,
        target_annual_irr=params.min_unlevered_irr,
    )

    if renewal_price_raw is None:
        # Fallback: tenta com TIR = 0 (breakeven)
        renewal_price_raw = find_price_for_irr(
            cashflow_builder=renewal_cf_builder,
            target_annual_irr=0.0,
        )

    if renewal_price_raw is None:
        return 0.0

    # Arredonda para o padrão X9,90
    return round_to_90(renewal_price_raw)
