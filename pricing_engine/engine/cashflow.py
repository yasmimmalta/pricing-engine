"""
Módulo de construção do fluxo de caixa da Allu Pricing Engine.

Implementa fielmente a lógica financeira da Allu:
- Fluxo desalavancado (CF_unlev): perspectiva do ativo/projeto
- Fluxo alavancado (CF_lev): perspectiva do equity, considerando a dívida (captação)
- Crédito de PIS/COFINS sobre o preço de aquisição
- Sem gap entre contrato inicial e renovação (renovação imediata)
- Custos pontuais (CAC, logística, risk query) no início e na renovação
- DRE da assinatura e DRE da venda (mensais)
- IR/CSLL sobre EBT (23,80%) — incide quando EBT > 0; entra em cf_unlev
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from pricing_engine.config import lookup_customer_benefits
from pricing_engine.engine.depreciation import calendar_to_economic
from pricing_engine.models.asset import Asset
from pricing_engine.models.contract import ClientType, ContractParams, PricingParams


def build_cashflows(
    asset: Asset,
    contract: ContractParams,
    params: PricingParams,
    P: float,
    renewal_price: float,
    dep_schedule: List[float],
    curves: dict,
    sale_book_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Constrói os fluxos de caixa desalavancado e alavancado para um dado preço P.

    O fluxo tem comprimento total_calendar_months + 1:
      - Índice 0: t=0 (desembolso inicial de aquisição)
      - Índices 1..N: meses calendário 1 até total_calendar_months

    Args:
        asset: O ativo (device) a ser precificado.
        contract: Parâmetros do contrato (prazo, tipo de cliente, renovação).
        params: Parâmetros de pricing (tributário, operacional, risco).
        P: Preço mensal do contrato inicial (variável de otimização).
        renewal_price: Preço mensal da renovação (pré-calculado; 0 se sem renovação).
        dep_schedule: Schedule de depreciação: dep_schedule[eco_m] = book value.
                      Usado para default mensal (M12). Deve ter comprimento >= eco_total + 1.
        curves: Dicionário de curvas de depreciação.
        sale_book_value: Valor de venda final do ativo (M5). Quando fornecido,
                         substitui dep_schedule[eco_final] no cálculo da venda.

    Retorna:
        Tupla (cf_unlev, cf_lev, monthly_details):
          - cf_unlev: np.ndarray com o fluxo desalavancado (tamanho N+1)
          - cf_lev: np.ndarray com o fluxo alavancado (tamanho N+1)
          - monthly_details: lista de dicts com breakdown de cada período
    """
    total_cal  = contract.total_calendar_months
    term       = contract.term_months
    has_renew  = contract.has_renewal
    renew_start = contract.renewal_start_calendar_month
    eco_total  = contract.eco_total
    is_b2c     = (contract.client_type == ClientType.B2C)

    # ----------------------------------------------------------
    # Provisão mensal de logística AT (troca por estrago)
    # logistica_AT_anual  = logistics_assinatura × logistics_at_pct × 4
    # logistica_AT_mensal = / 12 (constante, independe do prazo)
    # ----------------------------------------------------------
    logistics_troca_monthly = (
        params.logistics_at_pct * params.logistics_assinatura * 4
    ) / 12

    # ----------------------------------------------------------
    # Depreciação contábil
    #
    # dep_contabil_mensal: valor mensal reconhecido na DRE da venda
    #   = purchase_price × dep_contabil_pct / 12   (constante em todos os meses)
    #
    # accumulated_dep_contabil: depreciação acumulada ao final do ciclo
    #   = dep_contabil_mensal × eco_total
    #   → usada no custo da venda (linha "Custo da venda" = NBV)
    #
    # NBV (valor líquido contábil) = purchase_price − accumulated_dep
    # ----------------------------------------------------------
    dep_contabil_mensal      = asset.purchase_price * (params.dep_contabil_pct / 12)
    accumulated_dep_contabil = dep_contabil_mensal * eco_total
    nbv_at_sale              = asset.purchase_price - accumulated_dep_contabil

    # ----------------------------------------------------------
    # Alíquota de impostos sobre receita para DRE (PIS/COFINS + ISS)
    # = 9,25% + 2,5% = 11,75%
    #
    # Regra de reconhecimento:
    #   - Receita: reconhecida integralmente no primeiro mês de cada contrato/renovação
    #              receita_mes1 = price × prazo_do_período
    #   - Imposto: diferido mensalmente ao longo do prazo
    #              imposto_mensal = price × dre_tax_rate (constante por mês ativo)
    # ----------------------------------------------------------
    dre_tax_rate = params.pis_cofins_pct + params.iss_pct  # 11,75%

    # ----------------------------------------------------------
    # t=0: Aquisição do device
    # ----------------------------------------------------------
    funding      = asset.purchase_price * (1 + params.icms_pct)
    captacao_fee = funding * params.funding_captacao_pct

    icms_value = asset.purchase_price * params.icms_pct   # ICMS: vai para FCO e DRE

    cf_unlev_0 = -funding
    cf_lev_0   = -captacao_fee

    cf_unlev = np.zeros(total_cal + 1)
    cf_lev   = np.zeros(total_cal + 1)
    cf_unlev[0] = cf_unlev_0
    cf_lev[0]   = cf_lev_0

    # ----------------------------------------------------------
    # Crédito de PIS/COFINS sobre o preço de aquisição
    # ----------------------------------------------------------
    credit_saldo = asset.purchase_price * params.pis_cofins_credit_pct
    if contract.is_standalone:
        credit_saldo = 0.0

    monthly_rate = params.debt_annual_rate / 12

    monthly_details: List[Dict] = []

    # ----------------------------------------------------------
    # Detalhe do período 0
    # ----------------------------------------------------------
    monthly_details.append({
        "cal_m": 0,
        "eco_m": 0,
        "tipo": "aquisição",
        "preco_mensal": 0.0,
        "funding": funding,
        "captacao_fee": captacao_fee,
        "receita_bruta": 0.0,
        "pis_cofins_bruto": 0.0,
        "credit_used": 0.0,
        "credit_saldo_fim": credit_saldo,
        "pis_cofins_liq": 0.0,
        "iss": 0.0,
        "mdr": 0.0,
        "receita_liq": 0.0,
        "manutencao": 0.0,
        "customer_benefits": 0.0,
        "pdd": 0.0,
        "default_m": 0.0,
        "cac": 0.0,
        "risk_query": 0.0,
        "logistics_ass": 0.0,
        "logistics_prov_troca": 0.0,
        "logistics_devolucao": 0.0,
        "sale_net": 0.0,
        "juros_divida": 0.0,
        "principal_divida": 0.0,
        "book_value": dep_schedule[0],
        "cf_unlev": cf_unlev_0,
        "cf_lev": cf_lev_0,
        "cf_unlev_acum": cf_unlev_0,
        # DFC — Fluxo de Caixa por Natureza (t=0)
        "fco_receita_bruta": 0.0,
        "fco_pis_cofins": 0.0,
        "fco_iss": 0.0,
        "fco_mdr": 0.0,
        "fco_manutencao": 0.0,
        "fco_pdd": 0.0,
        "fco_default": 0.0,
        "fco_logistics_troca": 0.0,
        "fco_cac": 0.0,
        "fco_risk_query": 0.0,
        "fco_logistics_ass": 0.0,
        "fco_customer_benefits": 0.0,
        "fco_logistics_devolucao": 0.0,
        "fco_ir_csll": 0.0,
        "fco_icms": icms_value,
        "fco_total": -icms_value,
        "fci_compra": asset.purchase_price,
        "fci_sale_val": 0.0,
        "fci_cac_venda": 0.0,
        "fci_prep_venda": 0.0,
        "fci_logistics_venda": 0.0,
        "fci_total": -asset.purchase_price,
        "fcf_funding": funding,
        "fcf_captacao_fee": captacao_fee,
        "fcf_juros": 0.0,
        "fcf_principal": 0.0,
        "fcf_total": funding - captacao_fee,
        "cf_total": -captacao_fee,
        # DRE Assinatura
        "dre_ass_receita": 0.0,
        "dre_ass_impostos": 0.0,
        "dre_ass_receita_liq": 0.0,
        "dre_ass_customer_benefits": 0.0,
        "dre_ass_manutencao": 0.0,
        "dre_ass_logistica": 0.0,
        "dre_ass_credito_pis": 0.0,
        "dre_ass_icms": icms_value,
        "dre_ass_lucro_bruto": -icms_value,
        "dre_ass_risk_query": 0.0,
        "dre_ass_cac": 0.0,
        "dre_ass_pdd": 0.0,
        "dre_ass_default": 0.0,
        "dre_ass_ebitda": -icms_value,
        "dre_ass_ebit": -icms_value,
        # DRE Venda
        "dre_venda_receita": 0.0,
        "dre_venda_dep_acum": 0.0,
        "dre_venda_custo_venda": 0.0,
        "dre_venda_lucro_bruto": 0.0,
        "dre_venda_cac": 0.0,
        "dre_venda_logistica": 0.0,
        "dre_venda_prep": 0.0,
        "dre_venda_ebitda": 0.0,
        "dre_venda_dep_contabil": 0.0,
        "dre_venda_ebit": 0.0,
        # Consolidação DRE
        "dre_ebit_consolidado": -icms_value,
        "dre_juros": 0.0,
        "dre_mdr": 0.0,
        "dre_ebt": -icms_value,
        "dre_ir_csll": -params.ir_csll_pct * (-icms_value),
        "dre_lucro_liq": -icms_value + (-params.ir_csll_pct * (-icms_value)),
    })

    # ----------------------------------------------------------
    # Loop pelos meses calendário 1..total_cal
    # ----------------------------------------------------------
    for cal_m in range(1, total_cal + 1):

        eco_m = calendar_to_economic(cal_m, term, has_renew)
        eco_m_safe = min(eco_m, eco_total)
        book_val = dep_schedule[eco_m_safe]

        is_renewal_start = has_renew and (cal_m == renew_start)
        is_last_month    = (cal_m == total_cal)
        is_last_initial  = has_renew and (cal_m == term)  # último mês do contrato inicial

        # --------------------------------------------------------
        # MESES NORMAIS (contrato inicial ou renovação, sem gap)
        # --------------------------------------------------------

        if cal_m <= term:
            price_m = P
        else:
            price_m = renewal_price

        # --------------------------------------------------
        # TRIBUTOS SOBRE RECEITA
        # --------------------------------------------------
        pis_bruto    = price_m * params.pis_cofins_pct
        credit_used  = min(pis_bruto, credit_saldo)
        pis_liq      = pis_bruto - credit_used
        credit_saldo -= credit_used

        iss_m = price_m * params.iss_pct
        mdr_m = price_m * params.mdr_pct

        # Receita líquida para cálculo do cashflow (deduz MDR — cash outflow real)
        receita_liq = price_m - pis_liq - iss_m - mdr_m

        # --------------------------------------------------
        # CUSTOS RECORRENTES
        # --------------------------------------------------
        maintenance_m     = asset.maintenance_monthly if cal_m <= eco_total else 0.0
        pdd_m             = price_m * params.pdd_pct
        default_m         = asset.purchase_price * (params.default_pct / 12) if cal_m <= eco_total else 0.0
        logistics_troca_m = logistics_troca_monthly

        total_recorrentes = maintenance_m + pdd_m + default_m + logistics_troca_m

        # --------------------------------------------------
        # CUSTOS PONTUAIS
        # --------------------------------------------------
        cac_m          = 0.0
        risk_query_m   = 0.0
        logistics_m    = 0.0
        customer_ben_m = 0.0

        is_start = (cal_m == 1) or is_renewal_start

        if is_start:
            logistics_m    = params.logistics_assinatura
            customer_ben_m = lookup_customer_benefits(
                asset.category, getattr(asset, "condicao", "novo"), params.customer_benefits
            )

            if is_b2c:
                cac_m        = price_m
                risk_query_m = params.risk_query_value

        # Logística de devolução: no último mês de CADA contrato (inicial e renovação)
        logistics_devolucao_m = params.logistics_assinatura if (is_last_month or is_last_initial) else 0.0

        total_pontuais = cac_m + risk_query_m + logistics_m + logistics_devolucao_m + customer_ben_m

        # --------------------------------------------------
        # VENDA DO DEVICE AO FINAL DO CICLO
        # --------------------------------------------------
        sale_net         = 0.0
        fci_sale_val_m   = 0.0
        fci_cac_venda_m  = 0.0
        fci_prep_venda_m = 0.0
        fci_log_venda_m  = 0.0

        if is_last_month and contract.with_final_sale:
            eco_final = min(eco_total, eco_m_safe)
            sale_val  = sale_book_value if sale_book_value is not None else dep_schedule[eco_final]

            fci_sale_val_m   = sale_val
            fci_cac_venda_m  = sale_val * params.cac_venda_pct
            fci_prep_venda_m = sale_val * params.prep_venda
            fci_log_venda_m  = params.logistics_venda
            sale_net = fci_sale_val_m - fci_cac_venda_m - fci_prep_venda_m - fci_log_venda_m

        # --------------------------------------------------
        # SERVIÇO DA DÍVIDA
        # Prazo fixo: juros até prazo_divida_meses, amortização no mês seguinte
        # --------------------------------------------------
        prazo_div = params.prazo_divida_meses
        juros_divida     = funding * monthly_rate if cal_m <= prazo_div else 0.0
        principal_divida = funding if cal_m == prazo_div + 1 else 0.0

        # ==================================================
        # DRE DA ASSINATURA
        # ==================================================
        # Receita: reconhecida integralmente no PRIMEIRO mês de cada contrato/renovação
        #   - Mês 1:           dre_ass_receita = P × term_months
        #   - Mês renovação:   dre_ass_receita = renewal_price × renewal_months
        #   - Demais meses:    dre_ass_receita = 0
        #
        # Impostos: diferidos mensalmente ao longo do prazo
        #   - dre_ass_impostos = price_m × dre_tax_rate  (= price_m × 11,75%)
        #   - O total do ciclo = price_m × prazo × 11,75%  (correto por construção)
        #   - MDR não entra aqui — permanece abaixo do EBIT como despesa financeira
        if cal_m == 1:
            dre_ass_receita = price_m * term
        elif is_renewal_start:
            dre_ass_receita = price_m * contract.renewal_months
        else:
            dre_ass_receita = 0.0

        dre_ass_impostos    = price_m * dre_tax_rate   # diferido mensalmente
        dre_ass_receita_liq = dre_ass_receita - dre_ass_impostos

        # ICMS: reconhecido 100% no t=0 (Ano 0), não nos meses operacionais
        dre_ass_icms = 0.0

        # Custos (crédito PIS reduz o custo)
        dre_ass_logistica = logistics_m + logistics_troca_m + logistics_devolucao_m
        dre_ass_custos    = (customer_ben_m + maintenance_m + dre_ass_logistica - credit_used)
        dre_ass_lucro_bruto = dre_ass_receita_liq - dre_ass_custos

        # Despesas operacionais
        dre_ass_despesas  = risk_query_m + cac_m + pdd_m + default_m
        dre_ass_ebitda    = dre_ass_lucro_bruto - dre_ass_despesas
        dre_ass_ebit      = dre_ass_ebitda  # sem linha de depreciação na DRE de assinatura

        # ==================================================
        # DRE DA VENDA
        # ==================================================
        # Linha "Depreciação contábil" (TODOS os meses):
        #   dep_contabil_mensal = purchase_price × dep_contabil_pct / 12
        #   → aparece mensalmente na linha de Depreciação do EBITDA → EBIT
        #   → EBIT_venda = EBITDA_venda − dep_contabil_mensal
        #
        # Linha "Custo da venda" (somente último mês com venda):
        #   custo_venda = NBV = purchase_price − dep_acumulada
        #   dep_acumulada = dep_contabil_mensal × eco_total
        # ==================================================
        dre_venda_receita      = 0.0
        dre_venda_dep_acum     = 0.0
        dre_venda_custo_venda  = 0.0
        dre_venda_lucro_bruto  = 0.0
        dre_venda_cac          = 0.0
        dre_venda_logistica    = 0.0
        dre_venda_prep         = 0.0
        dre_venda_ebitda       = 0.0

        if is_last_month and contract.with_final_sale:
            # Custo da venda = NBV = purchase_price − depreciação acumulada contábil
            dre_venda_receita      = fci_sale_val_m
            dre_venda_dep_acum     = accumulated_dep_contabil
            dre_venda_custo_venda  = nbv_at_sale
            dre_venda_lucro_bruto  = dre_venda_receita - dre_venda_custo_venda
            dre_venda_cac          = fci_cac_venda_m
            dre_venda_logistica    = fci_log_venda_m
            dre_venda_prep         = fci_prep_venda_m
            dre_venda_ebitda       = dre_venda_lucro_bruto - (dre_venda_cac + dre_venda_logistica + dre_venda_prep)

        # Depreciação mensal: corre em TODOS os meses (inclusive não-terminais)
        # Posição: abaixo do EBITDA da venda → bridge para EBIT
        dre_venda_dep_contabil = dep_contabil_mensal
        dre_venda_ebit         = dre_venda_ebitda - dre_venda_dep_contabil

        # ==================================================
        # CONSOLIDAÇÃO DRE
        # ==================================================
        dre_ebit_consolidado = dre_ass_ebit + dre_venda_ebit

        # Despesas financeiras: juros da dívida + MDR (reclassificado da receita)
        dre_juros = juros_divida
        dre_mdr   = mdr_m

        dre_ebt = dre_ebit_consolidado - dre_juros - dre_mdr

        # IR/CSLL: 23,80% sobre EBT — simétrico:
        #   EBT > 0 → IR negativo (despesa)
        #   EBT < 0 → IR positivo (crédito fiscal)
        dre_ir_csll  = -params.ir_csll_pct * dre_ebt
        dre_lucro_liq = dre_ebt + dre_ir_csll

        # --------------------------------------------------
        # IR/CSLL — SOMENTE NA DRE (não entra no fluxo de caixa)
        # --------------------------------------------------
        ir_fcff = 0.0   # não impacta FCFF
        ir_fcfe = 0.0   # não impacta FCFE

        # --------------------------------------------------
        # FLUXO DE CAIXA DESALAVANCADO (FCFF)
        # Sem IR — IR fica apenas na DRE
        # --------------------------------------------------
        cf_m_unlev = receita_liq - total_recorrentes - total_pontuais + sale_net

        # --------------------------------------------------
        # FLUXO DE CAIXA ALAVANCADO (FCFE)
        # FCFE = FCFF - juros - principal (sem IR)
        # --------------------------------------------------
        cf_m_lev = (receita_liq - total_recorrentes - total_pontuais + sale_net
                    - juros_divida - principal_divida)

        # --------------------------------------------------
        # DFC — Fluxo de Caixa por Natureza
        # IR/CSLL removido do FCO
        # FCI: sem ICMS (ICMS está no FCO do t=0)
        # --------------------------------------------------
        fco_total_m = receita_liq - total_recorrentes - total_pontuais
        fci_total_m = sale_net
        fcf_total_m = -juros_divida - principal_divida

        cf_unlev[cal_m] = cf_m_unlev
        cf_lev[cal_m]   = cf_m_lev

        if cal_m == 1:
            tipo = "início_contrato"
        elif is_last_month and not is_renewal_start:
            tipo = "fim_ciclo"
        elif is_renewal_start:
            tipo = "início_renovação"
        elif cal_m <= term:
            tipo = "contrato_inicial"
        else:
            tipo = "renovação"

        monthly_details.append({
            "cal_m": cal_m,
            "eco_m": eco_m,
            "tipo": tipo,
            "preco_mensal": price_m,
            "funding": 0.0,
            "captacao_fee": 0.0,
            "receita_bruta": price_m,
            "pis_cofins_bruto": pis_bruto,
            "credit_used": credit_used,
            "credit_saldo_fim": credit_saldo,
            "pis_cofins_liq": pis_liq,
            "iss": iss_m,
            "mdr": mdr_m,
            "receita_liq": receita_liq,
            "manutencao": maintenance_m,
            "customer_benefits": customer_ben_m,
            "pdd": pdd_m,
            "default_m": default_m,
            "cac": cac_m,
            "risk_query": risk_query_m,
            "logistics_ass": logistics_m,
            "logistics_prov_troca": logistics_troca_m,
            "logistics_devolucao": logistics_devolucao_m,
            "sale_net": sale_net,
            "juros_divida": juros_divida,
            "principal_divida": principal_divida,
            "book_value": book_val,
            "cf_unlev": cf_m_unlev,
            "cf_lev": cf_m_lev,
            "cf_unlev_acum": 0.0,
            # DFC — Fluxo de Caixa por Natureza
            "fco_receita_bruta": price_m,
            "fco_pis_cofins": pis_liq,
            "fco_iss": iss_m,
            "fco_mdr": mdr_m,
            "fco_manutencao": maintenance_m,
            "fco_pdd": pdd_m,
            "fco_default": default_m,
            "fco_logistics_troca": logistics_troca_m,
            "fco_cac": cac_m,
            "fco_risk_query": risk_query_m,
            "fco_logistics_ass": logistics_m,
            "fco_customer_benefits": customer_ben_m,
            "fco_logistics_devolucao": logistics_devolucao_m,
            "fco_ir_csll": 0.0,            # IR/CSLL removido do fluxo de caixa
            "fco_icms": 0.0,               # ICMS pago no t=0, não nos meses operacionais
            "ir_fcff": ir_fcff,
            "ir_fcfe": ir_fcfe,
            "fco_total": fco_total_m,
            "fci_compra": 0.0,
            "fci_sale_val": fci_sale_val_m,
            "fci_cac_venda": fci_cac_venda_m,
            "fci_prep_venda": fci_prep_venda_m,
            "fci_logistics_venda": fci_log_venda_m,
            "fci_total": fci_total_m,
            "fcf_funding": 0.0,
            "fcf_captacao_fee": 0.0,
            "fcf_juros": juros_divida,
            "fcf_principal": principal_divida,
            "fcf_total": fcf_total_m,
            "cf_total": cf_m_lev,
            # DRE Assinatura
            "dre_ass_receita": dre_ass_receita,
            "dre_ass_impostos": dre_ass_impostos,
            "dre_ass_receita_liq": dre_ass_receita_liq,
            "dre_ass_customer_benefits": customer_ben_m,
            "dre_ass_manutencao": maintenance_m,
            "dre_ass_logistica": dre_ass_logistica,
            "dre_ass_credito_pis": credit_used,
            "dre_ass_icms": dre_ass_icms,
            "dre_ass_lucro_bruto": dre_ass_lucro_bruto,
            "dre_ass_risk_query": risk_query_m,
            "dre_ass_cac": cac_m,
            "dre_ass_pdd": pdd_m,
            "dre_ass_default": default_m,
            "dre_ass_ebitda": dre_ass_ebitda,
            "dre_ass_ebit": dre_ass_ebit,
            # DRE Venda
            "dre_venda_receita": dre_venda_receita,
            "dre_venda_dep_acum": dre_venda_dep_acum,
            "dre_venda_custo_venda": dre_venda_custo_venda,
            "dre_venda_lucro_bruto": dre_venda_lucro_bruto,
            "dre_venda_cac": dre_venda_cac,
            "dre_venda_logistica": dre_venda_logistica,
            "dre_venda_prep": dre_venda_prep,
            "dre_venda_ebitda": dre_venda_ebitda,
            "dre_venda_dep_contabil": dre_venda_dep_contabil,
            "dre_venda_ebit": dre_venda_ebit,
            # Consolidação DRE
            "dre_ebit_consolidado": dre_ebit_consolidado,
            "dre_juros": dre_juros,
            "dre_mdr": dre_mdr,
            "dre_ebt": dre_ebt,
            "dre_ir_csll": dre_ir_csll,
            "dre_lucro_liq": dre_lucro_liq,
        })

    # ----------------------------------------------------------
    # Preenche os fluxos acumulados
    # ----------------------------------------------------------
    acum_u = 0.0
    acum_l = 0.0
    for detail in monthly_details:
        acum_u += detail["cf_unlev"]
        acum_l += detail["cf_lev"]
        detail["cf_unlev_acum"] = acum_u
        detail["cf_lev_acum"] = acum_l

    return cf_unlev, cf_lev, monthly_details
