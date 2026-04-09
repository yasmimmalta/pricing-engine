"""
Módulo de exportação de resultados da Allu Pricing Engine.

Permite exportar os resultados de pricing para CSV e DataFrame pandas.
"""

import os
from typing import List

import pandas as pd

from pricing_engine.models.result import PricingResult


def result_to_dataframe(result: PricingResult) -> pd.DataFrame:
    """
    Converte o detalhamento mensal de um PricingResult em DataFrame pandas.

    O DataFrame contém uma linha por mês (incluindo t=0), com todos os
    componentes do fluxo de caixa.

    Args:
        result: PricingResult com monthly_details preenchido.

    Retorna:
        pd.DataFrame com colunas para cada componente financeiro.
        Retorna DataFrame vazio se monthly_details estiver vazio.
    """
    if not result.monthly_details:
        return pd.DataFrame()

    df = pd.DataFrame(result.monthly_details)

    # Adiciona colunas de identificação para facilitar análise
    df.insert(0, "ativo_id", result.asset.id)
    df.insert(1, "ativo_nome", result.asset.name)
    df.insert(2, "cliente", result.contract.client_type.value)
    df.insert(3, "prazo_meses", result.contract.term_months)
    df.insert(4, "preco_contrato", result.suggested_price)

    # Renomeia colunas para nomes mais descritivos em português
    rename_map = {
        "cal_m": "mes_calendario",
        "eco_m": "mes_economico",
        "tipo": "tipo_mes",
        "preco_mensal": "preco_mensal_R$",
        "receita_bruta": "receita_bruta_R$",
        "pis_cofins_bruto": "pis_cofins_bruto_R$",
        "credit_used": "credito_pis_cofins_usado_R$",
        "credit_saldo_fim": "saldo_credito_pis_cofins_R$",
        "pis_cofins_liq": "pis_cofins_liq_R$",
        "iss": "iss_R$",
        "mdr": "mdr_R$",
        "receita_liq": "receita_liquida_R$",
        "manutencao": "manutencao_R$",
        "customer_benefits": "beneficios_cliente_R$",
        "pdd": "pdd_R$",
        "default_m": "default_R$",
        "cac": "cac_R$",
        "risk_query": "consulta_risco_R$",
        "logistics_ass": "logistica_assinatura_R$",
        "logistics_prov_troca": "logistica_prov_troca_R$",
        "logistics_devolucao": "logistica_devolucao_R$",
        "sale_net": "venda_liq_R$",
        "juros_divida": "juros_divida_R$",
        "principal_divida": "principal_divida_R$",
        "book_value": "book_value_R$",
        "cf_unlev": "fluxo_desalavancado_R$",
        "cf_lev": "fluxo_alavancado_R$",
        "cf_unlev_acum": "fluxo_desalav_acumulado_R$",
        # DFC — Fluxo de Caixa por Natureza
        "fco_receita_bruta": "dfc_fco_receita_bruta_R$",
        "fco_pis_cofins": "dfc_fco_pis_cofins_R$",
        "fco_iss": "dfc_fco_iss_R$",
        "fco_mdr": "dfc_fco_mdr_R$",
        "fco_manutencao": "dfc_fco_manutencao_R$",
        "fco_pdd": "dfc_fco_pdd_R$",
        "fco_default": "dfc_fco_default_R$",
        "fco_logistics_troca": "dfc_fco_logistics_troca_R$",
        "fco_cac": "dfc_fco_cac_R$",
        "fco_risk_query": "dfc_fco_risk_query_R$",
        "fco_logistics_ass": "dfc_fco_logistics_ass_R$",
        "fco_customer_benefits": "dfc_fco_customer_benefits_R$",
        "fco_logistics_devolucao": "dfc_fco_logistics_devolucao_R$",
        "fco_ir_csll": "dfc_fco_ir_csll_R$",
        "fco_icms": "dfc_fco_icms_R$",
        "fco_total": "dfc_fco_total_R$",
        "fci_compra": "dfc_fci_compra_R$",
        "fci_sale_val": "dfc_fci_venda_bruta_R$",
        "fci_cac_venda": "dfc_fci_cac_venda_R$",
        "fci_prep_venda": "dfc_fci_prep_venda_R$",
        "fci_logistics_venda": "dfc_fci_logistics_venda_R$",
        "fci_total": "dfc_fci_total_R$",
        "fcf_funding": "dfc_fcf_funding_R$",
        "fcf_captacao_fee": "dfc_fcf_captacao_fee_R$",
        "fcf_juros": "dfc_fcf_juros_R$",
        "fcf_principal": "dfc_fcf_principal_R$",
        "fcf_total": "dfc_fcf_total_R$",
        "cf_total": "dfc_cf_total_R$",
        # DRE Assinatura
        "dre_ass_receita":           "dre_ass_receita_R$",
        "dre_ass_impostos":          "dre_ass_impostos_R$",
        "dre_ass_receita_liq":       "dre_ass_receita_liq_R$",
        "dre_ass_customer_benefits": "dre_ass_customer_benefits_R$",
        "dre_ass_manutencao":        "dre_ass_manutencao_R$",
        "dre_ass_logistica":         "dre_ass_logistica_R$",
        "dre_ass_credito_pis":       "dre_ass_credito_pis_R$",
        "dre_ass_icms":              "dre_ass_icms_R$",
        "dre_ass_lucro_bruto":       "dre_ass_lucro_bruto_R$",
        "dre_ass_risk_query":        "dre_ass_risk_query_R$",
        "dre_ass_cac":               "dre_ass_cac_R$",
        "dre_ass_pdd":               "dre_ass_pdd_R$",
        "dre_ass_default":           "dre_ass_default_R$",
        "dre_ass_ebitda":            "dre_ass_ebitda_R$",
        "dre_ass_ebit":              "dre_ass_ebit_R$",
        # DRE Venda
        "dre_venda_receita":         "dre_venda_receita_R$",
        "dre_venda_dep_acum":        "dre_venda_dep_acum_R$",
        "dre_venda_custo_venda":     "dre_venda_custo_venda_R$",
        "dre_venda_lucro_bruto":     "dre_venda_lucro_bruto_R$",
        "dre_venda_cac":             "dre_venda_cac_R$",
        "dre_venda_logistica":       "dre_venda_logistica_R$",
        "dre_venda_prep":            "dre_venda_prep_R$",
        "dre_venda_ebitda":          "dre_venda_ebitda_R$",
        "dre_venda_dep_contabil":    "dre_venda_dep_contabil_R$",
        "dre_venda_ebit":            "dre_venda_ebit_R$",
        # Consolidação DRE
        "dre_ebit_consolidado":      "dre_ebit_consolidado_R$",
        "dre_juros":                 "dre_juros_R$",
        "dre_mdr":                   "dre_mdr_R$",
        "dre_ebt":                   "dre_ebt_R$",
        "dre_ir_csll":               "dre_ir_csll_R$",
        "dre_lucro_liq":             "dre_lucro_liq_R$",
    }
    df = df.rename(columns=rename_map)

    return df


def export_to_csv(result: PricingResult, filepath: str) -> None:
    """
    Exporta o fluxo de caixa mensal detalhado de um PricingResult para CSV.

    Cria o diretório de destino se não existir.

    Args:
        result: PricingResult com monthly_details preenchido.
        filepath: Caminho completo do arquivo CSV de saída.

    Lança:
        ValueError: Se o result não tiver monthly_details.
        OSError: Se não for possível criar o arquivo.
    """
    if not result.monthly_details:
        raise ValueError(
            f"PricingResult para '{result.asset.name}' não tem monthly_details. "
            "Execute price_asset() antes de exportar."
        )

    # Cria o diretório se não existir
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    df = result_to_dataframe(result)
    df.to_csv(filepath, index=False, encoding="utf-8-sig", float_format="%.4f")

    print(f"Fluxo de caixa exportado: {filepath} ({len(df)} linhas)")


def export_summary_csv(results: List[PricingResult], filepath: str) -> None:
    """
    Exporta um resumo comparativo de múltiplos PricingResults para CSV.

    Útil para comparar diferentes cenários (prazos, ativos, tipos de cliente).

    Args:
        results: Lista de PricingResult para comparar.
        filepath: Caminho completo do arquivo CSV de saída.

    Lança:
        ValueError: Se a lista estiver vazia.
        OSError: Se não for possível criar o arquivo.
    """
    if not results:
        raise ValueError("A lista de resultados está vazia.")

    # Cria o diretório se não existir
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Coleta os summaries de cada resultado
    rows = []
    for result in results:
        summary = result.summary_dict()

        # Adiciona informações calculadas extras
        summary["margem_breakeven_pct"] = (
            (result.suggested_price - result.breakeven_price) / result.breakeven_price
            if result.suggested_price and result.breakeven_price and result.breakeven_price > 0
            else None
        )

        rows.append(summary)

    df = pd.DataFrame(rows)

    # Seleciona e ordena as colunas mais relevantes para comparação
    colunas_principais = [
        "ativo_nome",
        "cliente",
        "prazo_meses",
        "ciclo_economico_meses",
        "tem_renovacao",
        "meses_renovacao",
        "preco_sugerido_fmt",
        "preco_breakeven_fmt",
        "preco_renovacao_fmt",
        "tir_desalavancada_fmt",
        "tir_alavancada_fmt",
        "payback_meses",
        "tir_minima_fmt",
    ]

    # Usa apenas as colunas que existem no DataFrame
    colunas_existentes = [c for c in colunas_principais if c in df.columns]
    df_export = df[colunas_existentes]

    df_export.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"Resumo comparativo exportado: {filepath} ({len(df_export)} cenários)")


def export_cashflow_anual_csv(results: List[PricingResult], filepath: str) -> None:
    """
    Exporta a consolidação anual de FCO/FCI/FCF para CSV.

    Uma linha por (ativo, cliente, prazo, ano). Inclui linha de Total por cenário.
    Útil para análise de fluxo de caixa por natureza em FP&A / controladoria.

    Args:
        results: Lista de PricingResult com annual_cashflows preenchido.
        filepath: Caminho completo do arquivo CSV de saída.
    """
    if not results:
        raise ValueError("A lista de resultados está vazia.")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    rows = []
    for result in results:
        if not result.annual_cashflows:
            continue
        for ano_row in result.annual_cashflows:
            row = {
                "ativo_id":    result.asset.id,
                "ativo_nome":  result.asset.name,
                "cliente":     result.contract.client_type.value,
                "prazo_meses": result.contract.term_months,
                "ciclo_meses": result.contract.eco_total,
            }
            row.update(ano_row)
            rows.append(row)

    if not rows:
        print("Nenhum dado anual disponível para exportar.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"Fluxo anual exportado: {filepath} ({len(df)} linhas)")
