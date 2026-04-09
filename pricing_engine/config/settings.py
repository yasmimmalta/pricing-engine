"""
Configurações e parâmetros padrão da Allu Pricing Engine.

ATENÇÃO: Este arquivo é o fallback usado quando PricingParams() é instanciado
sem argumentos. A fonte de verdade para execução é params.csv, lido por run_pricing.py.
Mantenha sincronizado com params.csv.
"""

DEFAULT_PARAMS: dict = {
    # ----------------------------------------------------------
    # FISCAL E TRIBUTÁRIO
    # ----------------------------------------------------------

    # ICMS sobre preço de compra do device — 14%
    "icms_pct": 0.14,

    # Custo de captação sobre o funding total (upfront, t=0) — 2%
    # Funding = purchase_price * (1 + icms_pct)
    "funding_captacao_pct": 0.02,

    # Taxa de juros anual da dívida (COMPOSTA) — 23,72% a.a.
    # Taxa mensal efetiva ≈ 1,79%/mês = (1+0.2372)^(1/12) - 1
    "debt_annual_rate": 0.2372,

    # PIS + COFINS sobre receita bruta: PIS 1,65% + COFINS 7,60% = 9,25%
    "pis_cofins_pct": 0.0925,

    # Crédito de PIS/COFINS sobre o preço de aquisição do device — 9,25%
    "pis_cofins_credit_pct": 0.0925,

    # ISS (Imposto Sobre Serviços) sobre receita — 2,5%
    "iss_pct": 0.025,

    # MDR (taxa da maquininha/gateway de pagamento) sobre receita — 2%
    "mdr_pct": 0.02,

    # ----------------------------------------------------------
    # CUSTOS COMERCIAIS E OPERACIONAIS
    # ----------------------------------------------------------

    # CAC como percentual da venda do device ao final do ciclo — 10%
    "cac_venda_pct": 0.10,

    # Custo de preparação do device para venda — 2% do valor de venda
    "prep_venda": 0.02,

    # Logística para envio do device vendido ao comprador — R$ 85
    "logistics_venda": 85.0,

    # Logística de assinatura: envio ao assinante no início e na renovação — R$ 85
    "logistics_assinatura": 85.0,

    # Consulta de risco de crédito (B2C only) — R$ 16
    "risk_query_value": 16.0,

    # Benefícios ao cliente (pontual: mês 1 e início renovação) — R$ 67
    "customer_benefits": 67.0,

    # Logistica AT: % anual usado na provisão de logística para troca por estrago — 1,42%
    "logistics_at_pct": 0.0142,

    # ----------------------------------------------------------
    # RISCO E INADIMPLÊNCIA
    # ----------------------------------------------------------

    # PDD (Provisão para Devedores Duvidosos) — 8% da mensalidade
    "pdd_pct": 0.08,

    # Default anual sobre o valor líquido contábil do ativo — 3% a.a.
    # Fórmula: default_mensal = default_pct * (valor_liquido_m / prazo_meses)
    "default_pct": 0.03,

    # ----------------------------------------------------------
    # DEPRECIAÇÃO CONTÁBIL (base do cálculo de default M12)
    # ----------------------------------------------------------

    # Taxa de depreciação contábil anual — fallback universal
    # O dep_contabil.json por categoria tem prioridade sobre este valor
    "dep_contabil_pct": 0.20,

    # ----------------------------------------------------------
    # OTIMIZAÇÃO E TIR ALVO
    # ----------------------------------------------------------

    # TIR mínima desalavancada (hurdle rate) — 30% a.a.
    "min_unlevered_irr": 0.30,

    # ----------------------------------------------------------
    # PRAZO DA DÍVIDA
    # ----------------------------------------------------------

    # Prazo médio da dívida: juros até mês 30, amortização no mês 31
    "prazo_divida_meses": 30,
}
