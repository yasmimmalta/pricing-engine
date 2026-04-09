"""
Script de demonstração da Allu Pricing Engine.

Executa o cálculo de pricing para ativos de exemplo e exibe os resultados formatados.
"""

import os
import sys

# Configura encoding UTF-8 para o terminal Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd

# Garante que o pacote pricing_engine está no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pricing_engine.engine import price_asset
from pricing_engine.models import Asset, ClientType, ContractParams, PricingParams
from pricing_engine.outputs.exporter import export_summary_csv, export_to_csv


# ----------------------------------------------------------
# Diretório de saída dos arquivos CSV
# ----------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Caminho do CSV de ativos
ASSETS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pricing_engine", "data", "assets.csv"
)


def carregar_ativos(filepath: str) -> dict:
    """
    Carrega os ativos do CSV e retorna um dict {id: Asset}.

    Args:
        filepath: Caminho para o arquivo assets.csv.

    Retorna:
        Dict com os ativos indexados por id.
    """
    df = pd.read_csv(filepath)
    ativos = {}

    for _, row in df.iterrows():
        try:
            ativo = Asset(
                id=str(row["id"]),
                name=str(row["name"]),
                category=str(row["category"]),
                purchase_price=float(row["purchase_price"]),
                market_price=float(row["market_price"]),
                maintenance_annual_pct=float(row["maintenance_annual_pct"]),
            )
            ativos[ativo.id] = ativo
        except Exception as e:
            print(f"  Aviso: erro ao carregar ativo '{row.get('id', '?')}': {e}")

    return ativos


def formatar_resultado(result) -> None:
    """
    Imprime o resultado de pricing de forma formatada no terminal.

    Args:
        result: PricingResult com os dados calculados.
    """
    s = result.summary_dict()
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  RESULTADO DE PRICING — {s['ativo_nome']}")
    print(sep)
    print(f"  Cliente:           {s['cliente']}")
    print(f"  Prazo:             {s['prazo_meses']} meses")
    print(f"  Ciclo econômico:   {s['ciclo_economico_meses']} meses")
    print(f"  Renovação:         {'Sim (' + str(s['meses_renovacao']) + ' meses)' if s['tem_renovacao'] else 'Não'}")
    print(f"  Preço de compra:   {s['preco_compra']:>10.2f} R$")
    print(f"  Preço de mercado:  {s['preco_mercado']:>10.2f} R$")
    print(f"{'-' * 60}")
    print(f"  Preço Sugerido:    {s['preco_sugerido_fmt']:>14}")
    print(f"  Preço Breakeven:   {s['preco_breakeven_fmt']:>14}")
    if s['tem_renovacao']:
        print(f"  Preço Renovação:   {s['preco_renovacao_fmt']:>14}")
    print(f"{'-' * 60}")
    print(f"  TIR Desalavancada: {s['tir_desalavancada_fmt']:>14}  (alvo: {s['tir_minima_fmt']})")
    print(f"  TIR Alavancada:    {s['tir_alavancada_fmt']:>14}")
    print(f"  Payback:           {s['payback_meses'] if s['payback_meses'] else 'N/A':>14} meses")
    print(sep)


def main():
    """Função principal de demonstração."""
    print("\n" + "=" * 60)
    print("   ALLU PRICING ENGINE — Demonstração")
    print("=" * 60)

    # ----------------------------------------------------------
    # Carrega ativos do CSV
    # ----------------------------------------------------------
    print(f"\nCarregando ativos de: {ASSETS_CSV}")
    ativos = carregar_ativos(ASSETS_CSV)
    print(f"  {len(ativos)} ativos carregados: {', '.join(ativos.keys())}")

    # ----------------------------------------------------------
    # Parâmetros padrão
    # ----------------------------------------------------------
    params = PricingParams()
    print(f"\nParâmetros: TIR alvo={params.min_unlevered_irr:.0%}, "
          f"Juros={params.debt_annual_rate:.2%} a.a., "
          f"PIS/COFINS={params.pis_cofins_pct:.2%}")

    # ----------------------------------------------------------
    # Caso principal: iPhone 16 128GB, B2C, 12 meses
    # ----------------------------------------------------------
    print("\n" + "-" * 60)
    print("CASO 1: iPhone 16 128GB - B2C - 12 meses")
    print("-" * 60)

    ativo_principal = ativos["ip16_128"]
    contrato_principal = ContractParams(
        client_type=ClientType.B2C,
        term_months=12,
    )

    resultado_principal = price_asset(ativo_principal, contrato_principal, params)
    formatar_resultado(resultado_principal)

    # Exporta fluxo de caixa detalhado
    csv_detalhe = os.path.join(OUTPUT_DIR, "cashflow_ip16_128_b2c_12m.csv")
    export_to_csv(resultado_principal, csv_detalhe)

    # ----------------------------------------------------------
    # Comparação de todos os prazos para iPhone 16 128GB B2C
    # ----------------------------------------------------------
    print("\n" + "-" * 60)
    print("COMPARACAO: iPhone 16 128GB - B2C - Todos os prazos")
    print("-" * 60)

    resultados_b2c = []
    for prazo in [12, 24, 36]:
        print(f"  Calculando B2C {prazo}m...", end=" ", flush=True)
        try:
            contrato = ContractParams(client_type=ClientType.B2C, term_months=prazo)
            res = price_asset(ativo_principal, contrato, params)
            resultados_b2c.append(res)
            formatar_resultado(res)
        except Exception as e:
            print(f"ERRO: {e}")

    # Exporta comparação B2C
    csv_comp_b2c = os.path.join(OUTPUT_DIR, "comparacao_b2c_ip16_128.csv")
    if resultados_b2c:
        export_summary_csv(resultados_b2c, csv_comp_b2c)

    # ----------------------------------------------------------
    # Comparação de todos os prazos para iPhone 16 128GB B2B
    # ----------------------------------------------------------
    print("\n" + "-" * 60)
    print("COMPARACAO: iPhone 16 128GB - B2B - Todos os prazos")
    print("-" * 60)

    resultados_b2b = []
    for prazo in [12, 24, 36, 48]:
        print(f"  Calculando B2B {prazo}m...", end=" ", flush=True)
        try:
            contrato = ContractParams(client_type=ClientType.B2B, term_months=prazo)
            res = price_asset(ativo_principal, contrato, params)
            resultados_b2b.append(res)
            formatar_resultado(res)
        except Exception as e:
            print(f"ERRO: {e}")

    # Exporta comparação B2B
    csv_comp_b2b = os.path.join(OUTPUT_DIR, "comparacao_b2b_ip16_128.csv")
    if resultados_b2b:
        export_summary_csv(resultados_b2b, csv_comp_b2b)

    # ----------------------------------------------------------
    # Resumo de todos os ativos, B2C 12m
    # ----------------------------------------------------------
    print("\n" + "-" * 60)
    print("RESUMO: Todos os ativos - B2C - 12 meses")
    print("-" * 60)

    resultados_todos = []
    for ativo_id, ativo in ativos.items():
        print(f"  Calculando {ativo.name}...", end=" ", flush=True)
        try:
            contrato = ContractParams(client_type=ClientType.B2C, term_months=12)
            res = price_asset(ativo, contrato, params)
            resultados_todos.append(res)
            s = res.summary_dict()
            print(
                f"Preço: {s['preco_sugerido_fmt']}, "
                f"TIR: {s['tir_desalavancada_fmt']}, "
                f"Payback: {s['payback_meses']}m"
            )
        except Exception as e:
            print(f"ERRO: {e}")

    # Exporta resumo geral
    csv_resumo = os.path.join(OUTPUT_DIR, "resumo_todos_ativos_b2c_12m.csv")
    if resultados_todos:
        export_summary_csv(resultados_todos, csv_resumo)

    print("\n" + "=" * 60)
    print(f"  Arquivos salvos em: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
