"""
Allu Pricing Engine -- Runner Principal

Uso:
    python run_pricing.py

Fluxo:
  1. Lê parâmetros de pricing_engine/data/params.csv
  2. Lê base de ativos de pricing_engine/data/assets.csv
  3. Valida colunas obrigatórias e tipos
  4. Calcula preços para todas as combinações de prazo e tipo de cliente
  5. Salva resultado em outputs/pricing_results.csv
  6. Exibe tabela resumida no terminal

Para adicionar novos ativos: edite assets.csv e rode novamente.
Para ajustar parâmetros:   edite params.csv e rode novamente.
"""

import csv
import dataclasses
import io
import os
import sys
from datetime import datetime

# Força UTF-8 no terminal Windows para exibir acentos corretamente
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

from pricing_engine.config import get_depreciation_curves
from pricing_engine.engine.cashflow import build_cashflows
from pricing_engine.engine.depreciation import build_dep_schedule, get_dep_params, get_sale_book_value
from pricing_engine.engine.optimizer import annual_irr, build_irr_vectors, calculate_ebitda_margin, calculate_payback
from pricing_engine.engine.pricing import price_asset, _compute_annual_cashflows
from pricing_engine.models.asset import Asset
from pricing_engine.models.contract import ClientType, ContractParams, PricingParams
from pricing_engine.outputs.exporter import export_cashflow_anual_csv, export_summary_csv

# --- Caminhos -----------------------------------------------------------------
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH      = os.path.join(BASE_DIR, "pricing_engine", "data", "params.csv")
ASSETS_PATH      = os.path.join(BASE_DIR, "pricing_engine", "data", "assets.csv")
OUTPUT_DIR       = os.path.join(BASE_DIR, "outputs")
OUTPUT_PATH      = os.path.join(OUTPUT_DIR, "pricing_results.csv")
OUTPUT_ANUAL_PATH = os.path.join(OUTPUT_DIR, "cashflow_anual.csv")

# --- Configuração -------------------------------------------------------------
REQUIRED_ASSET_COLUMNS = {
    "id", "name", "category",
    "purchase_price", "market_price", "maintenance_annual_pct",
}

REQUIRED_PARAM_KEYS = {
    "icms_pct", "funding_captacao_pct", "debt_annual_rate",
    "pis_cofins_pct", "pis_cofins_credit_pct", "iss_pct", "mdr_pct",
    "cac_venda_pct", "prep_venda", "logistics_venda", "logistics_assinatura",
    "logistics_at_pct", "risk_query_value", "customer_benefits",
    "pdd_pct", "default_pct", "dep_contabil_pct", "min_unlevered_irr",
    "ir_csll_pct", "prazo_divida_meses",
}

COMBOS = [
    (ClientType.B2C, 12),
    (ClientType.B2C, 24),
    (ClientType.B2C, 36),
    (ClientType.B2B, 12),
    (ClientType.B2B, 24),
    (ClientType.B2B, 36),
    (ClientType.B2B, 48),
]


# --- Terminal helpers ----------------------------------------------------------
def _sep(char="-", width=70):
    print(char * width)


def _header(msg):
    print()
    _sep("-")
    print(f"  {msg}")
    _sep("-")


def _ok(msg):
    print(f"  [OK]  {msg}")


def _warn(msg):
    print(f"  [AVISO]  {msg}", file=sys.stderr)


def _fail(msg):
    print(f"\n  [ERRO]  {msg}\n", file=sys.stderr)
    sys.exit(1)


def _fmt_brl(v):
    if v is None:
        return "--"
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_pct(v):
    if v is None:
        return "--"
    return f"{v * 100:.1f}%"


def _fmt_pb(v):
    if v is None:
        return "--"
    return f"{v}m"


# --- Leitura e validação -------------------------------------------------------
def load_params(path: str) -> dict:
    if not os.path.exists(path):
        _fail(
            f"Arquivo de parâmetros não encontrado:\n"
            f"     {path}\n"
            f"  Certifique-se de que params.csv existe antes de executar."
        )

    params = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        if not reader.fieldnames or "parametro" not in reader.fieldnames:
            _fail(
                f"params.csv com formato inválido.\n"
                f"  Esperado: colunas 'parametro' e 'valor', separadas por ';'."
            )
        for row in reader:
            try:
                params[row["parametro"].strip()] = float(row["valor"].replace(",", "."))
            except (ValueError, KeyError) as e:
                _warn(f"Parâmetro ignorado (linha inválida): {dict(row)} -- {e}")

    # Converte campos inteiros
    if "prazo_divida_meses" in params:
        params["prazo_divida_meses"] = int(params["prazo_divida_meses"])

    missing = REQUIRED_PARAM_KEYS - set(params)
    if missing:
        _fail(
            f"Parâmetros obrigatórios ausentes em params.csv:\n"
            + "".join(f"     - {k}\n" for k in sorted(missing))
        )

    return params


def load_assets(path: str):
    if not os.path.exists(path):
        _fail(
            f"Arquivo de ativos não encontrado:\n"
            f"     {path}\n"
            f"  Adicione seus ativos em assets.csv e rode novamente."
        )

    assets = []
    row_errors = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        if not reader.fieldnames:
            _fail(f"assets.csv está vazio ou sem cabeçalho.")

        cols = {c.strip() for c in reader.fieldnames}
        missing = REQUIRED_ASSET_COLUMNS - cols
        if missing:
            _fail(
                f"Colunas obrigatórias ausentes em assets.csv:\n"
                + "".join(f"     - {c}\n" for c in sorted(missing))
                + f"\n  Colunas encontradas: {', '.join(sorted(cols))}"
                + f"\n  Colunas esperadas:   {', '.join(sorted(REQUIRED_ASSET_COLUMNS))}"
            )

        for i, row in enumerate(reader, start=2):
            row_id = row.get("id", f"linha_{i}").strip()
            try:
                assets.append(Asset(
                    id=row_id,
                    name=row["name"].strip(),
                    category=row["category"].strip(),
                    purchase_price=float(row["purchase_price"].replace(",", ".")),
                    market_price=float(row["market_price"].replace(",", ".")),
                    maintenance_annual_pct=float(row["maintenance_annual_pct"].replace(",", ".")),
                    condicao=row.get("condicao", "novo").strip(),
                ))
            except (ValueError, KeyError) as e:
                row_errors.append(f"Linha {i} (id={row_id}): {e}")

    return assets, row_errors


def build_pricing_params(p: dict) -> PricingParams:
    return PricingParams(
        icms_pct=p["icms_pct"],
        funding_captacao_pct=p["funding_captacao_pct"],
        debt_annual_rate=p["debt_annual_rate"],
        pis_cofins_pct=p["pis_cofins_pct"],
        pis_cofins_credit_pct=p["pis_cofins_credit_pct"],
        iss_pct=p["iss_pct"],
        mdr_pct=p["mdr_pct"],
        cac_venda_pct=p["cac_venda_pct"],
        prep_venda=p["prep_venda"],
        logistics_venda=p["logistics_venda"],
        logistics_assinatura=p["logistics_assinatura"],
        logistics_at_pct=p["logistics_at_pct"],
        risk_query_value=p["risk_query_value"],
        customer_benefits=p["customer_benefits"],
        pdd_pct=p["pdd_pct"],
        default_pct=p["default_pct"],
        dep_contabil_pct=p["dep_contabil_pct"],
        min_unlevered_irr=p["min_unlevered_irr"],
    )


# --- Enforcement da escada de preços -------------------------------------------
MIN_GAP = 10.0   # Diferença mínima entre prazos adjacentes
MAX_GAP = 30.0   # Diferença máxima entre prazos adjacentes


def _enforce_price_ladder(asset_results: dict) -> dict:
    """
    Garante gaps mín R$10 e máx R$30 entre prazos adjacentes.
    Se o cap faz o prazo curto perder viabilidade, propaga o ajuste
    para trás (sobe o prazo anterior).
    """
    from pricing_engine.engine.optimizer import round_to_90

    def enforce_chain(combos_asc):
        for i in range(1, len(combos_asc)):
            key_prev = combos_asc[i - 1]
            key_curr = combos_asc[i]
            r_prev = asset_results.get(key_prev)
            r_curr = asset_results.get(key_curr)
            if r_prev is None or r_curr is None:
                continue
            if r_prev.suggested_price is None or r_curr.suggested_price is None:
                continue

            piso = round_to_90(r_prev.suggested_price + MIN_GAP)
            teto = round_to_90(r_prev.suggested_price + MAX_GAP)
            p = r_curr.suggested_price

            if p < piso:
                p = piso

            if p > teto:
                # Preço necessário não cabe no gap → subir o anterior
                p_prev_min = round_to_90(r_curr.suggested_price - MAX_GAP)
                if p_prev_min > r_prev.suggested_price:
                    asset_results[key_prev] = dataclasses.replace(
                        r_prev, suggested_price=p_prev_min)
                    # Propaga para trás
                    if i - 1 > 0:
                        enforce_chain(combos_asc[:i])
                    # Recalcula teto/piso
                    teto = round_to_90(p_prev_min + MAX_GAP)
                    piso = round_to_90(p_prev_min + MIN_GAP)
                p = min(p, teto)
                p = max(p, piso)

            if p != r_curr.suggested_price:
                asset_results[key_curr] = dataclasses.replace(r_curr, suggested_price=p)

    b2c_asc = [
        (ClientType.B2C, 36),
        (ClientType.B2C, 24),
        (ClientType.B2C, 12),
    ]
    b2b_asc = [
        (ClientType.B2B, 48),
        (ClientType.B2B, 36),
        (ClientType.B2B, 24),
        (ClientType.B2B, 12),
    ]
    enforce_chain(b2c_asc)
    enforce_chain(b2b_asc)
    return asset_results


# --- Exibição da tabela final --------------------------------------------------
def print_results_table(results):
    W = 126
    print()
    print("-" * W)
    print(
        f"  {'ATIVO':<28} {'CLI':<4} {'PRAZO':>5} {'ECO':>4} {'RENOV':>5}"
        f" | {'P.SUGERIDO':>12} {'P.BREAKEVEN':>12} {'PAYBACK':>8}"
        f" {'TIR_UNLEV':>10} {'TIR_LEV':>10} {'MARGEM':>8}"
    )
    print("-" * W)

    prev_id = None
    for r in results:
        label = r.asset.name if r.asset.id != prev_id else ""
        prev_id = r.asset.id
        margem_str = f"{r.margem_ebitda:.1%}" if r.margem_ebitda is not None else "--"
        print(
            f"  {label:<28} {r.contract.client_type.value:<4}"
            f" {r.contract.term_months:>4}m"
            f" {r.contract.eco_total:>3}m"
            f" {(r.contract.renewal_months if r.contract.has_renewal else 0):>4}m"
            f" | {_fmt_brl(r.suggested_price):>12}"
            f" {_fmt_brl(r.breakeven_price):>12}"
            f" {_fmt_pb(r.payback_months):>8}"
            f" {_fmt_pct(r.unlevered_irr):>10}"
            f" {_fmt_pct(r.levered_irr):>10}"
            f" {margem_str:>8}"
        )

    print("-" * W)


# --- Main ----------------------------------------------------------------------
def main():
    started_at = datetime.now()

    _header(f"Allu Pricing Engine  --  {started_at.strftime('%d/%m/%Y %H:%M:%S')}")

    # -- 1. Parâmetros ----------------------------------------------------------
    print(f"\n[1/4] Carregando parâmetros de pricing...")
    p = load_params(PARAMS_PATH)
    pricing_params = build_pricing_params(p)
    _ok(f"{len(p)} parâmetros  <-  {os.path.relpath(PARAMS_PATH)}")

    # -- 2. Ativos -------------------------------------------------------------
    print(f"\n[2/4] Carregando base de ativos...")
    assets, load_errors = load_assets(ASSETS_PATH)

    if load_errors:
        for err in load_errors:
            _warn(f"Ativo ignorado -- {err}")

    if not assets:
        _fail(
            "Nenhum ativo válido encontrado em assets.csv.\n"
            "  Verifique o arquivo e tente novamente."
        )

    _ok(f"{len(assets)} ativo(s)  <-  {os.path.relpath(ASSETS_PATH)}")
    if load_errors:
        print(f"         {len(load_errors)} linha(s) ignorada(s) por erro de leitura")

    # -- 3. Cálculo ------------------------------------------------------------
    total = len(assets) * len(COMBOS)
    print(f"\n[3/4] Calculando preços  ({len(assets)} ativo(s) × {len(COMBOS)} combos = {total} combinações)...")

    results = []
    calc_errors = []
    done = 0

    for asset in assets:
        # ------------------------------------------------------------------
        # Regra de coerência de renovação:
        #   O preço de renovação deve ser idêntico ao preço da assinatura
        #   do prazo correspondente (não calculado sobre ativo depreciado).
        #
        # Ordem de cálculo por dependência (sem circularidade):
        #   B2C: 36m (independente) → 24m (renewal = 36m) → 12m (renewal = 24m)
        #   B2B: 48m (independente) → 36m (renewal = 48m) → 24m (renewal = 36m)
        #                           → 12m (renewal = 24m)
        #
        # Mapa de qual prazo é usado como renewal_price para cada combo:
        #   B2C 24m → usa preço B2C 36m
        #   B2C 12m → usa preço B2C 24m
        #   B2B 36m → usa preço B2B 48m
        #   B2B 24m → usa preço B2B 36m
        #   B2B 12m → usa preço B2B 24m
        # ------------------------------------------------------------------

        # Dicionário de preços calculados para este ativo: (client_type, term) → suggested_price
        price_cache: dict = {}

        # Resultados do ativo antes do enforcement: (client_type, term) → PricingResult
        asset_results: dict = {}

        # Ordem garante que cada dependência já está calculada quando necessária
        COMBOS_ORDERED = [
            (ClientType.B2C, 36),
            (ClientType.B2C, 24),
            (ClientType.B2C, 12),
            (ClientType.B2B, 48),
            (ClientType.B2B, 36),
            (ClientType.B2B, 24),
            (ClientType.B2B, 12),
        ]

        # Mapa: (client_type, term) → (client_type, renewal_term) que deve ser usado como renewal_price
        #
        # B2C (12m > 24m > 36m):
        #   36m = âncora (sem renovação)
        #   24m renova 12m → usa preço 36m (âncora, menor)
        #   12m renova 24m → usa preço 24m
        #
        # B2B (12m > 24m > 36m > 48m):
        #   48m = âncora (sem renovação, mais barato)
        #   36m renova 12m → usa preço 48m → P_36m ligeiramente acima de P_48m
        #   24m renova 24m → usa preço 36m → P_24m ligeiramente acima de P_36m
        #   12m renova 36m → usa preço 24m → P_12m ligeiramente acima de P_24m
        RENEWAL_OVERRIDE_MAP = {
            (ClientType.B2C, 24): (ClientType.B2C, 36),
            (ClientType.B2C, 12): (ClientType.B2C, 24),
            (ClientType.B2B, 36): (ClientType.B2B, 48),
            (ClientType.B2B, 24): (ClientType.B2B, 36),
            (ClientType.B2B, 12): (ClientType.B2B, 24),
        }

        for client_type, term in COMBOS_ORDERED:
            label = f"{asset.id}  {client_type.value}  {term}m"
            try:
                contract = ContractParams(client_type=client_type, term_months=term)

                # Determina renewal_price_override, se aplicável
                override_key = RENEWAL_OVERRIDE_MAP.get((client_type, term))
                renewal_override = price_cache.get(override_key) if override_key else None

                r = price_asset(
                    asset=asset,
                    contract=contract,
                    params=pricing_params,
                    renewal_price_override=renewal_override,
                )
                asset_results[(client_type, term)] = r
                price_cache[(client_type, term)] = r.suggested_price
            except Exception as e:
                calc_errors.append((label, str(e)))
                _warn(f"Erro em {label}: {e}")

        # Enforça gaps de preço: mín R$10, máx R$30 entre prazos adjacentes
        asset_results = _enforce_price_ladder(asset_results)

        # Rebuild cashflows para resultados cujo preço foi ajustado pelo enforcement
        def _rebuild(r, new_price):
            """Reconstrói cashflows no preço enforçado."""
            a = r.asset; c = r.contract; p = r.params
            rp = r.renewal_price or 0.0
            curves = get_depreciation_curves()
            dp = get_dep_params(a.category, curves)
            ds = build_dep_schedule(market_price=a.market_price, eco_total=c.eco_total,
                floor_pct=dp["floor_pct"], dep_method=dp.get("dep_method","linear"),
                annual_rate=dp.get("annual_rate"))
            sv = get_sale_book_value(market_price=a.market_price, eco_total=c.eco_total, dep_params=dp)
            cf_u, cf_l, md = build_cashflows(asset=a, contract=c, params=p, P=new_price,
                renewal_price=rp, dep_schedule=ds, curves=curves, sale_book_value=sv)
            vu, vl = build_irr_vectors(md)
            return dataclasses.replace(r,
                suggested_price=new_price, cf_unlev=cf_u, cf_lev=cf_l,
                monthly_details=md, unlevered_irr=annual_irr(vu), levered_irr=annual_irr(vl),
                payback_months=calculate_payback(cf_u), margem_ebitda=calculate_ebitda_margin(md),
                annual_cashflows=_compute_annual_cashflows(md))

        for client_type, term in COMBOS_ORDERED:
            key = (client_type, term)
            if key not in asset_results:
                continue
            r = asset_results[key]
            if not r.monthly_details:
                continue
            price_in_cf = next(
                (m["preco_mensal"] for m in r.monthly_details if m.get("preco_mensal", 0) > 0),
                None,
            )
            if price_in_cf is not None and abs(r.suggested_price - price_in_cf) > 0.01:
                asset_results[key] = _rebuild(r, r.suggested_price)

        for client_type, term in COMBOS_ORDERED:
            key = (client_type, term)
            if key not in asset_results:
                continue
            r = asset_results[key]
            label = f"{asset.id}  {client_type.value}  {term}m"
            results.append(r)
            done += 1
            print(
                f"    {done:>3}/{total}  {label:<38}  {_fmt_brl(r.suggested_price)}",
                flush=True,
            )

    _ok(f"{len(results)} combinações calculadas" + (f"  |  {len(calc_errors)} erro(s)" if calc_errors else ""))

    # -- 4. Output --------------------------------------------------------------
    print(f"\n[4/4] Salvando resultados...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        export_summary_csv(results, OUTPUT_PATH)
        _ok(f"Resumo salvo   ->  {os.path.relpath(OUTPUT_PATH)}")
    except Exception as e:
        _warn(f"Não foi possível salvar o CSV de resumo: {e}")

    try:
        export_cashflow_anual_csv(results, OUTPUT_ANUAL_PATH)
        _ok(f"DFC anual salvo ->  {os.path.relpath(OUTPUT_ANUAL_PATH)}")
    except Exception as e:
        _warn(f"Não foi possível salvar o CSV de DFC anual: {e}")

    # -- Tabela resumida --------------------------------------------------------
    if results:
        print_results_table(results)

    # -- Sumário final ----------------------------------------------------------
    elapsed = (datetime.now() - started_at).total_seconds()

    _header("Sumário")
    print(f"  Ativos processados:   {len(assets)}")
    print(f"  Combinações geradas:  {len(results)}")
    print(f"  Erros de cálculo:     {len(calc_errors)}")
    print(f"  Arquivo de output:    {os.path.relpath(OUTPUT_PATH)}")
    print(f"  DFC anual:            {os.path.relpath(OUTPUT_ANUAL_PATH)}")
    print(f"  Tempo total:          {elapsed:.1f}s")
    print()
    print(f"  Parametros utilizados:")
    print(f"    funding_captacao_pct {p['funding_captacao_pct']:.4f}  ({p['funding_captacao_pct']*100:.1f}%)")
    print(f"    default_pct          {p['default_pct']:.4f}  ({p['default_pct']*100:.1f}% a.a.)")
    print(f"    dep_contabil_pct     {p['dep_contabil_pct']:.4f}  ({p['dep_contabil_pct']*100:.0f}% a.a.)")
    print(f"    pdd_pct              {p['pdd_pct']:.4f}  ({p['pdd_pct']*100:.1f}%)")
    print(f"    min_unlevered_irr    {p['min_unlevered_irr']:.4f}  ({p['min_unlevered_irr']*100:.0f}% a.a.)")
    print(f"    logistics_at_pct     {p['logistics_at_pct']:.4f}  ({p['logistics_at_pct']*100:.2f}%)")
    _sep()
    print()

    if calc_errors:
        print("  Erros de cálculo:")
        for label, msg in calc_errors:
            print(f"    {label}: {msg}")
        print()


if __name__ == "__main__":
    main()
