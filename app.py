"""
Interface Streamlit da Allu Pricing Engine.

Interface profissional para cálculo e visualização de preços de assinatura.
"""

import dataclasses
import os
import sys
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from plotly.subplots import make_subplots

# Garante que o pacote pricing_engine está no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pricing_engine.config import get_depreciation_curves
from pricing_engine.engine import price_asset
from pricing_engine.engine.cashflow import build_cashflows
from pricing_engine.engine.depreciation import build_dep_schedule, get_dep_params, get_sale_book_value
from pricing_engine.engine.optimizer import annual_irr, build_irr_vectors, calculate_ebitda_margin, calculate_payback
from pricing_engine.engine.pricing import _compute_annual_cashflows
from pricing_engine.models import Asset, ClientType, ContractParams, PricingParams
from pricing_engine.outputs.exporter import export_summary_csv, result_to_dataframe

# ----------------------------------------------------------
# CASCADE DE RENOVAÇÃO + ENFORCEMENT DE ESCADA DE PREÇOS
# Mesma lógica de run_pricing.py aplicada na interface.
# ----------------------------------------------------------

_RENEWAL_OVERRIDE_MAP = {
    (ClientType.B2C, 24): (ClientType.B2C, 36),
    (ClientType.B2C, 12): (ClientType.B2C, 24),
    (ClientType.B2B, 36): (ClientType.B2B, 48),
    (ClientType.B2B, 24): (ClientType.B2B, 36),
    (ClientType.B2B, 12): (ClientType.B2B, 24),
}

_COMBOS_ORDERED = [
    (ClientType.B2C, 36),
    (ClientType.B2C, 24),
    (ClientType.B2C, 12),
    (ClientType.B2B, 48),
    (ClientType.B2B, 36),
    (ClientType.B2B, 24),
    (ClientType.B2B, 12),
]


def _rebuild_result_at_price(result, new_price: float):
    """
    Reconstrói monthly_details/cashflows de um PricingResult para um novo preço.
    Usado quando o enforcement da escada bumpa suggested_price acima do que foi
    calculado pelo otimizador — garantindo que a DRE reflita o preço correto.
    """
    asset         = result.asset
    contract      = result.contract
    params        = result.params
    renewal_price = result.renewal_price or 0.0

    curves     = get_depreciation_curves()
    dep_params = get_dep_params(asset.category, curves)
    dep_sched  = build_dep_schedule(
        market_price=asset.market_price,
        eco_total=contract.eco_total,
        floor_pct=dep_params["floor_pct"],
        dep_method=dep_params.get("dep_method", "linear"),
        annual_rate=dep_params.get("annual_rate"),
    )
    sale_value = get_sale_book_value(
        market_price=asset.market_price,
        eco_total=contract.eco_total,
        dep_params=dep_params,
    )

    cf_unlev, cf_lev, monthly_details = build_cashflows(
        asset=asset,
        contract=contract,
        params=params,
        P=new_price,
        renewal_price=renewal_price,
        dep_schedule=dep_sched,
        curves=curves,
        sale_book_value=sale_value,
    )

    vec_unlev, vec_lev = build_irr_vectors(monthly_details)

    return dataclasses.replace(
        result,
        suggested_price=new_price,
        cf_unlev=cf_unlev,
        cf_lev=cf_lev,
        monthly_details=monthly_details,
        unlevered_irr=annual_irr(vec_unlev),
        levered_irr=annual_irr(vec_lev),
        payback_months=calculate_payback(cf_unlev),
        payback_lev_months=calculate_payback(cf_lev),
        margem_ebitda=calculate_ebitda_margin(monthly_details),
        annual_cashflows=_compute_annual_cashflows(monthly_details),
    )


MIN_GAP = 10.0   # Diferença mínima entre prazos adjacentes (preços estritamente crescentes)
MAX_GAP = 30.0   # Diferença máxima entre prazos adjacentes


def _enforce_price_gaps(asset_results: dict) -> dict:
    """
    Garante que entre prazos adjacentes:
    - Gap mínimo de R$10 (preços estritamente crescentes: 12m > 24m > 36m)
    - Gap máximo de R$30

    Cadeia: 36m → 24m → 12m (B2C) ou 48m → 36m → 24m → 12m (B2B).
    O prazo mais longo é âncora.
    """
    from pricing_engine.engine.optimizer import round_to_90

    def enforce_chain(chain):
        for i in range(1, len(chain)):
            r_prev = asset_results.get(chain[i - 1])
            r_curr = asset_results.get(chain[i])
            if not r_prev or not r_curr:
                continue
            if r_prev.suggested_price is None or r_curr.suggested_price is None:
                continue

            piso = round_to_90(r_prev.suggested_price + MIN_GAP)  # mínimo: anterior + 10
            teto = round_to_90(r_prev.suggested_price + MAX_GAP)  # máximo: anterior + 30

            p = r_curr.suggested_price
            if p < piso:
                p = piso
            if p > teto:
                p = teto

            if p != r_curr.suggested_price:
                asset_results[chain[i]] = dataclasses.replace(r_curr, suggested_price=p)

    enforce_chain([(ClientType.B2C, 36), (ClientType.B2C, 24), (ClientType.B2C, 12)])
    enforce_chain([(ClientType.B2B, 48), (ClientType.B2B, 36), (ClientType.B2B, 24), (ClientType.B2B, 12)])
    return asset_results


def compute_all_prices(ativo, client_type, params):
    """
    Calcula preços para todos os prazos do client_type com:
    1. Cascade de renewal (prazo curto usa preço do prazo longo como renovação)
    2. Gap máximo de R$30 entre prazos adjacentes
    3. Rebuild dos cashflows quando o preço é capado

    Retorna dict {term_months: PricingResult}.
    """
    from pricing_engine.models.contract import VALID_TERMS
    combos = [(client_type, t) for t in sorted(VALID_TERMS[client_type], reverse=True)]
    price_cache = {}
    asset_results = {}

    from pricing_engine.engine.optimizer import round_to_90

    # 1ª passagem: calcula todos os preços com cascade de renewal
    for ct, term in combos:
        override_key   = _RENEWAL_OVERRIDE_MAP.get((ct, term))
        renewal_override = price_cache.get(override_key) if override_key else None
        contrato = ContractParams(client_type=ct, term_months=term, with_final_sale=True)
        r = price_asset(ativo, contrato, params, renewal_price_override=renewal_override)
        asset_results[(ct, term)] = r
        price_cache[(ct, term)] = r.suggested_price

    # Enforcement de gaps com propagação reversa:
    # Se o cap de MAX_GAP faz o prazo curto estourar payback,
    # sobe o prazo anterior para que o curto caiba dentro do gap.
    def _enforce_with_propagation(chain):
        """chain = [(ct,36), (ct,24), (ct,12)] em ordem crescente de prazo."""
        # Primeiro, aplica gaps normais
        for i in range(1, len(chain)):
            r_prev = asset_results.get(chain[i - 1])
            r_curr = asset_results.get(chain[i])
            if not r_prev or not r_curr:
                continue
            if r_prev.suggested_price is None or r_curr.suggested_price is None:
                continue

            piso = round_to_90(r_prev.suggested_price + MIN_GAP)
            teto = round_to_90(r_prev.suggested_price + MAX_GAP)
            p = r_curr.suggested_price
            if p < piso:
                p = piso
            if p > teto:
                # Preço foi capado — verificar se payback estoura
                # Se sim, precisamos subir o prazo anterior
                p_needed = r_curr.suggested_price  # preço que o solver achou necessário
                if p_needed > teto:
                    # Subir o anterior para que teto >= p_needed
                    # teto_novo = round_to_90(p_prev_novo + MAX_GAP) >= p_needed
                    # p_prev_novo >= p_needed - MAX_GAP
                    p_prev_min = round_to_90(p_needed - MAX_GAP)
                    if p_prev_min > r_prev.suggested_price:
                        asset_results[chain[i - 1]] = dataclasses.replace(
                            r_prev, suggested_price=p_prev_min)
                        # Recalcula teto com o novo prev
                        teto = round_to_90(p_prev_min + MAX_GAP)
                        piso = round_to_90(p_prev_min + MIN_GAP)
                        # Propaga para trás recursivamente
                        if i - 1 > 0:
                            _enforce_with_propagation(chain[:i])
                p = min(p, teto)
                p = max(p, piso)

            if p != r_curr.suggested_price:
                asset_results[chain[i]] = dataclasses.replace(r_curr, suggested_price=p)

    b2c_chain = [(ct, t) for ct, t in combos if ct == ClientType.B2C]
    b2b_chain = [(ct, t) for ct, t in combos if ct == ClientType.B2B]
    if b2c_chain:
        _enforce_with_propagation(b2c_chain)
    if b2b_chain:
        _enforce_with_propagation(b2b_chain)

    # 2ª passagem: recomputa contratos cujo renewal_price ficou desatualizado
    for ct, term in combos:
        key          = (ct, term)
        override_key = _RENEWAL_OVERRIDE_MAP.get(key)
        if not override_key or override_key not in asset_results:
            continue
        new_renewal = asset_results[override_key].suggested_price
        r = asset_results[key]
        if r.renewal_price == new_renewal:
            continue
        contrato = ContractParams(client_type=ct, term_months=term, with_final_sale=True)
        r_new = price_asset(ativo, contrato, params, renewal_price_override=new_renewal)
        asset_results[key] = r_new

    # Re-enforcement
    if b2c_chain:
        _enforce_with_propagation(b2c_chain)
    if b2b_chain:
        _enforce_with_propagation(b2b_chain)

    # Passagem final: rebuild monthly_details quando preço foi ajustado
    for ct, term in combos:
        key = (ct, term)
        r   = asset_results[key]
        if not r.monthly_details:
            continue
        price_in_cf = next(
            (m["preco_mensal"] for m in r.monthly_details if m.get("preco_mensal", 0) > 0),
            None,
        )
        if price_in_cf is not None and abs(r.suggested_price - price_in_cf) > 0.01:
            asset_results[key] = _rebuild_result_at_price(r, r.suggested_price)

    return {term: asset_results[(ct, term)] for ct, term in combos}


# ----------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ----------------------------------------------------------
st.set_page_config(
    page_title="Allu Pricing Engine",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------
# CSS CUSTOMIZADO
# ----------------------------------------------------------
st.markdown(
    """
    <style>
    /* Variáveis de cor */
    :root {
        --allu-azul: #304D3C;
        --allu-azul-claro: #3D6B4F;
        --allu-verde: #43A047;
        --allu-vermelho: #E74C3C;
        --allu-laranja: #FF8F00;
        --allu-cinza: #F8F9FA;
        --allu-borda: #DEE2E6;
    }

    /* Fundo da aplicação */
    .main .block-container {
        background-color: #FFFFFF;
        padding-top: 1.5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F2F2F2;
    }

    /* Cabeçalho customizado */
    .allu-header {
        background: linear-gradient(135deg, #304D3C 0%, #3D6B4F 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .allu-header h1 {
        color: white !important;
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .allu-header p {
        color: #EFF6EA;
        margin: 0.2rem 0 0 0;
        font-size: 0.85rem;
    }

    /* Cards de métricas customizados */
    .metric-card {
        background: white;
        border: 1px solid var(--allu-borda);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
        height: 100%;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #6C757D;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #304D3C;
        line-height: 1.2;
    }
    .metric-card .value.destaque {
        color: #43A047;
        font-size: 1.9rem;
    }
    .metric-card .value.alerta {
        color: #E74C3C;
    }
    .metric-card .sub {
        font-size: 0.75rem;
        color: #ADB5BD;
        margin-top: 0.3rem;
    }

    /* Tabela de breakdown */
    .breakdown-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .breakdown-table th {
        background-color: #304D3C;
        color: white;
        padding: 0.5rem 0.8rem;
        text-align: left;
        font-weight: 600;
    }
    .breakdown-table td {
        padding: 0.45rem 0.8rem;
        border-bottom: 1px solid #F0F0F0;
    }
    .breakdown-table tr:nth-child(even) td {
        background-color: #F8F9FA;
    }
    .breakdown-table tr:hover td {
        background-color: #EFF6EA;
    }
    .breakdown-table .valor-positivo {
        color: #43A047;
        font-weight: 600;
    }
    .breakdown-table .valor-negativo {
        color: #E74C3C;
    }

    /* Badge de status */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-azul { background: #EFF6EA; color: #304D3C; }
    .badge-verde { background: #D5F5E3; color: #1A7A42; }
    .badge-laranja { background: #FEF9E7; color: #B7770D; }

    /* Separador com título */
    .section-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #304D3C;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        border-left: 3px solid #304D3C;
        padding-left: 0.6rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Botão principal */
    [data-testid="stButton"] button[kind="primary"] {
        background-color: #304D3C;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }
    [data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #3D6B4F;
    }

    /* Info box */
    .info-box {
        background: #EFF6EA;
        border-left: 4px solid #304D3C;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #304D3C;
        margin: 0.8rem 0;
    }

    /* Aviso */
    .aviso-box {
        background: #FFF8E1;
        border-left: 4px solid #F39C12;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #7D6608;
        margin: 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# CONSTANTES
# ----------------------------------------------------------
ASSETS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pricing_engine", "data", "assets.csv"
)

COR_AZUL     = "#304D3C"     # Verde escuro Allu (principal)
COR_VERDE    = "#43A047"     # Verde médio Allu (destaque positivo)
COR_VERMELHO = "#E74C3C"     # Vermelho (alertas)
COR_LARANJA  = "#FF8F00"     # Âmbar (secundário)
COR_CINZA    = "#ADB5BD"     # Cinza neutro
COR_GAP      = "#FFF3CD"     # Amarelo claro


# ----------------------------------------------------------
# FUNÇÕES AUXILIARES
# ----------------------------------------------------------

@st.cache_data
def carregar_ativos_csv(filepath: str) -> pd.DataFrame:
    """Carrega o CSV de ativos com cache."""
    return pd.read_csv(filepath, sep=";")


def df_para_assets(df: pd.DataFrame) -> dict:
    """Converte DataFrame em dict de Asset."""
    ativos = {}
    for _, row in df.iterrows():
        ativo = Asset(
            id=str(row["id"]),
            name=str(row["name"]),
            category=str(row["category"]),
            purchase_price=float(row["purchase_price"]),
            market_price=float(row["market_price"]),
            maintenance_annual_pct=float(row["maintenance_annual_pct"]),
            condicao=str(row["condicao"]).strip() if "condicao" in row.index else "novo",
        )
        ativos[ativo.id] = ativo
    return ativos


def fmt_brl(val, decimais=2) -> str:
    """Formata valor como moeda brasileira."""
    if val is None:
        return "—"
    return f"R$ {val:,.{decimais}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(val) -> str:
    """Formata valor como percentual."""
    if val is None:
        return "—"
    return f"{val:.2%}".replace(".", ",")


def _brl(v) -> str:
    """Formata valor inline para tabelas HTML — padrão brasileiro (vírgula decimal)."""
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def card_metrica(label: str, value: str, sub: str = "", destaque: bool = False, alerta: bool = False) -> str:
    """Gera HTML de um card de métrica."""
    cls_value = "value destaque" if destaque else ("value alerta" if alerta else "value")
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="{cls_value}">{value}</div>
        {sub_html}
    </div>
    """


# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

def render_sidebar():
    """Renderiza a sidebar com os inputs de configuração."""
    with st.sidebar:
        # Logo / Título
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
                <div style="font-size:2rem; margin-bottom:0.3rem;">📱</div>
                <div style="font-size:1.2rem; font-weight:800; color:#304D3C;">Allu Pricing Engine</div>
                <div style="font-size:0.75rem; color:#6C757D; margin-top:0.2rem;">
                    Device as a Service — Precificação
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # --------------------------------------------------
        # Seleção do ativo
        # --------------------------------------------------
        st.markdown('<div class="section-title">Ativo</div>', unsafe_allow_html=True)

        df_ativos = carregar_ativos_csv(ASSETS_CSV)
        ativos_dict = df_para_assets(df_ativos)

        opcoes_ativos = {row["name"]: row["id"] for _, row in df_ativos.iterrows()}
        ativo_nome_sel = st.selectbox(
            "Selecione o device",
            options=list(opcoes_ativos.keys()),
            label_visibility="collapsed",
        )
        ativo_id_sel = opcoes_ativos[ativo_nome_sel]
        ativo_sel = ativos_dict[ativo_id_sel]

        # Exibe info do ativo selecionado
        st.markdown(
            f"""
            <div class="info-box">
                <b>{ativo_sel.name}</b><br>
                Compra: {fmt_brl(ativo_sel.purchase_price)} &nbsp;|&nbsp;
                Mercado: {fmt_brl(ativo_sel.market_price)}<br>
                Categoria: <span style="text-transform:capitalize">{ativo_sel.category}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --------------------------------------------------
        # Tipo de cliente
        # --------------------------------------------------
        st.markdown('<div class="section-title">Tipo de Cliente</div>', unsafe_allow_html=True)

        tipo_cliente = st.radio(
            "Tipo de cliente",
            options=["B2C", "B2B"],
            horizontal=True,
            label_visibility="collapsed",
        )
        client_type = ClientType.B2C if tipo_cliente == "B2C" else ClientType.B2B

        # --------------------------------------------------
        # Prazo do contrato
        # --------------------------------------------------
        st.markdown('<div class="section-title">Prazo do Contrato</div>', unsafe_allow_html=True)

        prazos_disponiveis = [12, 24, 36] if client_type == ClientType.B2C else [12, 24, 36, 48]
        prazo_labels = [f"{p} meses" for p in prazos_disponiveis]

        prazo_sel_label = st.radio(
            "Prazo",
            options=prazo_labels,
            horizontal=True,
            label_visibility="collapsed",
        )
        prazo_sel = int(prazo_sel_label.split(" ")[0])

        # Exibe informação de renovação
        from pricing_engine.models.contract import RENEWAL_MAP
        renewal_m = RENEWAL_MAP.get((client_type, prazo_sel), 0)
        if renewal_m > 0:
            eco_total = 36 if client_type == ClientType.B2C else 48
            st.markdown(
                f"""
                <div class="info-box">
                    Ciclo econômico: {eco_total} meses<br>
                    Renovação implícita: +{renewal_m} meses<br>
                    Calendário total: {prazo_sel + renewal_m} meses
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            eco_total = prazo_sel
            st.markdown(
                f"""
                <div class="info-box">
                    Ciclo econômico: {eco_total} meses<br>
                    Sem renovação automática
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --------------------------------------------------
        # Opções adicionais
        # --------------------------------------------------
        with_sale = st.checkbox("Incluir venda do device ao final", value=True)

        # --------------------------------------------------
        # Parâmetros Avançados
        # --------------------------------------------------
        st.markdown('<div class="section-title">Parâmetros</div>', unsafe_allow_html=True)

        with st.expander("Parâmetros Avançados", expanded=False):
            st.caption("Fiscal e Tributário")
            icms_pct = st.number_input("ICMS (%)", value=14.0, min_value=0.0, max_value=100.0, step=0.5, format="%.1f") / 100
            captacao_pct = st.number_input("Custo captação (%)", value=2.0, min_value=0.0, max_value=100.0, step=0.5, format="%.1f") / 100
            juros_aa = st.number_input("Juros a.a. (%)", value=23.72, min_value=0.0, max_value=200.0, step=0.5, format="%.2f") / 100
            pis_cofins = st.number_input("PIS+COFINS (%)", value=9.25, min_value=0.0, max_value=100.0, step=0.25, format="%.2f") / 100
            iss = st.number_input("ISS (%)", value=2.5, min_value=0.0, max_value=100.0, step=0.5, format="%.1f") / 100
            mdr = st.number_input("MDR (%)", value=2.0, min_value=0.0, max_value=100.0, step=0.5, format="%.1f") / 100

            st.caption("Custos Operacionais")
            logistics_ass = st.number_input("Logística assinatura (R$)", value=85.0, min_value=0.0, step=5.0, format="%.2f")
            logistics_at = st.number_input("Logística AT (%)", value=7.0, min_value=0.0, max_value=100.0, step=0.1, format="%.2f") / 100
            benefits = st.number_input("Benefícios cliente (R$)", value=67.0, min_value=0.0, step=5.0, format="%.2f")
            risk_query = st.number_input("Consulta risco (R$)", value=32.0, min_value=0.0, step=1.0, format="%.2f")
            prep_venda = st.number_input("Prep. venda (% do ativo)", value=2.0, min_value=0.0, max_value=100.0, step=0.5, format="%.1f") / 100
            logistics_venda = st.number_input("Logística venda (R$)", value=85.0, min_value=0.0, step=5.0, format="%.2f")
            cac_venda_pct = st.number_input("CAC venda (%)", value=10.0, min_value=0.0, max_value=100.0, step=1.0, format="%.1f") / 100

            st.caption("Risco")
            pdd = st.number_input("PDD (%)", value=8.0, min_value=0.0, max_value=100.0, step=0.5, format="%.1f") / 100
            default_p = st.number_input("Default mensal (%)", value=3.0, min_value=0.0, max_value=100.0, step=0.1, format="%.2f") / 100

            st.caption("Otimização")
            min_irr = st.number_input("TIR mínima desalav. (%)", value=30.0, min_value=0.0, max_value=100.0, step=1.0, format="%.1f") / 100

            st.caption("Dívida")
            prazo_div = st.number_input("Prazo da dívida (meses)", value=30, min_value=1, max_value=60, step=1)

        # Monta PricingParams com os valores do formulário
        pricing_params = PricingParams(
            icms_pct=icms_pct,
            funding_captacao_pct=captacao_pct,
            debt_annual_rate=juros_aa,
            pis_cofins_pct=pis_cofins,
            pis_cofins_credit_pct=pis_cofins,
            iss_pct=iss,
            mdr_pct=mdr,
            cac_venda_pct=cac_venda_pct,
            prep_venda=prep_venda,
            logistics_venda=logistics_venda,
            logistics_assinatura=logistics_ass,
            logistics_at_pct=logistics_at,
            risk_query_value=risk_query,
            customer_benefits=benefits,
            pdd_pct=pdd,
            default_pct=default_p,
            min_unlevered_irr=min_irr,
            prazo_divida_meses=prazo_div,
        )

        st.divider()

        # Botão calcular
        calcular = st.button("Calcular Preço", type="primary", use_container_width=True)

    return {
        "ativo": ativo_sel,
        "client_type": client_type,
        "prazo": prazo_sel,
        "with_sale": with_sale,
        "params": pricing_params,
        "calcular": calcular,
    }


# ----------------------------------------------------------
# TAB 1: RESULTADO
# ----------------------------------------------------------

def render_tab_resultado(result):
    """Renderiza a aba de resultado principal."""
    from pricing_engine.models.contract import VALID_TERMS

    # Calcula todos os prazos para cards e tabela
    all_results = compute_all_prices(result.asset, result.contract.client_type, result.params)
    is_b2c = result.contract.client_type == ClientType.B2C
    prazos_todos = [12, 24, 36, 48]

    # Linha 1: Cards de mensalidade por prazo
    st.markdown('<div class="section-title">Mensalidades por Prazo</div>', unsafe_allow_html=True)
    cols = st.columns(4)

    for i, prazo in enumerate(prazos_todos):
        with cols[i]:
            res_prazo = all_results.get(prazo)
            if is_b2c and prazo == 48:
                valor_str = "—"
                sub_str = "N/A para B2C"
                dest = False
            elif res_prazo and res_prazo.suggested_price:
                valor_str = fmt_brl(res_prazo.suggested_price)
                sub_str = f"Raw: {fmt_brl(res_prazo.suggested_price_raw, 2)}"
                dest = True
            else:
                valor_str = "—"
                sub_str = ""
                dest = False
            st.markdown(
                card_metrica(f"Mensalidade {prazo}m", valor_str, sub=sub_str, destaque=dest),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Linha 2: Indicadores-resumo
    cols2 = st.columns(5)

    with cols2[0]:
        tir_ok = result.unlevered_irr and result.unlevered_irr >= result.params.min_unlevered_irr - 0.001
        st.markdown(
            card_metrica(
                "TIR Desalavancada",
                fmt_pct(result.unlevered_irr),
                sub=f"Alvo: {fmt_pct(result.params.min_unlevered_irr)}",
                destaque=tir_ok,
                alerta=not tir_ok,
            ),
            unsafe_allow_html=True,
        )

    with cols2[1]:
        st.markdown(
            card_metrica(
                "TIR Alavancada",
                fmt_pct(result.levered_irr),
                sub="Equity return",
            ),
            unsafe_allow_html=True,
        )

    with cols2[2]:
        margem = result.margem_ebitda
        margem_ok = margem is not None and margem >= 0.13
        st.markdown(
            card_metrica(
                "Margem EBITDA",
                f"{margem:.1%}" if margem is not None else "—",
                sub="Alvo: ≥ 13%",
                destaque=margem_ok,
                alerta=not margem_ok,
            ),
            unsafe_allow_html=True,
        )

    with cols2[3]:
        pb_ok = result.payback_months is not None and result.payback_months <= 24
        st.markdown(
            card_metrica(
                "Payback Desalav.",
                f"{result.payback_months}m" if result.payback_months else "—",
                sub="Alvo: ≤ 24 meses",
                destaque=pb_ok,
                alerta=not pb_ok,
            ),
            unsafe_allow_html=True,
        )

    with cols2[4]:
        pb_lev_ok = result.payback_lev_months is not None and result.payback_lev_months <= 30
        st.markdown(
            card_metrica(
                "Payback Alav.",
                f"{result.payback_lev_months}m" if result.payback_lev_months else "—",
                sub="Alvo: ≤ 30 meses",
                destaque=pb_lev_ok,
                alerta=not pb_lev_ok,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------
    # Cards: Valores do Ativo
    # --------------------------------------------------
    st.markdown('<div class="section-title">Valores do Ativo</div>', unsafe_allow_html=True)

    sale_val = 0.0
    if result.monthly_details:
        last_m = result.monthly_details[-1]
        sale_val = last_m.get("fci_sale_val", 0.0)

    cols_val = st.columns(3)
    with cols_val[0]:
        st.markdown(
            card_metrica(
                "Valor de Mercado",
                fmt_brl(result.asset.market_price),
                sub="Preço de referência do device",
            ),
            unsafe_allow_html=True,
        )
    with cols_val[1]:
        st.markdown(
            card_metrica(
                "Valor de Compra",
                fmt_brl(result.asset.purchase_price),
                sub="Custo de aquisição pela Allu",
            ),
            unsafe_allow_html=True,
        )
    with cols_val[2]:
        st.markdown(
            card_metrica(
                "Valor de Venda (Final)",
                fmt_brl(sale_val) if sale_val > 0 else "—",
                sub=f"Ao final do ciclo ({result.contract.eco_total}m)",
                destaque=sale_val > 0,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------
    # Tabela: Comparação de Cenários
    # --------------------------------------------------
    st.markdown(
        f'<div class="section-title">Comparação de Cenários — {result.asset.name} ({result.contract.client_type.value})</div>',
        unsafe_allow_html=True,
    )

    resultados = [all_results[t] for t in sorted(all_results) if all_results[t] is not None]
    if resultados:
        linhas = []
        for res in resultados:
            margem_res = res.margem_ebitda
            linhas.append({
                "Prazo": f"{res.contract.term_months}m",
                "Ciclo Econ.": f"{res.contract.eco_total}m",
                "Renovação": f"+{res.contract.renewal_months}m" if res.contract.has_renewal else "—",
                "Preço Sugerido": fmt_brl(res.suggested_price),
                "Breakeven": fmt_brl(res.breakeven_price),
                "Preço Renovação": fmt_brl(res.renewal_price) if res.renewal_price else "—",
                "TIR Desalav.": fmt_pct(res.unlevered_irr),
                "TIR Alav.": fmt_pct(res.levered_irr),
                "Margem EBITDA": f"{margem_res:.1%}" if margem_res is not None else "—",
                "Payback": f"{res.payback_months}m" if res.payback_months else "—",
            })

        df_comp = pd.DataFrame(linhas)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # Gráficos: Preços por Prazo + TIR por Prazo
    # --------------------------------------------------
    if resultados:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        prazos_labels = [f"{r.contract.term_months}m" for r in resultados]

        with col1:
            st.markdown('<div class="section-title">Preços por Prazo</div>', unsafe_allow_html=True)
            precos_sug = [r.suggested_price or 0 for r in resultados]
            precos_be  = [r.breakeven_price or 0 for r in resultados]
            fig_preco = go.Figure()
            fig_preco.add_trace(go.Bar(
                x=prazos_labels, y=precos_sug, name="Sugerido", marker_color=COR_AZUL,
            ))
            fig_preco.add_trace(go.Bar(
                x=prazos_labels, y=precos_be, name="Breakeven", marker_color=COR_CINZA,
            ))
            fig_preco.update_layout(
                separators=",.",
                barmode="group", height=300,
                plot_bgcolor="white", paper_bgcolor="white",
                showlegend=True, legend=dict(orientation="h", y=1.1),
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(tickprefix="R$ ", tickformat=",.0f", gridcolor="#F0F0F0"),
            )
            st.plotly_chart(fig_preco, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">TIR por Prazo</div>', unsafe_allow_html=True)
            tirs_desalav = [(r.unlevered_irr or 0) * 100 for r in resultados]
            tirs_alav    = [(r.levered_irr or 0) * 100 for r in resultados]
            fig_tir = go.Figure()
            fig_tir.add_trace(go.Scatter(
                x=prazos_labels, y=tirs_desalav, name="TIR Desalav.",
                mode="lines+markers", line=dict(color=COR_AZUL, width=2.5), marker=dict(size=8),
            ))
            fig_tir.add_trace(go.Scatter(
                x=prazos_labels, y=tirs_alav, name="TIR Alav.",
                mode="lines+markers", line=dict(color=COR_LARANJA, width=2.5, dash="dot"), marker=dict(size=8),
            ))
            fig_tir.add_hline(
                y=result.params.min_unlevered_irr * 100,
                line_dash="dash", line_color=COR_VERMELHO,
                annotation_text=f"TIR mín. {result.params.min_unlevered_irr:.0%}",
            )
            fig_tir.update_layout(
                separators=",.",
                height=300, plot_bgcolor="white", paper_bgcolor="white",
                showlegend=True, legend=dict(orientation="h", y=1.1),
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(title="%", gridcolor="#F0F0F0"),
            )
            st.plotly_chart(fig_tir, use_container_width=True)


# ----------------------------------------------------------
# TAB PREMISSAS
# ----------------------------------------------------------

def render_tab_premissas(result):
    """Renderiza a aba de premissas — tabela com todas as premissas do modelo."""
    p = result.params
    asset = result.asset

    st.markdown('<div class="section-title">Premissas do Modelo</div>', unsafe_allow_html=True)

    premissas = [
        # Fiscal e Tributário
        ("FISCAL E TRIBUTÁRIO", "", "", "header"),
        ("ICMS", f"{p.icms_pct:.0%}", "Incide sobre o valor de compra do ativo"),
        ("PIS/COFINS", f"{p.pis_cofins_pct:.2%}", "Incide sobre o valor da assinatura mensal"),
        ("Crédito PIS/COFINS", f"{p.pis_cofins_credit_pct:.2%}", "Crédito sobre o valor de compra do ativo, abatido mensalmente"),
        ("ISS", f"{p.iss_pct:.1%}", "Incide sobre o valor da assinatura mensal"),
        ("MDR", f"{p.mdr_pct:.0%}", "Taxa da adquirência sobre receita mensal"),
        ("IR/CSLL", f"{p.ir_csll_pct:.1%}", "Incide sobre EBT na DRE (simétrico: despesa se EBT>0, crédito se EBT<0)"),
        # Financiamento
        ("FINANCIAMENTO", "", "", "header"),
        ("Taxa de captação", f"{p.funding_captacao_pct:.0%}", "Custo upfront sobre o funding total (compra + ICMS)"),
        ("Juros da dívida", f"{p.debt_annual_rate:.2%} a.a.", f"Taxa anual; mensal = {p.debt_annual_rate/12:.4%}"),
        ("Prazo da dívida", f"{p.prazo_divida_meses} meses", f"Juros até mês {p.prazo_divida_meses}, amortização no mês {p.prazo_divida_meses + 1}"),
        # Custos Operacionais
        ("CUSTOS OPERACIONAIS", "", "", "header"),
        ("Manutenção", f"{asset.maintenance_annual_pct:.2%} a.a.", f"Sobre purchase_price; mensal = {fmt_brl(asset.maintenance_monthly)}"),
        ("Customer benefits", fmt_brl(p.customer_benefits), "Custo pontual no início de cada contrato/renovação"),
        ("Logística assinatura", fmt_brl(p.logistics_assinatura), "Custo por evento de entrega/devolução"),
        ("Logística AT", f"{p.logistics_at_pct:.0%} a.a.", "Provisão mensal para troca por estrago durante contrato"),
        ("Logística de venda", fmt_brl(p.logistics_venda), "Custo de envio do device ao comprador final"),
        ("Prep. de venda", f"{p.prep_venda:.0%}", "Percentual sobre valor de venda para preparação/refurb"),
        ("CAC de venda", f"{p.cac_venda_pct:.0%}", "Comissão sobre valor de venda do device usado"),
        ("Consulta de risco", fmt_brl(p.risk_query_value), "Custo pontual de consulta de crédito (B2C only)"),
        # Risco
        ("RISCO", "", "", "header"),
        ("PDD", f"{p.pdd_pct:.0%}", "Provisão para devedores duvidosos — incide sobre mensalidade"),
        ("Default", f"{p.default_pct:.0%} a.a.", "Perda esperada sobre valor do ativo por mês"),
        # Depreciação
        ("DEPRECIAÇÃO", "", "", "header"),
        ("Dep. contábil", f"{p.dep_contabil_pct:.0%} a.a.", "Linear sobre purchase_price — base para NBV na DRE"),
        # Otimização
        ("OTIMIZAÇÃO", "", "", "header"),
        ("TIR mínima desalav.", f"{p.min_unlevered_irr:.0%}", "Hurdle rate — TIR desalavancada mínima alvo"),
        ("Payback máx. desalav.", "24 meses", "Restrição de payback desalavancado"),
        ("Payback máx. alav.", "30 meses", "Restrição de payback alavancado"),
        ("Margem EBITDA mín.", "30%", "Restrição de margem EBITDA mínima"),
        ("Gap mín. entre prazos", fmt_brl(10), "Diferença mínima de preço entre prazos adjacentes"),
        ("Gap máx. entre prazos", fmt_brl(30), "Diferença máxima de preço entre prazos adjacentes"),
    ]

    # Estilos
    th_s = "background:#304D3C;color:white;padding:0.45rem 0.7rem;font-size:0.82rem;white-space:nowrap;"
    hdr_s = "background:#EFF6EA;color:#304D3C;font-weight:700;padding:0.4rem 0.7rem;font-size:0.82rem;"
    td_s = "padding:0.38rem 0.7rem;font-size:0.82rem;border-bottom:1px solid #F0F0F0;"

    header = (f'<th style="{th_s}">Premissa</th>'
              f'<th style="{th_s};text-align:center;min-width:120px;">Valor</th>'
              f'<th style="{th_s}">Observação</th>')

    body = ""
    for item in premissas:
        if len(item) == 4 and item[3] == "header":
            body += f'<tr><td style="{hdr_s}" colspan="3">{item[0]}</td></tr>'
        else:
            nome, valor, obs = item[0], item[1], item[2]
            body += (f'<tr>'
                     f'<td style="{td_s}">{nome}</td>'
                     f'<td style="{td_s};text-align:center;font-weight:600;">{valor}</td>'
                     f'<td style="{td_s};color:#6C757D;">{obs}</td>'
                     f'</tr>')

    html = f"""
    <table style="width:100%;border-collapse:collapse;">
        <thead><tr>{header}</tr></thead>
        <tbody>{body}</tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------
# TAB 2: FLUXO DE CAIXA
# ----------------------------------------------------------

def render_tab_cashflow(result):
    """Renderiza a aba de fluxo de caixa — tabela com meses nas colunas e rubricas nas linhas."""
    if not result.monthly_details or result.cf_unlev is None:
        st.warning("Nenhum dado de fluxo de caixa disponível.")
        return

    details = result.monthly_details

    st.markdown('<div class="section-title">Fluxo de Caixa Mensal</div>', unsafe_allow_html=True)

    # Linhas de investimento do M0 (aparecem só no mês 0)
    funding = result.asset.purchase_price * (1 + result.params.icms_pct)
    icms_val = result.asset.purchase_price * result.params.icms_pct

    # Injeta campos de investimento no dict do mês 0
    m0 = details[0]
    m0["_investimento_total"] = -funding
    m0["_compra_aparelho"] = -result.asset.purchase_price
    m0["_icms_compra"] = -icms_val

    # Definição das rubricas (linhas)
    LINHAS = [
        ("Tipo do mês",              "tipo",              "text"),
        # --- Investimento (só M0) ---
        ("INVESTIMENTO",             None,                "header"),
        ("( - ) Compra do aparelho", "_compra_aparelho",  "m0_only"),
        ("( - ) ICMS da compra",     "_icms_compra",      "m0_only"),
        ("= Investimento total",     "_investimento_total","m0_total"),
        # --- Operacional ---
        ("OPERACIONAL",              None,                "header"),
        ("Preço mensal",             "preco_mensal",      "+"),
        ("( - ) PIS/COFINS líq.",    "pis_cofins_liq",    "-"),
        ("( - ) ISS",                "iss",               "-"),
        ("( - ) MDR",                "mdr",               "-"),
        ("= Receita Líquida",        "receita_liq",       "total"),
        ("( - ) Manutenção",         "manutencao",        "-"),
        ("( - ) Customer benefits",  "customer_benefits",  "-"),
        ("( - ) PDD",                "pdd",               "-"),
        ("( - ) Default",            "default_m",         "-"),
        ("( - ) CAC",                "cac",               "-"),
        ("( - ) Risk query",         "risk_query",        "-"),
        ("( - ) Logística entrega",  "logistics_ass",     "-"),
        ("( - ) Logística AT",       "logistics_prov_troca", "-"),
        ("( - ) Logística devolução","logistics_devolucao","-"),
        ("( + ) Venda líquida",      "sale_net",          "+"),
        ("( - ) Juros dívida",       "juros_divida",      "-"),
        ("( - ) Amort. principal",   "principal_divida",   "-"),
        ("= Fluxo Desalavancado",    "cf_unlev",          "grand_total"),
        ("= Fluxo Alavancado",       "cf_lev",            "grand_total"),
        ("= Desalav. Acumulado",     "cf_unlev_acum",     "total"),
        ("= Alavancado Acumulado",   "cf_lev_acum",       "total"),
    ]

    # Monta os meses como colunas
    mes_labels = [f"M{d['cal_m']}" for d in details]

    # Estilos
    th_s = "background:#304D3C;color:white;padding:0.35rem 0.5rem;font-size:0.78rem;white-space:nowrap;text-align:center;min-width:70px;"
    td_s = "padding:0.3rem 0.5rem;font-size:0.78rem;text-align:right;border-bottom:1px solid #F0F0F0;white-space:nowrap;"
    td_label_s = "padding:0.3rem 0.5rem;font-size:0.78rem;text-align:left;border-bottom:1px solid #F0F0F0;white-space:nowrap;position:sticky;left:0;background:white;z-index:1;"
    tot_s = "padding:0.35rem 0.5rem;font-size:0.78rem;text-align:right;font-weight:700;border-top:2px solid #304D3C;border-bottom:1px solid #304D3C;white-space:nowrap;"
    tot_label_s = tot_s.replace("text-align:right", "text-align:left") + "position:sticky;left:0;background:#EFF6EA;z-index:1;"
    grand_s = "padding:0.35rem 0.5rem;font-size:0.78rem;text-align:right;font-weight:700;background:#304D3C;color:white;white-space:nowrap;"
    grand_label_s = grand_s.replace("text-align:right", "text-align:left") + "position:sticky;left:0;z-index:1;"

    # Header
    header = f'<th style="{th_s};text-align:left;min-width:180px;position:sticky;left:0;z-index:2;">Rubrica</th>'
    for lbl in mes_labels:
        header += f'<th style="{th_s}">{lbl}</th>'

    # Estilos adicionais
    hdr_s = "background:#EFF6EA;color:#304D3C;font-weight:700;padding:0.4rem 0.5rem;font-size:0.78rem;"

    # Body
    body = ""
    ncols = len(details)
    for nome, key, tipo in LINHAS:
        if tipo == "header":
            body += f'<tr><td style="{hdr_s};position:sticky;left:0;z-index:1;" colspan="{ncols + 1}">{nome}</td></tr>'
            continue

        if tipo == "text":
            row_html = f'<td style="{td_label_s}">{nome}</td>'
            for d in details:
                row_html += f'<td style="{td_s};text-align:center;color:#6C757D;font-size:0.72rem;">{d.get(key,"")}</td>'

        elif tipo == "m0_only":
            # Só mostra valor no M0, demais vazios
            row_html = f'<td style="{td_label_s}">{nome}</td>'
            for d in details:
                v = d.get(key, 0) if d["cal_m"] == 0 else 0
                if v != 0:
                    row_html += f'<td style="{td_s}color:#E74C3C;">{_brl(v)}</td>'
                else:
                    row_html += f'<td style="{td_s}">—</td>'

        elif tipo == "m0_total":
            row_html = f'<td style="{tot_label_s}">{nome}</td>'
            for d in details:
                v = d.get(key, 0) if d["cal_m"] == 0 else 0
                if v != 0:
                    cor = "color:#43A047;" if v >= 0 else "color:#E74C3C;"
                    row_html += f'<td style="{tot_s}{cor}">{_brl(v)}</td>'
                else:
                    row_html += f'<td style="{tot_s}">—</td>'

        elif tipo in ("total", "grand_total"):
            ls = grand_label_s if tipo == "grand_total" else tot_label_s
            vs = grand_s if tipo == "grand_total" else tot_s
            row_html = f'<td style="{ls}">{nome}</td>'
            for d in details:
                v = d.get(key, 0)
                cor = "color:#43A047;" if v >= 0 else "color:#FF9999;" if tipo == "grand_total" else ("color:#43A047;" if v >= 0 else "color:#E74C3C;")
                row_html += f'<td style="{vs}{cor}">{_brl(v)}</td>'
        else:
            row_html = f'<td style="{td_label_s}">{nome}</td>'
            for d in details:
                v = d.get(key, 0)
                display_v = -v if tipo == "-" and v != 0 else v
                cor = "color:#E74C3C;" if tipo == "-" and v > 0 else ("color:#43A047;" if tipo == "+" and v > 0 else "")
                row_html += f'<td style="{td_s}{cor}">{_brl(display_v)}</td>'

        body += f"<tr>{row_html}</tr>"

    html = f"""
    <div style="overflow-x:auto;max-height:650px;">
        <table style="border-collapse:collapse;min-width:100%;">
            <thead><tr>{header}</tr></thead>
            <tbody>{body}</tbody>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # --------------------------------------------------
    # Gráfico anual: Fluxo de Caixa + Acumulado (estilo payback)
    # --------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Fluxo de Caixa Anual</div>', unsafe_allow_html=True)

    import math

    # Agrupa por ano
    cf_unlev_anual = {}
    cf_lev_anual = {}
    for d in details:
        cal = d["cal_m"]
        ano = 0 if cal == 0 else math.ceil(cal / 12)
        cf_unlev_anual[ano] = cf_unlev_anual.get(ano, 0) + d.get("cf_unlev", 0)
        cf_lev_anual[ano] = cf_lev_anual.get(ano, 0) + d.get("cf_lev", 0)

    anos = sorted(cf_unlev_anual.keys())
    labels = [f"Ano {a}" for a in anos]
    vals_u = [cf_unlev_anual[a] for a in anos]

    # Acumulado desalavancado
    acum_u = []
    s = 0
    for v in vals_u:
        s += v
        acum_u.append(s)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Fluxo Desalavancado por Ano", "Fluxo Acumulado Desalavancado"),
        row_heights=[0.50, 0.50],
    )

    # Painel 1: Barras desalavancado por ano
    cores = [COR_VERDE if v >= 0 else COR_VERMELHO for v in vals_u]
    fig.add_trace(
        go.Bar(
            x=labels, y=vals_u,
            name="Fluxo Desalavancado",
            marker_color=cores,
            opacity=0.85,
            hovertemplate="%{x}<br>Fluxo: R$ %{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="#DEE2E6", row=1, col=1)

    # Painel 2: Linha acumulada com área preenchida
    fig.add_trace(
        go.Scatter(
            x=labels, y=acum_u,
            name="Acumulado Desalavancado",
            fill="tozeroy",
            fillcolor="rgba(30, 58, 95, 0.08)",
            line=dict(color=COR_AZUL, width=2.5),
            mode="lines+markers",
            marker=dict(size=8, color=COR_AZUL),
            hovertemplate="%{x}<br>Acumulado: R$ %{y:,.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color="#ADB5BD", row=2, col=1)

    # Marca payback no gráfico acumulado
    if result.payback_months is not None:
        pb_ano = math.ceil(result.payback_months / 12)
        if pb_ano in anos:
            idx = anos.index(pb_ano)
            fig.add_vline(
                x=labels[idx],
                line_width=2, line_dash="dash", line_color=COR_VERDE,
                row=2, col=1,
            )
            fig.add_annotation(
                x=labels[idx], y=acum_u[idx],
                text=f"Payback: mês {result.payback_months}",
                showarrow=True, arrowhead=2, arrowcolor=COR_VERDE,
                font=dict(color=COR_VERDE, size=11, weight="bold"),
                bgcolor="white", bordercolor=COR_VERDE, borderwidth=1,
                row=2, col=1,
            )

    fig.update_layout(
        separators=",.",
        height=520,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="#F0F0F0", tickprefix="R$ ", tickformat=",.0f")
    fig.update_xaxes(gridcolor="#F0F0F0")

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------
# TAB 3: BREAKDOWN
# ----------------------------------------------------------

def render_tab_breakdown(result):
    """Renderiza a aba de análise de composição de custos."""
    if not result.monthly_details:
        st.warning("Nenhum dado de breakdown disponível.")
        return

    # Filtra apenas meses operacionais
    meses = [
        d for d in result.monthly_details
        if d["cal_m"] > 0 and d["tipo"] != "aquisição"
    ]

    if not meses:
        st.info("Nenhum mês operacional encontrado.")
        return

    col_pizza, col_barra = st.columns([1, 2])

    # --------------------------------------------------
    # Gráfico de pizza: composição média da mensalidade
    # --------------------------------------------------
    with col_pizza:
        st.markdown('<div class="section-title">Composição Média da Mensalidade</div>', unsafe_allow_html=True)

        n = len(meses)
        avg = lambda key: sum(d.get(key, 0) for d in meses) / n

        labels = [
            "PIS/COFINS Líq.",
            "ISS",
            "MDR",
            "Manutenção",
            "Benefícios Cliente",
            "PDD",
            "Default",
            "CAC/Logística (diluído)",
            "Margem / Outros",
        ]

        pis_val   = avg("pis_cofins_liq")
        iss_val   = avg("iss")
        mdr_val   = avg("mdr")
        manut_val = avg("manutencao")
        ben_val   = avg("customer_benefits")
        pdd_val   = avg("pdd")
        def_val   = avg("default_m")
        cac_val   = avg("cac") + avg("risk_query") + avg("logistics_ass")
        preco_m   = avg("preco_mensal")
        total_custos = pis_val + iss_val + mdr_val + manut_val + ben_val + pdd_val + def_val + cac_val
        margem_val = max(0, preco_m - total_custos)

        values = [pis_val, iss_val, mdr_val, manut_val, ben_val, pdd_val, def_val, cac_val, margem_val]

        cores_pizza = [
            "#E74C3C", "#C0392B", "#E67E22",
            "#F39C12", "#43A047", "#1ABC9C",
            "#3498DB", "#2980B9", "#304D3C",
        ]

        fig_pizza = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=cores_pizza,
                textinfo="label+percent",
                textfont_size=11,
                hole=0.4,
                hovertemplate="%{label}<br>R$ %{value:,.2f}<br>%{percent}<extra></extra>",
            )
        )
        fig_pizza.update_layout(
            separators=",.",
            showlegend=False,
            height=360,
            margin=dict(l=5, r=5, t=10, b=10),
            paper_bgcolor="white",
            annotations=[
                dict(
                    text=f"{fmt_brl(preco_m)}",
                    x=0.5, y=0.5,
                    font_size=14,
                    font_color=COR_AZUL,
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig_pizza, use_container_width=True)

    # --------------------------------------------------
    # Gráfico de barras empilhadas: evolução mensal
    # --------------------------------------------------
    with col_barra:
        st.markdown('<div class="section-title">Composição por Mês</div>', unsafe_allow_html=True)

        cal_ms   = [d["cal_m"] for d in meses]
        receitas = [d["receita_liq"] for d in meses]
        manuts   = [-d["manutencao"] for d in meses]
        pdds     = [-d["pdd"] for d in meses]
        defaults = [-d["default_m"] for d in meses]
        bens     = [-d["customer_benefits"] for d in meses]

        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            x=cal_ms, y=receitas,
            name="Receita Líquida",
            marker_color=COR_VERDE,
            hovertemplate="Mês %{x}<br>Receita Liq.: R$ %{y:,.2f}<extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            x=cal_ms, y=manuts,
            name="Manutenção",
            marker_color=COR_LARANJA,
            hovertemplate="Mês %{x}<br>Manutenção: R$ %{y:,.2f}<extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            x=cal_ms, y=pdds,
            name="PDD",
            marker_color="#E67E22",
            hovertemplate="Mês %{x}<br>PDD: R$ %{y:,.2f}<extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            x=cal_ms, y=defaults,
            name="Default",
            marker_color=COR_VERMELHO,
            hovertemplate="Mês %{x}<br>Default: R$ %{y:,.2f}<extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            x=cal_ms, y=bens,
            name="Benefícios",
            marker_color="#3498DB",
            hovertemplate="Mês %{x}<br>Benefícios: R$ %{y:,.2f}<extra></extra>",
        ))

        fig_bar.update_layout(
            separators=",.",
            barmode="relative",
            height=360,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=11),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Mês Calendário", gridcolor="#F0F0F0"),
            yaxis=dict(title="R$", gridcolor="#F0F0F0", tickprefix="R$ ", tickformat=",.0f"),
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # --------------------------------------------------
    # Evolução da depreciação
    # --------------------------------------------------
    st.markdown('<div class="section-title">Evolução do Book Value do Device</div>', unsafe_allow_html=True)

    book_values = [d["book_value"] for d in result.monthly_details if d["cal_m"] >= 0]
    cal_ms_all  = [d["cal_m"] for d in result.monthly_details if d["cal_m"] >= 0]

    fig_dep = go.Figure()
    fig_dep.add_trace(go.Scatter(
        x=cal_ms_all,
        y=book_values,
        mode="lines+markers",
        line=dict(color=COR_AZUL, width=2.5),
        marker=dict(size=4),
        name="Book Value",
        hovertemplate="Mês %{x}<br>Book Value: R$ %{y:,.2f}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(30,58,95,0.06)",
    ))

    # Linha do valor de compra
    fig_dep.add_hline(
        y=result.asset.purchase_price,
        line_dash="dash",
        line_color=COR_LARANJA,
        annotation_text=f"Custo: {fmt_brl(result.asset.purchase_price)}",
        annotation_position="right",
    )

    fig_dep.update_layout(
        separators=",.",
        height=260,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        margin=dict(l=10, r=10, t=15, b=10),
        xaxis=dict(title="Mês Calendário", gridcolor="#F0F0F0"),
        yaxis=dict(title="R$", gridcolor="#F0F0F0", tickprefix="R$ ", tickformat=",.0f"),
    )

    st.plotly_chart(fig_dep, use_container_width=True)


# ----------------------------------------------------------
# TAB 4: COMPARAR CENÁRIOS
# ----------------------------------------------------------

def render_tab_comparar(ativo, client_type, params_base):
    """Renderiza a aba de comparação de cenários por prazo."""
    from pricing_engine.models.contract import VALID_TERMS

    prazos = VALID_TERMS[client_type]

    st.markdown(
        f'<div class="section-title">Comparação de Prazos — {ativo.name} ({client_type.value})</div>',
        unsafe_allow_html=True,
    )

    # Barra de progresso enquanto calcula (cascade completo)
    progress_bar = st.progress(0, text="Calculando cenários...")
    try:
        precos_por_prazo = compute_all_prices(ativo, client_type, params_base)
        # Ordena do mais curto para o mais longo (exibição)
        resultados = [precos_por_prazo[t] for t in sorted(precos_por_prazo)]
    except Exception as e:
        st.warning(f"Erro ao calcular cenários: {e}")
        resultados = []
    progress_bar.empty()

    if not resultados:
        st.error("Não foi possível calcular nenhum cenário.")
        return

    # --------------------------------------------------
    # Tabela comparativa
    # --------------------------------------------------
    linhas = []
    for res in resultados:
        s = res.summary_dict()
        linhas.append({
            "Prazo (meses)": s["prazo_meses"],
            "Ciclo Econômico": f"{s['ciclo_economico_meses']}m",
            "Renovação": f"+{s['meses_renovacao']}m" if s["tem_renovacao"] else "—",
            "Preço Sugerido": fmt_brl(res.suggested_price),
            "Breakeven": fmt_brl(res.breakeven_price),
            "Preço Renovação": fmt_brl(res.renewal_price) if res.renewal_price else "—",
            "TIR Desalav.": fmt_pct(res.unlevered_irr),
            "TIR Alav.": fmt_pct(res.levered_irr),
            "Payback (meses)": res.payback_months if res.payback_months else "—",
        })

    df_comp = pd.DataFrame(linhas)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # Gráfico de comparação de preços e TIR
    # --------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Preços por Prazo</div>', unsafe_allow_html=True)

        prazos_labels = [f"{r.contract.term_months}m" for r in resultados]
        precos_sug    = [r.suggested_price or 0 for r in resultados]
        precos_be     = [r.breakeven_price or 0 for r in resultados]

        fig_preco = go.Figure()
        fig_preco.add_trace(go.Bar(
            x=prazos_labels, y=precos_sug,
            name="Sugerido",
            marker_color=COR_AZUL,
        ))
        fig_preco.add_trace(go.Bar(
            x=prazos_labels, y=precos_be,
            name="Breakeven",
            marker_color=COR_CINZA,
        ))
        fig_preco.update_layout(
            separators=",.",
            barmode="group",
            height=300,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis=dict(tickprefix="R$ ", tickformat=",.0f", gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig_preco, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">TIR por Prazo</div>', unsafe_allow_html=True)

        tirs_desalav = [(r.unlevered_irr or 0) * 100 for r in resultados]
        tirs_alav    = [(r.levered_irr or 0) * 100 for r in resultados]

        fig_tir = go.Figure()
        fig_tir.add_trace(go.Scatter(
            x=prazos_labels, y=tirs_desalav,
            name="TIR Desalav.",
            mode="lines+markers",
            line=dict(color=COR_AZUL, width=2.5),
            marker=dict(size=8),
        ))
        fig_tir.add_trace(go.Scatter(
            x=prazos_labels, y=tirs_alav,
            name="TIR Alav.",
            mode="lines+markers",
            line=dict(color=COR_LARANJA, width=2.5, dash="dot"),
            marker=dict(size=8),
        ))
        # Linha da TIR mínima
        fig_tir.add_hline(
            y=params_base.min_unlevered_irr * 100,
            line_dash="dash",
            line_color=COR_VERMELHO,
            annotation_text=f"TIR mín. {params_base.min_unlevered_irr:.0%}",
        )
        fig_tir.update_layout(
            separators=",.",
            height=300,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis=dict(title="%", gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig_tir, use_container_width=True)

    # --------------------------------------------------
    # Botão de exportação
    # --------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Exportar Comparação como CSV", use_container_width=False):
        buffer = io.StringIO()
        df_export_rows = []
        for res in resultados:
            s = res.summary_dict()
            df_export_rows.append({
                "Ativo": s["ativo_nome"],
                "Cliente": s["cliente"],
                "Prazo (meses)": s["prazo_meses"],
                "Ciclo Econômico (meses)": s["ciclo_economico_meses"],
                "Renovação (meses)": s["meses_renovacao"],
                "Preço Sugerido (R$)": res.suggested_price,
                "Preço Breakeven (R$)": res.breakeven_price,
                "Preço Renovação (R$)": res.renewal_price,
                "TIR Desalavancada": res.unlevered_irr,
                "TIR Alavancada": res.levered_irr,
                "Payback (meses)": res.payback_months,
            })
        pd.DataFrame(df_export_rows).to_csv(buffer, index=False)
        st.download_button(
            label="Baixar CSV",
            data=buffer.getvalue(),
            file_name=f"comparacao_{ativo.id}_{client_type.value}.csv",
            mime="text/csv",
        )

    return resultados


# ----------------------------------------------------------
# TAB 5: DFC — FLUXO DE CAIXA POR NATUREZA
# ----------------------------------------------------------

def render_tab_dfc(result):
    """Renderiza a aba de DFC: FCO / FCI / FCF mensal e anual."""
    if not result.monthly_details:
        st.warning("Nenhum dado de DFC disponível.")
        return

    details = result.monthly_details

    # --------------------------------------------------
    # Cards de totais do ciclo completo
    # --------------------------------------------------
    if result.annual_cashflows:
        total_row = result.annual_cashflows[-1]  # última linha = Total
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(card_metrica(
                "FCO Total do Ciclo",
                fmt_brl(total_row.get("fco_total", 0)),
                sub="Operacional",
                destaque=total_row.get("fco_total", 0) > 0,
            ), unsafe_allow_html=True)
        with col2:
            st.markdown(card_metrica(
                "FCI Total do Ciclo",
                fmt_brl(total_row.get("fci_total", 0)),
                sub="Investimento",
            ), unsafe_allow_html=True)
        with col3:
            st.markdown(card_metrica(
                "FCF Total do Ciclo",
                fmt_brl(total_row.get("fcf_total", 0)),
                sub="Financiamento",
            ), unsafe_allow_html=True)
        with col4:
            cf_t = total_row.get("cf_total", 0)
            st.markdown(card_metrica(
                "CF Total do Ciclo",
                fmt_brl(cf_t),
                sub="FCO + FCI + FCF",
                destaque=cf_t > 0,
                alerta=cf_t < 0,
            ), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------
    # Gráfico 2: FCO / FCI / FCF por ano
    # --------------------------------------------------
    if result.annual_cashflows and len(result.annual_cashflows) > 1:
        st.markdown('<div class="section-title">Consolidação Anual por Natureza</div>', unsafe_allow_html=True)

        # Exclui a linha "Total" do gráfico
        anos_rows = [r for r in result.annual_cashflows if r["ano"] != "Total"]
        anos_labels = [f"Ano {r['ano']}" for r in anos_rows]
        fco_anual  = [r.get("fco_total", 0) for r in anos_rows]
        fci_anual  = [r.get("fci_total", 0) for r in anos_rows]
        fcf_anual  = [r.get("fcf_total", 0) for r in anos_rows]
        cf_anual   = [r.get("cf_total",  0) for r in anos_rows]

        fig_anual = go.Figure()
        fig_anual.add_trace(go.Bar(
            x=anos_labels, y=fco_anual,
            name="FCO — Operacional",
            marker_color=COR_VERDE,
            hovertemplate="%{x}<br>FCO: R$ %{y:,.2f}<extra></extra>",
        ))
        fig_anual.add_trace(go.Bar(
            x=anos_labels, y=fci_anual,
            name="FCI — Investimento",
            marker_color=COR_VERMELHO,
            hovertemplate="%{x}<br>FCI: R$ %{y:,.2f}<extra></extra>",
        ))
        fig_anual.add_trace(go.Bar(
            x=anos_labels, y=fcf_anual,
            name="FCF — Financiamento",
            marker_color=COR_LARANJA,
            hovertemplate="%{x}<br>FCF: R$ %{y:,.2f}<extra></extra>",
        ))
        fig_anual.add_trace(go.Scatter(
            x=anos_labels, y=cf_anual,
            name="CF Total",
            mode="lines+markers",
            line=dict(color=COR_AZUL, width=2.5),
            marker=dict(size=8),
            hovertemplate="%{x}<br>CF Total: R$ %{y:,.2f}<extra></extra>",
        ))
        fig_anual.add_hline(y=0, line_width=1, line_color="#DEE2E6")
        fig_anual.update_layout(
            separators=",.",
            barmode="group",
            height=320,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=11),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor="#F0F0F0"),
            yaxis=dict(gridcolor="#F0F0F0", tickprefix="R$ ", tickformat=",.0f"),
        )
        st.plotly_chart(fig_anual, use_container_width=True)

    # --------------------------------------------------
    # Tabela anual detalhada (todas as linhas abertas)
    # --------------------------------------------------
    if result.annual_cashflows:
        st.markdown('<div class="section-title">Demonstração Anual — Linhas Abertas</div>', unsafe_allow_html=True)

        anos_cols = [r for r in result.annual_cashflows if r["ano"] != "Total"]
        total_col = next((r for r in result.annual_cashflows if r["ano"] == "Total"), {})
        col_headers = [f"Ano {r['ano']}" for r in anos_cols] + ["<b>Total</b>"]
        all_rows = anos_cols + [total_col]

        def cell(row, key):
            v = row.get(key, 0.0)
            if v is None:
                return "—"
            try:
                return fmt_brl(float(v))
            except (TypeError, ValueError):
                return "—"

        LINHAS_DFC = [
            ("FCO — OPERACIONAL", None, "header"),
            ("Receita bruta", "fco_receita_bruta", "+"),
            ("( - ) PIS/COFINS líquido", "fco_pis_cofins", "-"),
            ("( - ) ISS", "fco_iss", "-"),
            ("( - ) MDR", "fco_mdr", "-"),
            ("( - ) Manutenção", "fco_manutencao", "-"),
            ("( - ) PDD", "fco_pdd", "-"),
            ("( - ) Default", "fco_default", "-"),
            ("( - ) Logística AT (troca)", "fco_logistics_troca", "-"),
            ("( - ) CAC", "fco_cac", "-"),
            ("( - ) Consulta de risco", "fco_risk_query", "-"),
            ("( - ) Logística entrega", "fco_logistics_ass", "-"),
            ("( - ) Benefícios ao cliente", "fco_customer_benefits", "-"),
            ("( - ) Logística devolução", "fco_logistics_devolucao", "-"),
            ("( - ) ICMS", "fco_icms", "-"),
            ("= TOTAL FCO", "fco_total", "total"),
            ("FCI — INVESTIMENTO", None, "header"),
            ("( - ) Compra do ativo", "fci_compra", "-"),
            ("( + ) Venda bruta do ativo", "fci_sale_val", "+"),
            ("( - ) CAC de venda", "fci_cac_venda", "-"),
            ("( - ) Prep. de venda", "fci_prep_venda", "-"),
            ("( - ) Logística de venda", "fci_logistics_venda", "-"),
            ("= TOTAL FCI", "fci_total", "total"),
            ("FCF — FINANCIAMENTO", None, "header"),
            ("( + ) Entrada do funding", "fcf_funding", "+"),
            ("( - ) Custo de captação", "fcf_captacao_fee", "-"),
            ("( - ) Juros da dívida", "fcf_juros", "-"),
            ("( - ) Amortização (principal)", "fcf_principal", "-"),
            ("= TOTAL FCF", "fcf_total", "total"),
            ("= CF TOTAL (FCO + FCI + FCF)", "cf_total", "grand_total"),
        ]

        th_style  = "background:#304D3C;color:white;padding:0.45rem 0.7rem;font-size:0.82rem;white-space:nowrap;"
        hdr_style = "background:#EFF6EA;color:#304D3C;font-weight:700;padding:0.4rem 0.7rem;font-size:0.82rem;"
        tot_style = "background:#EFF6EA;font-weight:700;border-top:2px solid #304D3C;padding:0.4rem 0.7rem;font-size:0.82rem;"
        grand_style = "background:#304D3C;color:white;font-weight:700;padding:0.5rem 0.7rem;font-size:0.85rem;"
        td_style  = "padding:0.38rem 0.7rem;font-size:0.82rem;border-bottom:1px solid #F0F0F0;"
        pos_style = "color:#43A047;"
        neg_style = "color:#E74C3C;"

        header_cells = "".join(f'<th style="{th_style}">{h}</th>' for h in ["Linha"] + col_headers)
        html = f'<table style="width:100%;border-collapse:collapse;"><thead><tr>{header_cells}</tr></thead><tbody>'

        for nome, key, tipo in LINHAS_DFC:
            if tipo == "header":
                cells = "".join(f'<td style="{hdr_style}" colspan="{len(all_rows) + 1}">{nome}</td>')
                html += f"<tr>{cells}</tr>"
                continue

            row_style = grand_style if tipo == "grand_total" else (tot_style if tipo == "total" else td_style)
            td_nome_style = grand_style if tipo == "grand_total" else row_style
            row_html = f'<td style="{td_nome_style}">{nome}</td>'

            for r in all_rows:
                v = r.get(key, 0.0) if key else 0.0
                try:
                    v_f = float(v)
                except (TypeError, ValueError):
                    v_f = 0.0

                display_v = -v_f if tipo == "-" and v_f != 0 else v_f

                if tipo == "grand_total":
                    color = pos_style if v_f >= 0 else neg_style.replace("color:#E74C3C;", "color:#FF9999;")
                    row_html += f'<td style="{grand_style}{color}">{fmt_brl(v_f)}</td>'
                elif tipo == "total":
                    color = pos_style if v_f >= 0 else neg_style
                    row_html += f'<td style="{tot_style}{color}">{fmt_brl(v_f)}</td>'
                else:
                    color = pos_style if (tipo == "+" and v_f > 0) else (neg_style if v_f > 0 and tipo == "-" else "")
                    row_html += f'<td style="{td_style}{color}">{fmt_brl(display_v)}</td>'

            html += f"<tr>{row_html}</tr>"

        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)

    # --------------------------------------------------
    # Tabela mensal DFC — meses nas colunas, rubricas nas linhas
    # --------------------------------------------------
    st.markdown('<div class="section-title">DFC Mensal — Por Natureza</div>', unsafe_allow_html=True)

    LINHAS_MENSAL = [
        ("Tipo do mês",                  "tipo",                "text"),
        ("FCO — OPERACIONAL",            None,                  "header"),
        ("( + ) Receita bruta",          "fco_receita_bruta",   "+"),
        ("( - ) PIS/COFINS líq.",        "fco_pis_cofins",      "-"),
        ("( - ) ISS",                    "fco_iss",             "-"),
        ("( - ) MDR",                    "fco_mdr",             "-"),
        ("( - ) Manutenção",             "fco_manutencao",      "-"),
        ("( - ) PDD",                    "fco_pdd",             "-"),
        ("( - ) Default",                "fco_default",         "-"),
        ("( - ) Log. AT (troca)",        "fco_logistics_troca", "-"),
        ("( - ) CAC",                    "fco_cac",             "-"),
        ("( - ) Consulta de risco",      "fco_risk_query",      "-"),
        ("( - ) Log. entrega",           "fco_logistics_ass",   "-"),
        ("( - ) Benef. cliente",         "fco_customer_benefits","-"),
        ("( - ) Log. devolução",         "fco_logistics_devolucao","-"),
        ("( - ) ICMS",                   "fco_icms",            "-"),
        ("= TOTAL FCO",                  "fco_total",           "total"),
        ("FCI — INVESTIMENTO",           None,                  "header"),
        ("( - ) Compra do ativo",        "fci_compra",          "-"),
        ("( + ) Venda bruta",            "fci_sale_val",        "+"),
        ("( - ) CAC de venda",           "fci_cac_venda",       "-"),
        ("( - ) Prep. venda",            "fci_prep_venda",      "-"),
        ("( - ) Log. venda",             "fci_logistics_venda", "-"),
        ("= TOTAL FCI",                  "fci_total",           "total"),
        ("FCF — FINANCIAMENTO",          None,                  "header"),
        ("( + ) Entrada do funding",     "fcf_funding",         "+"),
        ("( - ) Custo de captação",      "fcf_captacao_fee",    "-"),
        ("( - ) Juros da dívida",        "fcf_juros",           "-"),
        ("( - ) Amortização (principal)","fcf_principal",        "-"),
        ("= TOTAL FCF",                  "fcf_total",           "total"),
        ("= CF TOTAL (FCO+FCI+FCF)",     "cf_total",            "grand_total"),
    ]

    mes_labels = [f"M{d['cal_m']}" for d in details]

    # Estilos
    th_s = "background:#304D3C;color:white;padding:0.35rem 0.5rem;font-size:0.76rem;white-space:nowrap;text-align:center;min-width:68px;"
    td_s = "padding:0.28rem 0.5rem;font-size:0.76rem;text-align:right;border-bottom:1px solid #F0F0F0;white-space:nowrap;"
    td_lbl = "padding:0.28rem 0.5rem;font-size:0.76rem;text-align:left;border-bottom:1px solid #F0F0F0;white-space:nowrap;position:sticky;left:0;background:white;z-index:1;"
    hdr_s = "background:#EFF6EA;color:#304D3C;font-weight:700;padding:0.35rem 0.5rem;font-size:0.76rem;"
    tot_s = "padding:0.32rem 0.5rem;font-size:0.76rem;text-align:right;font-weight:700;border-top:2px solid #304D3C;border-bottom:1px solid #304D3C;white-space:nowrap;"
    tot_lbl = tot_s.replace("text-align:right", "text-align:left") + "position:sticky;left:0;background:#EFF6EA;z-index:1;"
    grd_s = "padding:0.32rem 0.5rem;font-size:0.76rem;text-align:right;font-weight:700;background:#304D3C;color:white;white-space:nowrap;"
    grd_lbl = grd_s.replace("text-align:right", "text-align:left") + "position:sticky;left:0;z-index:1;"

    header_html = f'<th style="{th_s};text-align:left;min-width:180px;position:sticky;left:0;z-index:2;">Rubrica</th>'
    for lbl in mes_labels:
        header_html += f'<th style="{th_s}">{lbl}</th>'

    body_html = ""
    ncols = len(details)
    for nome, key, tipo in LINHAS_MENSAL:
        if tipo == "header":
            body_html += f'<tr><td style="{hdr_s};position:sticky;left:0;z-index:1;" colspan="{ncols + 1}">{nome}</td></tr>'
            continue
        if tipo == "text":
            rh = f'<td style="{td_lbl}">{nome}</td>'
            for d in details:
                rh += f'<td style="{td_s};text-align:center;color:#6C757D;font-size:0.70rem;">{d.get(key,"")}</td>'
        elif tipo in ("total", "grand_total"):
            ls = grd_lbl if tipo == "grand_total" else tot_lbl
            vs = grd_s if tipo == "grand_total" else tot_s
            rh = f'<td style="{ls}">{nome}</td>'
            for d in details:
                v = d.get(key, 0)
                cor = "color:#43A047;" if v >= 0 else ("color:#FF9999;" if tipo == "grand_total" else "color:#E74C3C;")
                rh += f'<td style="{vs}{cor}">{_brl(v)}</td>'
        else:
            rh = f'<td style="{td_lbl}">{nome}</td>'
            for d in details:
                v = d.get(key, 0)
                dv = -v if tipo == "-" and v != 0 else v
                cor = "color:#E74C3C;" if tipo == "-" and v > 0 else ("color:#43A047;" if tipo == "+" and v > 0 else "")
                rh += f'<td style="{td_s}{cor}">{_brl(dv)}</td>'
        body_html += f"<tr>{rh}</tr>"

    st.markdown(f"""
    <div style="overflow-x:auto;max-height:600px;">
        <table style="border-collapse:collapse;min-width:100%;">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{body_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------
# ABA DRE — Demonstração do Resultado do Exercício
# ----------------------------------------------------------

def render_tab_dre(result):
    """Renderiza a aba de DRE: assinatura, venda e consolidação (mensal + anual)."""
    if not result.monthly_details:
        st.warning("Nenhum dado de DRE disponível.")
        return

    details = result.monthly_details
    annual  = result.annual_cashflows or []

    # ----------------------------------------------------------------
    # Estilos da tabela
    # ----------------------------------------------------------------
    th_style    = "background:#304D3C;color:white;padding:0.45rem 0.7rem;font-size:0.82rem;white-space:nowrap;"
    hdr_style   = "background:#EFF6EA;color:#304D3C;font-weight:700;padding:0.4rem 0.7rem;font-size:0.82rem;"
    sub_style   = "background:#EFF6EA;color:#3D6B4F;font-weight:600;padding:0.38rem 0.7rem;font-size:0.80rem;font-style:italic;"
    tot_style   = "background:#EFF6EA;font-weight:700;border-top:2px solid #304D3C;padding:0.4rem 0.7rem;font-size:0.82rem;"
    grand_style = "background:#304D3C;color:white;font-weight:700;padding:0.5rem 0.7rem;font-size:0.85rem;"
    td_style    = "padding:0.38rem 0.7rem;font-size:0.82rem;border-bottom:1px solid #F0F0F0;"
    pos_style   = "color:#43A047;"
    neg_style   = "color:#E74C3C;"

    def _brl(v):
        try:
            return fmt_brl(float(v))
        except (TypeError, ValueError):
            return "—"

    def _color(v, tipo):
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return ""
        if tipo == "grand_total":
            return pos_style if vf >= 0 else "color:#FF9999;"
        if tipo == "total":
            return pos_style if vf >= 0 else neg_style
        if tipo == "+":
            return pos_style if vf > 0 else ""
        if tipo == "-":
            return neg_style if vf > 0 else ""
        return ""

    def build_dre_table(linhas, col_headers, all_rows):
        header_cells = "".join(f'<th style="{th_style}">{h}</th>' for h in ["Linha"] + col_headers)
        html = f'<table style="width:100%;border-collapse:collapse;"><thead><tr>{header_cells}</tr></thead><tbody>'

        for nome, key, tipo in linhas:
            if tipo == "header":
                cells = f'<td style="{hdr_style}" colspan="{len(all_rows)+1}">{nome}</td>'
                html += f"<tr>{cells}</tr>"
                continue
            if tipo == "subheader":
                cells = f'<td style="{sub_style}" colspan="{len(all_rows)+1}">{nome}</td>'
                html += f"<tr>{cells}</tr>"
                continue

            row_s = grand_style if tipo == "grand_total" else (tot_style if tipo == "total" else td_style)
            row_html = f'<td style="{row_s}">{nome}</td>'
            for r in all_rows:
                v = r.get(key, 0.0) if key else 0.0
                display_v = -v if tipo == "-" and v != 0 else v
                c = _color(v, tipo)
                row_html += f'<td style="{row_s}{c}">{_brl(display_v)}</td>'
            html += f"<tr>{row_html}</tr>"

        html += "</tbody></table>"
        return html

    # ----------------------------------------------------------------
    # Visão anual
    # ----------------------------------------------------------------
    anos_rows  = [r for r in annual if r.get("ano") != "Total"]
    total_row  = next((r for r in annual if r.get("ano") == "Total"), {})
    all_rows   = anos_rows + [total_row]
    col_headers = [f"Ano {r['ano']}" for r in anos_rows] + ["<b>Total</b>"]

    # ---- DRE da Assinatura (anual) ----
    st.markdown('<div class="section-title">DRE da Assinatura — Consolidação Anual</div>', unsafe_allow_html=True)

    LINHAS_ASS = [
        ("RECEITA", None, "header"),
        ("( + ) Receita de assinatura", "dre_ass_receita", "+"),
        ("DEDUÇÕES / IMPOSTOS SOBRE RECEITA", None, "header"),
        ("( - ) Impostos sobre receita  (PIS/COFINS + ISS)", "dre_ass_impostos", "-"),
        ("= Receita líquida", "dre_ass_receita_liq", "total"),
        ("CUSTOS", None, "header"),
        ("( - ) Customer benefits", "dre_ass_customer_benefits", "-"),
        ("( - ) Manutenção", "dre_ass_manutencao", "-"),
        ("( - ) Logística", "dre_ass_logistica", "-"),
        ("( + ) Crédito de PIS/COFINS", "dre_ass_credito_pis", "+"),
        ("( - ) ICMS", "dre_ass_icms", "-"),
        ("= Lucro bruto", "dre_ass_lucro_bruto", "total"),
        ("DESPESAS", None, "header"),
        ("( - ) Consulta de risco", "dre_ass_risk_query", "-"),
        ("( - ) CAC", "dre_ass_cac", "-"),
        ("( - ) PDD", "dre_ass_pdd", "-"),
        ("( - ) Default", "dre_ass_default", "-"),
        ("= EBITDA", "dre_ass_ebitda", "total"),
        ("= EBIT da assinatura", "dre_ass_ebit", "grand_total"),
    ]

    if all_rows:
        st.markdown(build_dre_table(LINHAS_ASS, col_headers, all_rows), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ---- DRE da Venda (anual) ----
    st.markdown('<div class="section-title">DRE da Venda — Consolidação Anual</div>', unsafe_allow_html=True)
    st.caption("Evento terminal — ocorre somente no último mês do ciclo econômico.")

    LINHAS_VENDA = [
        ("RECEITA", None, "header"),
        ("( + ) Receita de venda", "dre_venda_receita", "+"),
        ("CUSTOS DA VENDA", None, "header"),
        ("( - ) Valor líquido contábil (NBV)", "dre_venda_custo_venda", "-"),
        ("    → Compra − dep. acumulada (10% a.a.)", "dre_venda_dep_acum", "subheader"),
        ("= Lucro bruto", "dre_venda_lucro_bruto", "total"),
        ("DESPESAS", None, "header"),
        ("( - ) CAC de venda", "dre_venda_cac", "-"),
        ("( - ) Logística de venda", "dre_venda_logistica", "-"),
        ("( - ) Preparação de venda", "dre_venda_prep", "-"),
        ("= EBITDA da venda", "dre_venda_ebitda", "total"),
        ("DEPRECIAÇÃO", None, "header"),
        ("( - ) Depreciação contábil acumulada", "dre_venda_dep_contabil", "-"),
        ("= EBIT da venda", "dre_venda_ebit", "grand_total"),
    ]

    if all_rows:
        st.markdown(build_dre_table(LINHAS_VENDA, col_headers, all_rows), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ---- Consolidação: EBIT → Lucro Líquido (anual) ----
    st.markdown('<div class="section-title">Consolidação — EBIT até Lucro Líquido</div>', unsafe_allow_html=True)

    LINHAS_CONSOL = [
        ("EBIT CONSOLIDADO", None, "header"),
        ("EBIT da assinatura", "dre_ass_ebit", "+"),
        ("EBIT da venda", "dre_venda_ebit", "+"),
        ("= EBIT consolidado", "dre_ebit_consolidado", "total"),
        ("DESPESAS FINANCEIRAS", None, "header"),
        ("( - ) Juros da dívida", "dre_juros", "-"),
        ("( - ) MDR (adquirência)", "dre_mdr", "-"),
        ("= EBT", "dre_ebt", "total"),
        ("IR / CSLL", None, "header"),
        ("IR / CSLL (23,80% × EBT quando positivo)", "dre_ir_csll", "-"),
        ("= Lucro (Prejuízo) Líquido", "dre_lucro_liq", "grand_total"),
    ]

    if all_rows:
        st.markdown(build_dre_table(LINHAS_CONSOL, col_headers, all_rows), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ---- Cards de totais do ciclo ----
    if total_row:
        ebt  = total_row.get("dre_ebt", 0)
        ll   = total_row.get("dre_lucro_liq", 0)
        ebit = total_row.get("dre_ebit_consolidado", 0)
        ir   = total_row.get("dre_ir_csll", 0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(card_metrica(
                "EBIT Consolidado",
                fmt_brl(ebit),
                sub="Ciclo completo",
                destaque=ebit > 0,
                alerta=ebit < 0,
            ), unsafe_allow_html=True)
        with col2:
            st.markdown(card_metrica(
                "EBT",
                fmt_brl(ebt),
                sub="Antes do IR",
                destaque=ebt > 0,
                alerta=ebt < 0,
            ), unsafe_allow_html=True)
        with col3:
            st.markdown(card_metrica(
                "IR / CSLL",
                fmt_brl(ir),
                sub="23,80% sobre EBT+",
                alerta=ir < 0,
            ), unsafe_allow_html=True)
        with col4:
            st.markdown(card_metrica(
                "Lucro Líquido",
                fmt_brl(ll),
                sub="Ciclo completo",
                destaque=ll > 0,
                alerta=ll < 0,
            ), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # Tabela mensal DRE — meses nas colunas, rubricas nas linhas
    # ----------------------------------------------------------------
    st.markdown('<div class="section-title">DRE Mensal — Detalhamento</div>', unsafe_allow_html=True)

    LINHAS_DRE_MENSAL = [
        ("Tipo do mês",                              "tipo",                    "text"),
        ("DRE ASSINATURA",                            None,                     "header"),
        ("( + ) Receita de assinatura",              "dre_ass_receita",         "+"),
        ("( - ) Impostos (PIS/COFINS + ISS)",        "dre_ass_impostos",        "-"),
        ("= Receita líquida",                        "dre_ass_receita_liq",     "total"),
        ("( - ) Customer benefits",                  "dre_ass_customer_benefits","-"),
        ("( - ) Manutenção",                         "dre_ass_manutencao",      "-"),
        ("( - ) Logística",                          "dre_ass_logistica",       "-"),
        ("( + ) Crédito PIS/COFINS",                 "dre_ass_credito_pis",     "+"),
        ("( - ) ICMS",                               "dre_ass_icms",            "-"),
        ("= Lucro bruto",                            "dre_ass_lucro_bruto",     "total"),
        ("( - ) Consulta de risco",                  "dre_ass_risk_query",      "-"),
        ("( - ) CAC",                                "dre_ass_cac",             "-"),
        ("( - ) PDD",                                "dre_ass_pdd",             "-"),
        ("( - ) Default",                            "dre_ass_default",         "-"),
        ("= EBITDA assinatura",                      "dre_ass_ebitda",          "total"),
        ("= EBIT assinatura",                        "dre_ass_ebit",            "grand_total"),
        ("DRE VENDA",                                 None,                     "header"),
        ("( + ) Receita de venda",                   "dre_venda_receita",       "+"),
        ("( - ) Custo da venda (NBV)",               "dre_venda_custo_venda",   "-"),
        ("= Lucro bruto venda",                      "dre_venda_lucro_bruto",   "total"),
        ("( - ) Depreciação contábil",               "dre_venda_dep_contabil",  "-"),
        ("= EBIT venda",                             "dre_venda_ebit",          "grand_total"),
        ("CONSOLIDAÇÃO",                              None,                     "header"),
        ("= EBIT consolidado",                       "dre_ebit_consolidado",    "total"),
        ("( - ) Juros da dívida",                    "dre_juros",               "-"),
        ("( - ) MDR",                                "dre_mdr",                 "-"),
        ("= EBT",                                    "dre_ebt",                 "total"),
        ("IR / CSLL (23,80% × EBT)",                "dre_ir_csll",             "-"),
        ("= Lucro (Prejuízo) Líquido",              "dre_lucro_liq",           "grand_total"),
    ]

    mes_labels = [f"M{d['cal_m']}" for d in details]

    # Estilos (reutiliza padrão das outras tabelas)
    _th = "background:#304D3C;color:white;padding:0.35rem 0.5rem;font-size:0.76rem;white-space:nowrap;text-align:center;min-width:68px;"
    _td = "padding:0.28rem 0.5rem;font-size:0.76rem;text-align:right;border-bottom:1px solid #F0F0F0;white-space:nowrap;"
    _lbl = "padding:0.28rem 0.5rem;font-size:0.76rem;text-align:left;border-bottom:1px solid #F0F0F0;white-space:nowrap;position:sticky;left:0;background:white;z-index:1;"
    _hdr = "background:#EFF6EA;color:#304D3C;font-weight:700;padding:0.35rem 0.5rem;font-size:0.76rem;"
    _tot = "padding:0.32rem 0.5rem;font-size:0.76rem;text-align:right;font-weight:700;border-top:2px solid #304D3C;border-bottom:1px solid #304D3C;white-space:nowrap;"
    _tot_lbl = _tot.replace("text-align:right", "text-align:left") + "position:sticky;left:0;background:#EFF6EA;z-index:1;"
    _grd = "padding:0.32rem 0.5rem;font-size:0.76rem;text-align:right;font-weight:700;background:#304D3C;color:white;white-space:nowrap;"
    _grd_lbl = _grd.replace("text-align:right", "text-align:left") + "position:sticky;left:0;z-index:1;"

    h_html = f'<th style="{_th};text-align:left;min-width:200px;position:sticky;left:0;z-index:2;">Rubrica</th>'
    for lbl in mes_labels:
        h_html += f'<th style="{_th}">{lbl}</th>'

    b_html = ""
    ncols = len(details)
    for nome, key, tipo in LINHAS_DRE_MENSAL:
        if tipo == "header":
            b_html += f'<tr><td style="{_hdr};position:sticky;left:0;z-index:1;" colspan="{ncols + 1}">{nome}</td></tr>'
            continue
        if tipo == "text":
            rh = f'<td style="{_lbl}">{nome}</td>'
            for d in details:
                rh += f'<td style="{_td};text-align:center;color:#6C757D;font-size:0.70rem;">{d.get(key,"")}</td>'
        elif tipo in ("total", "grand_total"):
            ls = _grd_lbl if tipo == "grand_total" else _tot_lbl
            vs = _grd if tipo == "grand_total" else _tot
            rh = f'<td style="{ls}">{nome}</td>'
            for d in details:
                v = d.get(key, 0)
                cor = "color:#43A047;" if v >= 0 else ("color:#FF9999;" if tipo == "grand_total" else "color:#E74C3C;")
                rh += f'<td style="{vs}{cor}">{_brl(v)}</td>'
        else:
            rh = f'<td style="{_lbl}">{nome}</td>'
            for d in details:
                v = d.get(key, 0)
                dv = -v if tipo == "-" and v != 0 else v
                cor = "color:#E74C3C;" if tipo == "-" and v > 0 else ("color:#43A047;" if tipo == "+" and v > 0 else "")
                rh += f'<td style="{_td}{cor}">{_brl(dv)}</td>'
        b_html += f"<tr>{rh}</tr>"

    st.markdown(f"""
    <div style="overflow-x:auto;max-height:650px;">
        <table style="border-collapse:collapse;min-width:100%;">
            <thead><tr>{h_html}</tr></thead>
            <tbody>{b_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------
# APLICAÇÃO PRINCIPAL
# ----------------------------------------------------------

def main():
    """Função principal da aplicação Streamlit."""

    # Inicializa o estado da sessão
    if "resultado" not in st.session_state:
        st.session_state.resultado = None

    # Renderiza a sidebar e coleta os inputs
    inputs = render_sidebar()

    # --------------------------------------------------
    # Cabeçalho principal
    # --------------------------------------------------
    st.markdown(
        """
        <div class="allu-header">
            <h1>Allu Pricing Engine</h1>
            <p>Precificação de assinaturas de eletrônicos — Device as a Service (DaaS)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # Dispara o cálculo quando o botão é clicado
    # --------------------------------------------------
    if inputs["calcular"]:
        with st.spinner("Calculando preço ótimo..."):
            try:
                # Computa todos os prazos com cascade + enforcement para garantir
                # escada correta, depois extrai o prazo selecionado.
                precos_por_prazo = compute_all_prices(
                    inputs["ativo"], inputs["client_type"], inputs["params"]
                )
                resultado = precos_por_prazo.get(inputs["prazo"])
                if resultado is None:
                    raise ValueError(f"Prazo {inputs['prazo']}m não encontrado no resultado.")
                st.session_state.resultado = resultado
                st.session_state.inputs_snapshot = inputs.copy()
            except Exception as e:
                st.error(f"Erro no cálculo: {e}")
                st.session_state.resultado = None

    # --------------------------------------------------
    # Exibe resultado ou mensagem inicial
    # --------------------------------------------------
    if st.session_state.resultado is None:
        # Estado inicial: nenhum cálculo realizado ainda
        st.markdown(
            """
            <div class="info-box" style="font-size:1rem; padding: 1.5rem 2rem;">
                <b>Bem-vindo à Allu Pricing Engine!</b><br><br>
                Configure o ativo, tipo de cliente e prazo na barra lateral,
                e clique em <b>Calcular Preço</b> para ver os resultados.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Exibe tabela de ativos disponíveis
        st.markdown('<div class="section-title">Ativos Disponíveis</div>', unsafe_allow_html=True)
        df_ativos = carregar_ativos_csv(ASSETS_CSV)
        df_display = df_ativos.rename(columns={
            "id": "ID",
            "name": "Nome",
            "category": "Categoria",
            "purchase_price": "Preço Compra (R$)",
            "market_price": "Preço Mercado (R$)",
            "maintenance_annual_pct": "Manutenção a.a.",
        })
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        return

    # --------------------------------------------------
    # Resultado disponível — renderiza as abas
    # --------------------------------------------------
    result = st.session_state.resultado
    snap   = st.session_state.get("inputs_snapshot", inputs)

    # Subtítulo com resumo do que foi calculado
    st.markdown(
        f"""
        <div style="color:#6C757D; font-size:0.88rem; margin-bottom:1rem;">
            Calculado para: <b>{result.asset.name}</b> &nbsp;|&nbsp;
            Cliente: <b>{result.contract.client_type.value}</b> &nbsp;|&nbsp;
            Prazo: <b>{result.contract.term_months} meses</b>
            {"&nbsp;|&nbsp; Renovação: <b>+" + str(result.contract.renewal_months) + " meses</b>" if result.contract.has_renewal else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Resultado",
        "Fluxo de Caixa",
        "Breakdown",
        "DFC — Por Natureza",
        "DRE",
        "Premissas",
    ])

    with tab1:
        render_tab_resultado(result)

    with tab2:
        render_tab_cashflow(result)

    with tab3:
        render_tab_breakdown(result)

    with tab4:
        render_tab_dfc(result)

    with tab5:
        render_tab_dre(result)

    with tab6:
        render_tab_premissas(result)


# Ponto de entrada com autenticação
def run_with_auth():
    """Envolve a aplicação com tela de login."""
    # Carrega credenciais
    auth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auth_config.yaml")
    with open(auth_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )

    authenticator.login()

    if st.session_state.get("authentication_status"):
        # Usuário logado — mostra botão de logout e roda o app
        authenticator.logout("Sair", "sidebar")
        st.sidebar.markdown(
            f'<div style="color:#304D3C;font-size:0.85rem;margin-bottom:1rem;">'
            f'Logado como: <b>{st.session_state.get("name", "")}</b></div>',
            unsafe_allow_html=True,
        )
        main()

    elif st.session_state.get("authentication_status") is False:
        st.error("Usuário ou senha incorretos.")

    elif st.session_state.get("authentication_status") is None:
        st.markdown(
            """
            <div style="text-align:center;margin-top:3rem;">
                <h1 style="color:#304D3C;">Allu Pricing Engine</h1>
                <p style="color:#6C757D;">Faça login para acessar a ferramenta de precificação.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    run_with_auth()
