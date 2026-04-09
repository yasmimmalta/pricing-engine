"""
Módulo de depreciação de ativos (devices) da Allu Pricing Engine.

Implementa:
- Construção do schedule de depreciação (linear ou exponencial) com floor
- Mapeamento de mês calendário para mês econômico
- Seleção da curva de depreciação por categoria

Métodos suportados:
    linear      — depreciação linear do market_price até o floor_value
    exponential — depreciação por taxa de declínio composta (declining balance),
                  mais realista para ativos como iPhone onde a queda é maior nos
                  primeiros anos e diminui ao longo do tempo
"""

import math
from typing import Dict, List, Optional


def build_dep_schedule(
    market_price: float,
    eco_total: int,
    floor_pct: float,
    dep_method: str = "linear",
    annual_rate: Optional[float] = None,
) -> List[float]:
    """
    Constrói o schedule de depreciação do ativo ao longo de eco_total meses.

    Suporta dois métodos:

    LINEAR (dep_method="linear"):
        Deprecia uniformemente do market_price até o floor_value.
        floor_value  = market_price * floor_pct
        depr_mensal  = (market_price - floor_value) / eco_total
        dep[m]       = market_price - depr_mensal * m

    EXPONENCIAL (dep_method="exponential"):
        Deprecia por taxa composta (declining balance). A queda é maior nos
        primeiros meses e diminui ao longo do tempo, refletindo o comportamento
        real do mercado de seminovos para ativos como iPhone.
        monthly_retention = (1 - annual_rate) ^ (1/12)
        dep[m]            = market_price * monthly_retention ^ m
        O floor_pct atua como valor mínimo de segurança.

    Em ambos os métodos, o valor nunca cai abaixo de floor_value.

    Args:
        market_price: Valor de mercado atual do device (R$).
        eco_total: Duração total do ciclo econômico em meses (36 B2C, 48 B2B).
        floor_pct: Percentual mínimo residual do market_price ao final do ciclo.
        dep_method: "linear" (padrão) ou "exponential".
        annual_rate: Taxa anual de declínio (obrigatório para dep_method="exponential").
                     Ex: 0.27 para 27% a.a. → retém ~73% após 12m, ~53% após 24m,
                     ~39% após 36m.

    Retorna:
        Lista com eco_total+1 valores (índice 0 até eco_total inclusive).
        dep_schedule[0]        = market_price
        dep_schedule[eco_total] = valor residual ao fim do ciclo (>= floor_value)

    Exemplos:
        Linear (macbook, floor=30%):
            >>> dep = build_dep_schedule(5000, 36, 0.30, "linear")
            >>> dep[0]   # 5000.0
            >>> dep[36]  # 1500.0 (30% de 5000)

        Exponencial (iPhone, annual_rate=27%):
            >>> dep = build_dep_schedule(6999, 36, 0.30, "exponential", 0.27)
            >>> dep[0]   # 6999.0
            >>> dep[12]  # ~5109  (73% — 1 ano)
            >>> dep[24]  # ~3727  (53% — 2 anos)
            >>> dep[36]  # ~2720  (39% — 3 anos, acima do floor de 30% = 2100)
    """
    if market_price <= 0:
        raise ValueError(f"market_price deve ser positivo. Recebido: {market_price}")
    if eco_total <= 0:
        raise ValueError(f"eco_total deve ser positivo. Recebido: {eco_total}")
    if not (0 <= floor_pct < 1):
        raise ValueError(
            f"floor_pct deve estar entre 0 e 1 (exclusive). Recebido: {floor_pct}"
        )

    floor_value = market_price * floor_pct

    if dep_method == "exponential":
        if annual_rate is None or annual_rate <= 0 or annual_rate >= 1:
            raise ValueError(
                f"annual_rate obrigatório e entre 0 e 1 para dep_method='exponential'. "
                f"Recebido: {annual_rate}"
            )
        # Taxa mensal de retenção: (1 - annual_rate)^(1/12)
        monthly_retention = (1.0 - annual_rate) ** (1.0 / 12.0)
        schedule = [
            market_price * (monthly_retention ** m)
            for m in range(eco_total + 1)
        ]

    else:
        # Linear: 20% ao ano → 1,667%/mês (independente do eco_total)
        # A taxa anual vem do annual_rate do JSON. Fallback para o método
        # antigo (proporcional ao eco_total) apenas se annual_rate for None.
        if annual_rate is not None:
            monthly_rate = annual_rate / 12.0
            schedule = [
                market_price * (1.0 - monthly_rate * m)
                for m in range(eco_total + 1)
            ]
        else:
            depr_mensal = (market_price - floor_value) / eco_total
            schedule = [
                market_price - depr_mensal * m
                for m in range(eco_total + 1)
            ]

    # Garante que o valor nunca caia abaixo do floor
    schedule = [max(v, floor_value) for v in schedule]

    return schedule


def get_dep_params(category: str, curves: Dict) -> Dict:
    """
    Retorna os parâmetros completos de depreciação para a categoria do device.

    Se a categoria não for encontrada no dict de curvas, usa a curva 'default'.

    Args:
        category: Categoria do device (ex: 'iphone', 'macbook'). Case-insensitive.
        curves: Dicionário de curvas carregado do depreciation_curves.json.
                Estrutura esperada:
                {
                    "iphone": {
                        "dep_method": "exponential",
                        "annual_rate": 0.27,
                        "floor_pct": 0.30,
                        ...
                    },
                    "default": {...}
                }

    Retorna:
        Dict com os campos: dep_method, annual_rate, floor_pct, description.

    Exemplo:
        >>> params = get_dep_params("iphone", curves)
        >>> params["dep_method"]   # "exponential"
        >>> params["annual_rate"]  # 0.27
        >>> params["floor_pct"]    # 0.30
    """
    cat_lower = category.lower().strip()

    if cat_lower in curves:
        return curves[cat_lower]

    if "default" not in curves:
        raise KeyError(
            "Chave 'default' não encontrada no dicionário de curvas de depreciação. "
            "Verifique o arquivo depreciation_curves.json."
        )

    return curves["default"]


def get_floor_pct(category: str, curves: Dict) -> float:
    """
    Retorna apenas o floor_pct para a categoria. Wrapper de get_dep_params.

    Mantido para compatibilidade com código legado.
    Prefira get_dep_params() para acessar todos os parâmetros.
    """
    return get_dep_params(category, curves)["floor_pct"]


# Tabela de depreciação escalonada para o valor de venda final (demais categorias)
# Base: sempre market_price. Aplicação: acumulada sobre o saldo anterior.
# Fonte: planilha Excel Allu (PROCH sobre tabela de checkpoints)
_STEPPED_SALE_DEPRECIATIONS = [
    (12, 0.20),   # 12 meses: abate 20% do market_price → saldo = 80%
    (24, 0.10),   # 24 meses: abate mais 10%           → saldo = 70%
    (36, 0.10),   # 36 meses: abate mais 10%           → saldo = 60%
    (48, 0.10),   # 48 meses: abate mais 10%           → saldo = 50%
]


def get_sale_book_value(
    market_price: float,
    eco_total: int,
    dep_params: Dict,
) -> float:
    """
    Calcula o valor de venda final do ativo ao término do ciclo econômico.

    Este valor é usado EXCLUSIVAMENTE para o cálculo do valor de venda final
    (M5). NÃO substitui o dep_schedule usado para o default (M12).

    Dois comportamentos distintos por método:

    EXPONENCIAL (iPhone):
        Usa a curva exponencial calculada em eco_total.
        sale_value = market_price × (1 - annual_rate)^(eco_total/12)
        Limitado pelo floor_pct como valor mínimo.

    LINEAR / Stepped (demais categorias):
        Lógica escalonada da planilha Excel, com base sempre no market_price:
        - 12 meses: balance = market_price - market_price × 20%   → 80%
        - 24 meses: balance = balance_12 - market_price × 10%     → 70%
        - 36 meses: balance = balance_24 - market_price × 10%     → 60%
        - 48 meses: balance = balance_36 - market_price × 10%     → 50%
        Percorre os checkpoints até eco_total e para.

    Args:
        market_price: Valor de mercado (varejo) do ativo no início do ciclo (R$).
        eco_total: Duração total do ciclo econômico em meses (36 B2C / 48 B2B).
        dep_params: Dict de parâmetros da categoria (de get_dep_params).

    Retorna:
        float com o valor de venda estimado ao final do ciclo (R$).

    Exemplos (market_price = R$ 1.000):
        Stepped — eco_total=12: 1.000 - 200          = R$ 800
        Stepped — eco_total=24: 800 - 100             = R$ 700
        Stepped — eco_total=36: 700 - 100             = R$ 600
        Stepped — eco_total=48: 600 - 100             = R$ 500

        iPhone (annual_rate=27%, eco_total=36): 1.000 × 0.73^3 ≈ R$ 389
    """
    dep_method = dep_params.get("dep_method", "linear")

    if dep_method == "exponential":
        annual_rate = dep_params["annual_rate"]
        monthly_retention = (1.0 - annual_rate) ** (1.0 / 12.0)
        floor_value = market_price * dep_params.get("floor_pct", 0.0)
        return max(market_price * (monthly_retention ** eco_total), floor_value)

    # Stepped (demais categorias)
    balance = market_price
    for months, pct in _STEPPED_SALE_DEPRECIATIONS:
        if eco_total >= months:
            balance -= market_price * pct
        else:
            break

    # Aplica floor como segurança mínima
    floor_value = market_price * dep_params.get("floor_pct", 0.0)
    return max(balance, floor_value)


def calendar_to_economic(
    cal_m: int,
    term_months: int,
    has_renewal: bool,
) -> int:
    """
    Converte um mês calendário em mês econômico.

    O mês econômico é usado para acessar o schedule de depreciação.
    Não há intervalo entre o contrato inicial e a renovação — a renovação
    começa imediatamente após o término do contrato inicial. Portanto o
    mapeamento é sempre direto: eco_m == cal_m.

    Args:
        cal_m: Mês calendário (1-indexed, ex: 1 = primeiro mês após assinatura).
        term_months: Prazo do contrato inicial em meses.
        has_renewal: True se houver período de renovação.

    Retorna:
        int com o mês econômico correspondente.

    Exemplo (B2C 12m com renovação 24m):
        >>> calendar_to_economic(12, 12, True)   # último mês do contrato
        12
        >>> calendar_to_economic(13, 12, True)   # primeiro mês da renovação
        13
        >>> calendar_to_economic(36, 12, True)   # último mês da renovação
        36
    """
    # Sem gap: mapeamento direto (cal_m == eco_m)
    return cal_m
