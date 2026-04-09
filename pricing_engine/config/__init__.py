"""
Módulo de configuração da Allu Pricing Engine.

Exporta funções para carregar parâmetros padrão e curvas de depreciação.

Arquivos de depreciação:
    dep_venda.json    — curvas econômicas/comerciais por categoria (valor de venda final)
    dep_contabil.json — taxas contábeis por categoria (base do cálculo de default)
"""

import json
import os
from functools import lru_cache
from typing import Dict

from pricing_engine.config.settings import DEFAULT_PARAMS

_CONFIG_DIR = os.path.dirname(__file__)

# Curvas de depreciação para VALOR DE VENDA (M5)
_DEP_VENDA_PATH = os.path.join(_CONFIG_DIR, "dep_venda.json")

# Taxas de depreciação CONTÁBIL por categoria (M12 — base do default)
_DEP_CONTABIL_PATH = os.path.join(_CONFIG_DIR, "dep_contabil.json")

# Mantém referência ao arquivo legado para compatibilidade retroativa
_CURVES_PATH_LEGACY = os.path.join(_CONFIG_DIR, "depreciation_curves.json")


@lru_cache(maxsize=1)
def get_depreciation_curves() -> Dict:
    """
    Carrega as curvas de depreciação para VALOR DE VENDA (dep_venda.json).

    Usado em: pricing.py → get_sale_book_value() → calcula o valor de venda final (M5).
    NÃO usado para o cálculo de default — ver get_dep_contabil_curves().

    Usa lru_cache para evitar releituras do disco.

    Retorna:
        Dict com parâmetros de depreciação econômica/comercial por categoria.
        Exemplo: {"iphone": {"dep_method": "exponential", "annual_rate": 0.27, "floor_pct": 0.30}, ...}

    Lança:
        FileNotFoundError: se dep_venda.json não for encontrado.
        json.JSONDecodeError: se o arquivo tiver formato inválido.
    """
    path = _DEP_VENDA_PATH if os.path.exists(_DEP_VENDA_PATH) else _CURVES_PATH_LEGACY

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Arquivo de curvas de depreciação de venda não encontrado.\n"
            f"  Esperado: {_DEP_VENDA_PATH}\n"
            f"  Legado:   {_CURVES_PATH_LEGACY}"
        )

    with open(path, "r", encoding="utf-8") as f:
        curves = json.load(f)

    if "default" not in curves:
        raise ValueError(
            "dep_venda.json deve conter a chave 'default' como fallback de categoria."
        )

    return curves


@lru_cache(maxsize=1)
def get_dep_contabil_curves() -> Dict:
    """
    Carrega as taxas de depreciação CONTÁBIL por categoria (dep_contabil.json).

    Usado em: pricing.py → resolve a taxa contábil por categoria → passa para cashflow
              como params.dep_contabil_pct, substituindo o fallback universal do params.csv.

    A taxa do arquivo por categoria tem PRIORIDADE sobre dep_contabil_pct do params.csv.
    Se a categoria não for encontrada, o motor usa dep_contabil_pct como fallback.

    Usa lru_cache para evitar releituras do disco.

    Retorna:
        Dict com annual_rate por categoria.
        Exemplo: {"iphone": {"annual_rate": 0.20}, "default": {"annual_rate": 0.20}, ...}

    Lança:
        FileNotFoundError: se dep_contabil.json não for encontrado.
    """
    if not os.path.exists(_DEP_CONTABIL_PATH):
        raise FileNotFoundError(
            f"Arquivo de depreciação contábil não encontrado: {_DEP_CONTABIL_PATH}"
        )

    with open(_DEP_CONTABIL_PATH, "r", encoding="utf-8") as f:
        curves = json.load(f)

    if "default" not in curves:
        raise ValueError(
            "dep_contabil.json deve conter a chave 'default' como fallback de categoria."
        )

    return curves


_DATA_DIR = os.path.join(os.path.dirname(_CONFIG_DIR), "data")
_CUSTOMER_BENEFITS_PATH = os.path.join(_DATA_DIR, "customer_benefits.csv")


@lru_cache(maxsize=1)
def get_customer_benefits() -> Dict:
    """
    Carrega a tabela de customer benefits por (categoria, condicao).

    Retorna dict com chave (categoria, condicao) e valor float.
    """
    import csv

    if not os.path.exists(_CUSTOMER_BENEFITS_PATH):
        return {}

    result = {}
    with open(_CUSTOMER_BENEFITS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            cat = row["categoria"].strip().lower()
            cond = row["condicao"].strip().lower()
            val = float(row["customer_benefits"].replace(",", "."))
            result[(cat, cond)] = val
    return result


def lookup_customer_benefits(category: str, condicao: str, fallback: float) -> float:
    """
    Busca customer_benefits para categoria+condição.

    Prioridade:
    1. (categoria, condicao) exato — ex: ("iphone", "novo")
    2. (categoria, "geral")        — ex: ("tablet", "geral")
    3. ("default", "geral")        — fallback da planilha
    4. fallback global (PricingParams.customer_benefits)
    """
    table = get_customer_benefits()
    if not table:
        return fallback

    cat = category.lower().strip()
    cond = condicao.lower().strip() if condicao else "novo"

    if (cat, cond) in table:
        return table[(cat, cond)]
    if (cat, "geral") in table:
        return table[(cat, "geral")]
    if ("default", "geral") in table:
        return table[("default", "geral")]
    return fallback


def get_default_params() -> Dict:
    """
    Retorna uma cópia dos parâmetros padrão do modelo de pricing.

    Retorna uma cópia para evitar mutação acidental dos defaults.
    """
    return DEFAULT_PARAMS.copy()
