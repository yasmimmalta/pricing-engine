"""
Modelo de dados para o ativo (device) a ser precificado.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Asset:
    """
    Representa um device (ativo) para fins de precificação.

    Atributos:
        id: Identificador único do modelo (ex: 'ip16_128').
        name: Nome comercial do device (ex: 'iPhone 16 128GB').
        category: Categoria para seleção da curva de depreciação.
                  Valores válidos: 'iphone', 'macbook', 'console', 'apple_watch', 'default'.
        purchase_price: Preço de aquisição do device pela Allu (R$).
        market_price: Valor de mercado atual (usado como base da depreciação — R$).
                      Geralmente menor que purchase_price após ICMS.
        maintenance_annual_pct: Percentual anual de custo de manutenção sobre purchase_price.
                                Padrão Allu: 11,22% a.a. (0.1122).
        storage: Capacidade de armazenamento (ex: '128GB', '256GB'). Opcional.
        generation: Geração/modelo (ex: 'iPhone 16', 'MacBook Air M3'). Opcional.
    """

    id: str
    name: str
    category: str
    purchase_price: float
    market_price: float
    maintenance_annual_pct: float  # ex: 0.1122 para 11.22% a.a.
    storage: Optional[str] = None
    generation: Optional[str] = None
    condicao: str = "novo"  # "novo" ou "usado" — usado para customer_benefits diferenciado

    # Categorias válidas reconhecidas pelas curvas de depreciação
    VALID_CATEGORIES = frozenset(
        ["iphone", "macbook", "notebook_windows", "console", "apple_watch", "default"]
    )

    def __post_init__(self):
        """Valida os campos após inicialização."""
        # Validação: id não pode ser vazio
        if not self.id or not self.id.strip():
            raise ValueError("Asset.id não pode ser vazio.")

        # Validação: name não pode ser vazio
        if not self.name or not self.name.strip():
            raise ValueError("Asset.name não pode ser vazio.")

        # Validação: purchase_price deve ser positivo
        if self.purchase_price <= 0:
            raise ValueError(
                f"Asset.purchase_price deve ser positivo. Recebido: {self.purchase_price}"
            )

        # Validação: market_price deve ser positivo
        if self.market_price <= 0:
            raise ValueError(
                f"Asset.market_price deve ser positivo. Recebido: {self.market_price}"
            )

        # Validação: maintenance_annual_pct deve estar em range razoável (0 a 100%)
        if not (0 <= self.maintenance_annual_pct <= 1.0):
            raise ValueError(
                f"Asset.maintenance_annual_pct deve estar entre 0 e 1 (ex: 0.1122 para 11.22%). "
                f"Recebido: {self.maintenance_annual_pct}"
            )

        # Normaliza categoria e condição para lowercase
        self.category = self.category.lower().strip()
        self.condicao = self.condicao.lower().strip() if self.condicao else "novo"

        # Aviso se categoria não reconhecida (não bloqueia — usa 'default' como fallback)
        if self.category not in self.VALID_CATEGORIES:
            import warnings
            warnings.warn(
                f"Categoria '{self.category}' não reconhecida. "
                f"Será usada a curva 'default'. "
                f"Categorias válidas: {sorted(self.VALID_CATEGORIES)}",
                UserWarning,
                stacklevel=2,
            )

    @property
    def maintenance_monthly(self) -> float:
        """Custo mensal de manutenção (R$) = purchase_price * maintenance_annual_pct / 12."""
        return self.purchase_price * self.maintenance_annual_pct / 12

    def __repr__(self) -> str:
        return (
            f"Asset(id='{self.id}', name='{self.name}', "
            f"purchase_price=R${self.purchase_price:.2f}, "
            f"market_price=R${self.market_price:.2f})"
        )
