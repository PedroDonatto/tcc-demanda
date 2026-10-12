"""Leitura/escrita de tabelas (Parquet). Centralizado para facilitar testes."""
from pathlib import Path

import pandas as pd


def salvar(df: pd.DataFrame, caminho: Path) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(caminho, index=False)


def ler(caminho: Path) -> pd.DataFrame:
    return pd.read_parquet(caminho)


def existe(caminho: Path) -> bool:
    return Path(caminho).exists()
