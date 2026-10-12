"""Etapa 6 - Treino e avaliação walk-forward: baselines x RF x XGBoost x LightGBM.

Uso:
    python -m src.train                 # painel_regiao, últimos 12 meses de teste
    python -m src.train estado 12
    python -m src.train regiao 24

Protocolo (seção 2.4 do texto):
  - Os últimos N meses são o TESTE. Para cada mês T do teste, cada modelo é
    treinado do zero só com meses < T e prevê o mês T (origem T-1, h = 1).
    É a validação walk-forward com janela expansiva.
  - Nenhum modelo pré-treinado é reaproveitado.
  - Métricas: MAE, RMSE, R², tempo de treino (s) e de inferência (ms por mês
    previsto, todas as unidades de uma vez).

Saídas em outputs/: metricas_<nivel>.csv, previsoes_<nivel>.csv,
importancias_<nivel>.csv e grafico_<nivel>.png
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config, io_utils
from .features import colunas_atributos, criar_atributos

SEMENTE = 42


# ------------------------------------------------------------------ modelos
def modelos_ml() -> dict:
    """Hiperparâmetros iniciais, razoáveis para séries mensais pequenas.
    Ajuste fino deve usar só o período de treino (nunca o teste)."""
    m = {
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, max_features=0.5,
            n_jobs=-1, random_state=SEMENTE),
    }
    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = lambda: XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, random_state=SEMENTE)
    except ImportError:
        print("  (xgboost não instalado: pulando)")
    try:
        from lightgbm import LGBMRegressor
        m["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=15,
            min_child_samples=10, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEMENTE, verbose=-1)
    except ImportError:
        print("  (lightgbm não instalado: pulando)")
    return m


def baselines(teste: pd.DataFrame) -> dict[str, np.ndarray]:
    """Tabela 1 do texto. Usam só valores defasados -> sem vazamento."""
    return {
        "Media_historica": teste["y_media_exp"].to_numpy(),
        "Ingenuo": teste["y_lag1"].to_numpy(),
        "Sazonal_ingenuo": teste["y_lag12"].to_numpy(),
    }


# ------------------------------------------------------------------ avaliação
def walk_forward(df: pd.DataFrame, n_teste: int = 12):
    X_cols = colunas_atributos(df)
    meses = np.sort(df["mes"].unique())
    meses_teste = meses[-n_teste:]
    fabricas = modelos_ml()

    previsoes, custos, importancias = [], [], []
    for i, T in enumerate(meses_teste, 1):
        treino = df[df["mes"] < T]
        teste = df[df["mes"] == T]
        print(f"  origem {pd.Timestamp(T) - pd.offsets.MonthBegin(1):%Y-%m} "
              f"-> prevê {pd.Timestamp(T):%Y-%m} "
              f"(treino: {len(treino):,} linhas) [{i}/{len(meses_teste)}]")
        saida = teste[["unidade", "mes", "y"]].copy()

        for nome, prev in baselines(teste).items():
            saida[nome] = prev

        for nome, fab in fabricas.items():
            modelo = fab()
            t0 = time.perf_counter()
            modelo.fit(treino[X_cols], treino["y"])
            t_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            saida[nome] = np.clip(modelo.predict(teste[X_cols]), 0, None)
            t_pred = time.perf_counter() - t0
            custos.append({"modelo": nome, "mes": T, "treino_s": t_fit,
                           "inferencia_ms": 1000 * t_pred})
            if hasattr(modelo, "feature_importances_"):
                importancias.append(pd.DataFrame({
                    "modelo": nome, "mes": T, "atributo": X_cols,
                    "importancia": modelo.feature_importances_}))
        previsoes.append(saida)

    prev = pd.concat(previsoes, ignore_index=True)
    custos = pd.DataFrame(custos)
    imp = (pd.concat(importancias) if importancias
           else pd.DataFrame(columns=["modelo", "atributo", "importancia"]))
    return prev, custos, imp


def tabela_metricas(prev: pd.DataFrame, custos: pd.DataFrame) -> pd.DataFrame:
    modelos = [c for c in prev.columns if c not in ("unidade", "mes", "y")]
    linhas = []
    for m in modelos:
        ok = prev[m].notna()
        y, p = prev.loc[ok, "y"], prev.loc[ok, m]
        linhas.append({
            "modelo": m,
            "MAE": mean_absolute_error(y, p),
            "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p),
            "n": int(ok.sum()),
        })
    met = pd.DataFrame(linhas)
    if len(custos):
        c = custos.groupby("modelo")[["treino_s", "inferencia_ms"]].mean()
        met = met.merge(c, left_on="modelo", right_index=True, how="left")
    return met.sort_values("MAE").reset_index(drop=True)


def grafico(prev: pd.DataFrame, df: pd.DataFrame, nivel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = df.groupby("mes")["y"].sum()
    tot = prev.drop(columns="unidade").groupby("mes").sum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hist.index, hist.values, color="#444", lw=1.4, label="Observado")
    for m in [c for c in tot.columns if c not in ("y",)]:
        estilo = "--" if m in ("Media_historica", "Ingenuo",
                               "Sazonal_ingenuo") else "-"
        ax.plot(tot.index, tot[m], estilo, lw=1.2, label=m)
    ax.set_title(f"Internações F30–F39 ({config.UF}) – total do período de teste")
    ax.set_ylabel("Internações/mês")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.OUTPUTS / f"grafico_{nivel}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    nivel = sys.argv[1] if len(sys.argv) > 1 else "regiao"
    n_teste = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    painel = io_utils.ler(config.PROCESSED / f"painel_{nivel}.parquet")
    df = criar_atributos(painel)
    print(f"Atributos: {len(colunas_atributos(df))} | linhas: {len(df):,}")

    prev, custos, imp = walk_forward(df, n_teste)
    met = tabela_metricas(prev, custos)

    O = config.OUTPUTS
    met.to_csv(O / f"metricas_{nivel}.csv", index=False)
    prev.to_csv(O / f"previsoes_{nivel}.csv", index=False)
    if len(imp):
        (imp.groupby(["modelo", "atributo"])["importancia"].mean()
            .reset_index().sort_values(["modelo", "importancia"],
                                       ascending=[True, False])
            .to_csv(O / f"importancias_{nivel}.csv", index=False))
    grafico(prev, df, nivel)

    pd.set_option("display.width", 140)
    print("\n=== Métricas no teste (ordenado por MAE) ===")
    print(met.round(3).to_string(index=False))
    print(f"\nArquivos salvos em {O}")


if __name__ == "__main__":
    main()
