"""Etapa 7 - Previsão do futuro (2026) com margem de erro + modelos salvos.

Uso:
    python -m src.prever            # 6 meses à frente, 18 origens de backtest
    python -m src.prever 6 18

O que faz:
  1. Estratégia DIRETA: um modelo para cada horizonte h = 1..H meses à frente
     (features.criar_atributos(h=...)). Nada de previsão recursiva.
  2. Backtest multi-horizonte: para cada uma das N últimas origens de 2024-25,
     treina só com o passado e prevê h meses à frente. Os erros desse teste
     medem a precisão por horizonte E calibram a margem de erro.
  3. Treina a versão final com tudo até o último mês (dez/2025), SALVA os
     modelos em models/ e prevê jan..jun/2026.
  4. Margem de erro (intervalo de 80%): quantis 10% e 90% dos erros do
     backtest. Por região, os erros são padronizados pela raiz da previsão
     (escala de contagem / Poisson), para servir a regiões grandes e pequenas.
  5. Compara com o REAL de 2026, reconstruído do cache do SIH (arquivos até
     jun/2026). Meses com menos de 3 competências posteriores são marcados
     como "parcial" (AIHs atrasadas ainda podem chegar) e ficam fora da nota.

Saídas em outputs/: previsao_total.csv, previsao_regiao.csv,
backtest_horizontes.csv, grafico_previsao.png; modelos em models/.
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import config, io_utils
from .features import colunas_atributos, criar_atributos

SEMENTE = 42
QUANTIS = (0.10, 0.90)       # intervalo de 80%
MESES_P_COMPLETO = 3         # competências posteriores p/ considerar mês completo
MODELS_DIR = config.ROOT / "models"


# ------------------------------------------------------------------ modelos
def fabricas() -> dict:
    m = {"RandomForest": lambda: RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=0.5,
        n_jobs=-1, random_state=SEMENTE)}
    try:
        from lightgbm import LGBMRegressor
        m["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=15,
            min_child_samples=10, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEMENTE, verbose=-1)
    except ImportError:
        print("  (lightgbm não instalado: usando só Random Forest)")
    return m


def prever_modelos(fab: dict, treino: pd.DataFrame, alvo: pd.DataFrame,
                   X: list[str]):
    """Treina cada modelo e devolve previsões (+ Ensemble = média)."""
    saida, ajustados = {}, {}
    for nome, f in fab.items():
        m = f()
        m.fit(treino[X], treino["y"])
        saida[nome] = np.clip(m.predict(alvo[X]), 0, None)
        ajustados[nome] = m
    saida["Ensemble"] = np.mean([saida[n] for n in fab], axis=0)
    return saida, ajustados


def estender(painel: pd.DataFrame, H: int) -> pd.DataFrame:
    """Acrescenta H meses futuros (y desconhecido) para cada unidade."""
    ultimo = painel["mes"].max()
    futuros = pd.date_range(ultimo + pd.offsets.MonthBegin(1), periods=H,
                            freq="MS")
    fut = pd.MultiIndex.from_product([painel["unidade"].unique(), futuros],
                                     names=["unidade", "mes"]).to_frame(index=False)
    return pd.concat([painel, fut], ignore_index=True)


# ------------------------------------------------------------------ backtest
def backtest_e_previsao(painel: pd.DataFrame, H: int, n_origens: int):
    ultimo = painel["mes"].max()
    ext = estender(painel, H)
    fab = fabricas()
    MODELS_DIR.mkdir(exist_ok=True)

    linhas_bt, linhas_prev, X_ref = [], [], None
    for h in range(1, H + 1):
        df = criar_atributos(ext, h=h)
        X = colunas_atributos(df)
        X_ref = X
        obs = df[df["y"].notna()]

        # origens: últimos n_origens meses cujo alvo (origem + h) é observado
        fim = ultimo - pd.DateOffset(months=h)
        origens = pd.date_range(end=fim, periods=n_origens, freq="MS")
        t0 = time.perf_counter()
        for o in origens:
            alvo_mes = o + pd.DateOffset(months=h)
            treino = obs[obs["mes"] <= o]
            teste = obs[obs["mes"] == alvo_mes]
            if teste.empty:
                continue
            prev, _ = prever_modelos(fab, treino, teste, X)
            base = teste[["unidade", "mes", "y"]].copy()
            base["h"], base["origem"] = h, o
            base["Ingenuo"] = teste["y_lag1"].to_numpy()
            base["Sazonal_ingenuo"] = teste["y_sazonal"].to_numpy()
            for nome, p in prev.items():
                base[nome] = p
            linhas_bt.append(base)
        print(f"  h={h}: backtest com {len(origens)} origens "
              f"({time.perf_counter() - t0:.0f}s)")

        # modelo final: tudo até o último mês observado
        alvo = df[df["mes"] == ultimo + pd.DateOffset(months=h)]
        prev, ajustados = prever_modelos(fab, obs, alvo, X)
        for nome, m in ajustados.items():
            joblib.dump(m, MODELS_DIR / f"{nome}_h{h}.joblib")
        out = alvo[["unidade", "mes"]].copy()
        out["h"] = h
        out["Ingenuo"] = alvo["y_lag1"].to_numpy()
        out["Sazonal_ingenuo"] = alvo["y_sazonal"].to_numpy()
        for nome, p in prev.items():
            out[nome] = p
        linhas_prev.append(out)

    meta = {"ultimo_mes_treino": str(ultimo.date()), "horizontes": H,
            "modelos": list(fab), "atributos": X_ref,
            "alvo": f"internações {config.ALVO} {config.GRUPOS_CID[config.ALVO]}",
            "uf": config.UF}
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2,
                                                     ensure_ascii=False))
    return (pd.concat(linhas_bt, ignore_index=True),
            pd.concat(linhas_prev, ignore_index=True), list(fab) + ["Ensemble"])


# ------------------------------------------------------------------ margem de erro
def intervalos(bt: pd.DataFrame, prev: pd.DataFrame, modelos: list[str]):
    """Adiciona <modelo>_lo / <modelo>_hi (80%) por região e no total."""
    q_lo, q_hi = QUANTIS
    prev = prev.copy()
    for m in modelos:
        # por região: erro padronizado e = (y - p) / sqrt(max(p, 1))
        e = (bt["y"] - bt[m]) / np.sqrt(bt[m].clip(lower=1))
        qs = e.groupby(bt["h"]).quantile([q_lo, q_hi]).unstack()
        esc = np.sqrt(prev[m].clip(lower=1))
        prev[f"{m}_lo"] = (prev[m] + prev["h"].map(qs[q_lo]) * esc).clip(lower=0)
        prev[f"{m}_hi"] = prev[m] + prev["h"].map(qs[q_hi]) * esc
        _garantir_ponto(prev, m)

    # total do estado: erros dos totais em cada origem
    cols = ["y", "Ingenuo", "Sazonal_ingenuo"] + modelos
    tot_bt = bt.groupby(["h", "origem", "mes"], as_index=False)[cols].sum()
    tot = prev.groupby(["mes", "h"], as_index=False)[
        ["Ingenuo", "Sazonal_ingenuo"] + modelos].sum()
    for m in modelos:
        err = tot_bt["y"] - tot_bt[m]
        qs = err.groupby(tot_bt["h"]).quantile([q_lo, q_hi]).unstack()
        tot[f"{m}_lo"] = (tot[m] + tot["h"].map(qs[q_lo])).clip(lower=0)
        tot[f"{m}_hi"] = tot[m] + tot["h"].map(qs[q_hi])
        _garantir_ponto(tot, m)
    return prev, tot, tot_bt


def _garantir_ponto(df: pd.DataFrame, m: str) -> None:
    """Se o modelo tiver viés no backtest, a faixa empírica pode não conter a
    própria previsão; alargamos a faixa até ela (escolha conservadora)."""
    df[f"{m}_lo"] = df[[f"{m}_lo", m]].min(axis=1)
    df[f"{m}_hi"] = df[[f"{m}_hi", m]].max(axis=1)


def resumo_backtest(bt: pd.DataFrame, tot_bt: pd.DataFrame,
                    modelos: list[str]) -> pd.DataFrame:
    linhas = []
    for m in ["Ingenuo", "Sazonal_ingenuo"] + modelos:
        for h, g in bt.groupby("h"):
            ok = g[m].notna()
            t = tot_bt[tot_bt["h"] == h]
            linhas.append({
                "modelo": m, "h": h,
                "MAE_regiao": (g.loc[ok, "y"] - g.loc[ok, m]).abs().mean(),
                "RMSE_regiao": ((g.loc[ok, "y"] - g.loc[ok, m]) ** 2).mean() ** .5,
                "erro_pct_total": ((t["y"] - t[m]).abs() / t["y"]).mean() * 100,
            })
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ real 2026
def real_pos_treino(ultimo: pd.Timestamp, mapa: pd.DataFrame) -> pd.DataFrame:
    """Internações reais depois do último mês de treino, a partir do cache SIH."""
    from . import ingest_sih

    arquivos = sorted((config.INTERIM / "sih").glob("*.parquet"))
    if not arquivos:
        return pd.DataFrame(columns=["unidade", "mes", "real", "status"])
    ult_comp = max(pd.Timestamp(f"20{a.stem[-4:-2]}-{a.stem[-2:]}-01")
                   for a in arquivos)
    fim_original = config.ANO_FIM
    try:
        config.ANO_FIM = ult_comp.year
        aih = ingest_sih.consolidar([io_utils.ler(a) for a in arquivos])
    finally:
        config.ANO_FIM = fim_original
    aih = aih[(aih["grupo"] == config.ALVO) & (aih["mes"] > ultimo)
              & (aih["mes"] <= ult_comp)]
    aih = aih.merge(mapa, left_on="munic_res", right_on="munic", how="inner")
    real = (aih.groupby(["reg_imediata_id", "mes"]).size()
               .rename("real").reset_index()
               .rename(columns={"reg_imediata_id": "unidade"}))
    meses_depois = ((ult_comp.year - real["mes"].dt.year) * 12
                    + ult_comp.month - real["mes"].dt.month)
    real["status"] = np.where(meses_depois >= MESES_P_COMPLETO,
                              "completo", "parcial")
    real["unidade"] = real["unidade"].astype(str)
    return real


# ------------------------------------------------------------------ gráfico
def grafico(painel: pd.DataFrame, tot: pd.DataFrame, modelo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = painel.groupby("mes")["y"].sum()
    hist = hist[hist.index >= hist.index.max() - pd.DateOffset(months=35)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hist.index, hist.values, color="#333", lw=1.6, label="Histórico")
    ax.fill_between(tot["mes"], tot[f"{modelo}_lo"], tot[f"{modelo}_hi"],
                    color="#2a7de1", alpha=.18, label="Faixa de 80%")
    ax.plot(tot["mes"], tot[modelo], color="#2a7de1", lw=2, marker="o",
            ms=4, label=f"Previsão ({modelo})")
    if "real" in tot:
        for st, mk in (("completo", "o"), ("parcial", "x")):
            r = tot[tot["status"] == st]
            if len(r):
                ax.scatter(r["mes"], r["real"], color="#d9480f", marker=mk,
                           zorder=5, label=f"Real 2026 ({st})")
    ax.set_title(f"Internações F30–F39 ({config.UF}): previsão com margem de erro")
    ax.set_ylabel("Internações/mês")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.OUTPUTS / "grafico_previsao.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_origens = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    P, O = config.PROCESSED, config.OUTPUTS

    painel = io_utils.ler(P / "painel_regiao.parquet")
    mun = io_utils.ler(P / "ibge_municipios.parquet")
    nomes = (mun.drop_duplicates("reg_imediata_id")
                .set_index("reg_imediata_id")["reg_imediata"])
    ultimo = painel["mes"].max()
    print(f"Treino até {ultimo:%Y-%m} | prevendo {H} meses | "
          f"{n_origens} origens de backtest")

    bt, prev, modelos = backtest_e_previsao(painel, H, n_origens)
    prev, tot, tot_bt = intervalos(bt, prev, modelos)
    res = resumo_backtest(bt, tot_bt, modelos)

    real = real_pos_treino(ultimo, mun[["munic", "reg_imediata_id"]])
    prev = prev.merge(real, on=["unidade", "mes"], how="left")
    rt = real.groupby("mes").agg(real=("real", "sum"), status=("status", "first"))
    tot = tot.merge(rt, on="mes", how="left")
    prev.insert(1, "regiao", prev["unidade"].map(nomes))

    # escolhe o melhor modelo pelo erro no total (média dos horizontes)
    ml = res[res["modelo"].isin(modelos)]
    melhor = ml.groupby("modelo")["erro_pct_total"].mean().idxmin()

    res.to_csv(O / "backtest_horizontes.csv", index=False)
    prev.to_csv(O / "previsao_regiao.csv", index=False)
    tot.to_csv(O / "previsao_total.csv", index=False)
    bt.to_csv(O / "backtest_previsoes.csv", index=False)
    grafico(painel, tot, melhor)

    pd.set_option("display.width", 160)
    print("\n=== Erro médio no total do estado por horizonte (%) ===")
    print(res.pivot(index="modelo", columns="h", values="erro_pct_total")
             .round(1).to_string())
    print(f"\nMelhor modelo: {melhor}")
    print(f"\n=== Previsão {config.UF} – total (faixa de 80%) ===")
    t = tot[["mes", melhor, f"{melhor}_lo", f"{melhor}_hi"]].copy()
    if "real" in tot:
        t["real"], t["status"] = tot["real"], tot["status"]
        ok = t["status"] == "completo"
        t["erro_%"] = np.where(ok, (t[melhor] - t["real"]) / t["real"] * 100,
                               np.nan)
        t["dentro_da_faixa"] = np.where(
            t["real"].notna(),
            (t["real"] >= t[f"{melhor}_lo"]) & (t["real"] <= t[f"{melhor}_hi"]),
            pd.NA)
    t["mes"] = t["mes"].dt.strftime("%Y-%m")
    print(t.round(1).to_string(index=False))
    print(f"\nModelos salvos em {MODELS_DIR}")
    print(f"Arquivos em {O}")


if __name__ == "__main__":
    main()
