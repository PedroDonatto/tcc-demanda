"""Etapa 8 - Laboratório de rigor: os experimentos que a banca vai pedir.

Uso:
    python -m src.experimentos

Tudo no mesmo protocolo walk-forward (12 meses de 2025, 1 mês à frente,
treino do zero a cada mês, só com o passado):

  1. Comparação ampla: chutes, ETS (Holt-Winters, estatístico clássico),
     Random Forest, LightGBM padrão e LightGBM ajustado.
  2. Ablação da engenharia de atributos (o coração do título do TCC):
     A) só meses recentes -> B) + sazonalidade -> C) + janelas móveis
     -> D) + variáveis externas. Mostra o ganho de cada grupo.
  3. Ajuste de hiperparâmetros por busca aleatória, validado SÓ em 2024
     (nunca no teste), incluindo perda de Poisson para contagens.
  4. Significância estatística: teste de Diebold-Mariano (com correção de
     Harvey-Leybourne-Newbold) e Wilcoxon pareado contra o melhor modelo.
  5. Análise de erro por porte da região e por mês; viés.
  6. Explicação das previsões de 2026 (SHAP via pred_contrib do LightGBM),
     agrupada em pistas compreensíveis.

Saída: outputs/experimentos.json (+ experimentos_resumo.csv)
"""
from __future__ import annotations

import json
import random
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from . import config, io_utils
from .features import colunas_atributos, criar_atributos

SEMENTE = 42
N_CONFIGS = 15

GRUPOS_ABLACAO = [
    ("A", "Só os meses recentes", ("y_lag1", "y_lag2", "y_lag3")),
    ("B", "+ sazonalidade", ("y_lag6", "y_lag12", "y_diff12", "mes_sin", "mes_cos")),
    ("C", "+ médias e oscilação", ("y_media", "y_dp", "y_diff1")),
    ("D", "+ variáveis externas", ("int_ansiedade", "leitos_", "pop_lag", "pandemia",
                                   "ano", "unidade_cod")),
]

GRUPO_SHAP = [  # (rótulo, prefixos)
    ("Último mês e recentes", ("y_lag1", "y_lag2", "y_lag3", "y_diff1")),
    ("Mesmo período do ano anterior", ("y_lag6", "y_lag12", "y_diff12")),
    ("Época do ano", ("mes_sin", "mes_cos")),
    ("Tendência (médias)", ("y_media",)),
    ("Oscilação recente", ("y_dp",)),
    ("Ansiedade", ("int_ansiedade",)),
    ("Leitos disponíveis", ("leitos_",)),
    ("População", ("pop_lag",)),
    ("Ano e pandemia", ("ano", "pandemia")),
    ("Perfil da região", ("unidade_cod",)),
]


def lgbm(**kw):
    from lightgbm import LGBMRegressor
    base = dict(n_estimators=600, learning_rate=0.03, num_leaves=15,
                min_child_samples=10, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, random_state=SEMENTE, verbose=-1)
    base.update(kw)
    return LGBMRegressor(**base)


def rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                 max_features=0.5, n_jobs=-1,
                                 random_state=SEMENTE)


def casa(c: str, prefixos) -> bool:
    """Prefixo terminado em dígito é nome exato (y_lag1 não pega y_lag12)."""
    return any(c == p or (c.startswith(p) and not p[-1].isdigit())
               for p in prefixos)


def cols_ate(X: list[str], grupo: str) -> list[str]:
    """Atributos acumulados até o grupo de ablação indicado."""
    prefixos = []
    for g, _, pref in GRUPOS_ABLACAO:
        prefixos += list(pref)
        if g == grupo:
            break
    return [c for c in X if casa(c, prefixos)]


# ------------------------------------------------------------------ ETS
def ets_previsao(serie: np.ndarray) -> float:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    s = np.asarray(serie, dtype=float)
    if s.sum() == 0 or len(s) < 30:
        return float(s[-12:].mean()) if len(s) else 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ExponentialSmoothing(s, trend="add", damped_trend=True,
                                     seasonal="add", seasonal_periods=12,
                                     initialization_method="estimated").fit()
            return float(max(0.0, m.forecast(1)[0]))
    except Exception:  # noqa: BLE001
        return float(s[-12:].mean())


# ------------------------------------------------------------------ tuning
def buscar_hiperparametros(df: pd.DataFrame, X: list[str], meses_val) -> dict:
    rnd = random.Random(SEMENTE)
    espaco = {
        "n_estimators": [300, 600, 1000],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [5, 10, 20, 40],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0],
        "objective": ["regression", "poisson"],
    }
    configs = [{}] + [{k: rnd.choice(v) for k, v in espaco.items()}
                      for _ in range(N_CONFIGS - 1)]
    hist = []
    for i, cfg in enumerate(configs, 1):
        erros = []
        for T in meses_val:
            tr, te = df[df["mes"] < T], df[df["mes"] == T]
            m = lgbm(**cfg).fit(tr[X], tr["y"])
            erros.append(np.abs(np.clip(m.predict(te[X]), 0, None) - te["y"]).mean())
        hist.append({**({"config": "padrão"} if not cfg else cfg),
                     "MAE_validacao_2024": round(float(np.mean(erros)), 3)})
        print(f"    config {i}/{len(configs)}: MAE val = {np.mean(erros):.3f}")
    melhor_i = int(np.argmin([h["MAE_validacao_2024"] for h in hist]))
    return {"melhor": configs[melhor_i], "historico": hist, "melhor_idx": melhor_i}


# ------------------------------------------------------------------ testes
def diebold_mariano(e_a: np.ndarray, e_b: np.ndarray, h: int = 1):
    """DM com perda absoluta e correção HLN. Positivo => A erra mais que B."""
    d = np.abs(e_a) - np.abs(e_b)
    T = len(d)
    dbar = d.mean()
    gamma0 = ((d - dbar) ** 2).mean()
    var = gamma0
    for k in range(1, h):
        var += 2 * ((d[k:] - dbar) * (d[:-k] - dbar)).mean()
    if var <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var / T)
    dm *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * (1 - stats.t.cdf(abs(dm), df=T - 1))
    return float(dm), float(p)


# ------------------------------------------------------------------ SHAP
def explicar_2026(painel: pd.DataFrame, H: int = 6) -> dict:
    import joblib
    from .prever import MODELS_DIR, estender

    meta = json.loads((MODELS_DIR / "meta.json").read_text(encoding="utf-8"))
    X = meta["atributos"]
    ultimo = painel["mes"].max()
    ext = estender(painel, H)
    grupos = [g for g, _ in GRUPO_SHAP]
    idx_grupo = []
    for c in X:
        gi = next((i for i, (_, pref) in enumerate(GRUPO_SHAP)
                   if casa(c, pref)), None)
        idx_grupo.append(gi if gi is not None else len(grupos) - 1)

    saida = {"grupos": grupos, "base": [], "entidades": {}}
    tot = np.zeros((H, len(grupos)))
    for h in range(1, H + 1):
        arq = MODELS_DIR / f"LightGBM_h{h}.joblib"
        if not arq.exists():
            return {}
        m = joblib.load(arq)
        df = criar_atributos(ext, h=h)
        alvo = df[df["mes"] == ultimo + pd.DateOffset(months=h)]
        contrib = m.predict(alvo[X], pred_contrib=True)
        base = float(contrib[0, -1])
        saida["base"].append(round(base, 3))
        for u, linha in zip(alvo["unidade"].astype(str), contrib):
            g = np.zeros(len(grupos))
            for j, gi in enumerate(idx_grupo):
                g[gi] += linha[j]
            ent = saida["entidades"].setdefault(u, [None] * H)
            ent[h - 1] = [round(float(v), 2) for v in g]
            tot[h - 1] += g
        saida.setdefault("n_regioes", int(len(alvo)))
    saida["entidades"]["total"] = [[round(float(v), 1) for v in tot[h]]
                                   for h in range(H)]
    return saida


# ------------------------------------------------------------------ main
def main() -> None:
    t_ini = time.perf_counter()
    painel = io_utils.ler(config.PROCESSED / "painel_regiao.parquet")
    df = criar_atributos(painel)
    X = colunas_atributos(df)
    meses = np.sort(df["mes"].unique())
    teste = meses[-12:]
    val = meses[-24:-12]
    print(f"Teste: {pd.Timestamp(teste[0]):%Y-%m} a {pd.Timestamp(teste[-1]):%Y-%m} "
          f"| validação do ajuste: {pd.Timestamp(val[0]):%Y-%m} a {pd.Timestamp(val[-1]):%Y-%m}")

    print("\n[1/5] Ajuste de hiperparâmetros (só 2024)...")
    tun = buscar_hiperparametros(df, X, val)
    cfg_melhor = tun["melhor"]

    print("\n[2/5] Walk-forward 2025: chutes, ETS, RF, LightGBM, ablação...")
    serie_reg = {u: g.set_index("mes")["y"] for u, g in painel.groupby("unidade")}
    linhas = []
    for i, T in enumerate(teste, 1):
        tr, te = df[df["mes"] < T], df[df["mes"] == T]
        out = te[["unidade", "mes", "y"]].copy()
        out["Ingenuo"] = te["y_lag1"].to_numpy()
        out["Sazonal_ingenuo"] = te["y_lag12"].to_numpy()
        out["ETS"] = [ets_previsao(serie_reg[u][serie_reg[u].index < T].to_numpy())
                      for u in te["unidade"]]
        out["RandomForest"] = np.clip(rf().fit(tr[X], tr["y"]).predict(te[X]), 0, None)
        out["LightGBM"] = np.clip(lgbm().fit(tr[X], tr["y"]).predict(te[X]), 0, None)
        out["LightGBM_ajustado"] = np.clip(
            lgbm(**cfg_melhor).fit(tr[X], tr["y"]).predict(te[X]), 0, None)
        for g, _, _ in GRUPOS_ABLACAO[:-1]:
            c = cols_ate(X, g)
            out[f"Ablacao_{g}"] = np.clip(lgbm().fit(tr[c], tr["y"]).predict(te[c]), 0, None)
        out["Ablacao_D"] = out["LightGBM"]
        # combinação estatística + IA (lição da competição M4)
        out["Combinacao"] = out[["ETS", "RandomForest", "LightGBM"]].mean(axis=1)
        linhas.append(out)
        print(f"    {pd.Timestamp(T):%Y-%m} ok [{i}/12]")
    wf = pd.concat(linhas, ignore_index=True)
    wf.to_csv(config.OUTPUTS / "experimentos_previsoes.csv", index=False)

    principais = ["Ingenuo", "Sazonal_ingenuo", "ETS", "RandomForest",
                  "LightGBM", "LightGBM_ajustado", "Combinacao"]
    tot = wf.groupby("mes")[["y"] + principais + [f"Ablacao_{g}" for g, _, _ in GRUPOS_ABLACAO]].sum()

    def resumo(col):
        e = wf[col] - wf["y"]
        et = tot[col] - tot["y"]
        return {"MAE": round(float(e.abs().mean()), 3),
                "RMSE": round(float(np.sqrt((e ** 2).mean())), 3),
                "erro_total_pct": round(float((et.abs() / tot["y"]).mean() * 100), 2),
                "vies_pct": round(float((et / tot["y"]).mean() * 100), 2)}

    modelos = {m: resumo(m) for m in principais}
    melhor = min(["ETS", "RandomForest", "LightGBM", "LightGBM_ajustado", "Combinacao"],
                 key=lambda m: modelos[m]["MAE"])

    print("\n[3/5] Ablação e significância...")
    ablacao = []
    for g, rot, _ in GRUPOS_ABLACAO:
        r = resumo(f"Ablacao_{g}")
        ablacao.append({"grupo": g, "rotulo": rot,
                        "n_atributos": len(cols_ate(X, g)), **r})

    def comparar(a, b):
        """Testa se o modelo a erra mais que o b."""
        e_a, e_b = (wf[a] - wf["y"]).to_numpy(), (wf[b] - wf["y"]).to_numpy()
        d_a = wf.assign(z=np.abs(e_a)).groupby("mes")["z"].mean().to_numpy()
        d_b = wf.assign(z=np.abs(e_b)).groupby("mes")["z"].mean().to_numpy()
        dm, p = diebold_mariano(d_a, d_b)
        try:
            w = stats.wilcoxon(np.abs(e_a), np.abs(e_b), alternative="greater")
            pw = float(w.pvalue) if np.isfinite(w.pvalue) else 1.0
        except ValueError:
            pw = 1.0
        return {"modelo": a, "contra": b, "dm": round(dm, 3), "p_dm": round(p, 4),
                "p_wilcoxon": round(pw, 4), "significativo_5pct": bool(min(p, pw) < 0.05)}

    sig = [comparar(m, melhor) for m in principais if m != melhor]
    pares = [comparar(a, b) for a in principais for b in principais if a != b]

    print("\n[4/5] Análise de erro...")
    porte = wf.groupby("unidade")["y"].mean()
    faixas = pd.qcut(porte, 3, labels=["Pequenas", "Médias", "Grandes"])
    wf["porte"] = wf["unidade"].map(faixas)
    por_porte = []
    for p, g in wf.groupby("porte", observed=True):
        tt = g.groupby("mes")[["y", melhor, "Ingenuo"]].sum()
        por_porte.append({
            "porte": str(p), "n_regioes": int(g["unidade"].nunique()),
            "media_mensal": round(float(g["y"].mean()), 1),
            "erro_pct_modelo": round(float(((tt[melhor] - tt["y"]).abs() / tt["y"]).mean() * 100), 1),
            "erro_pct_chute": round(float(((tt["Ingenuo"] - tt["y"]).abs() / tt["y"]).mean() * 100), 1)})
    por_mes = [{"mes": pd.Timestamp(m).strftime("%Y-%m"), "real": float(r["y"]),
                **{k: round(float(r[k]), 1) for k in principais}}
               for m, r in tot.iterrows()]

    print("\n[5/5] Explicando as previsões de 2026 (SHAP)...")
    try:
        shap = explicar_2026(painel)
    except Exception as e:  # noqa: BLE001
        print(f"    AVISO: SHAP não gerado ({e})")
        shap = {}

    res = {"teste_meses": [pd.Timestamp(m).strftime("%Y-%m") for m in teste],
           "modelos": modelos, "melhor": melhor, "ablacao": ablacao,
           "tuning": {"melhor_config": cfg_melhor or {"config": "padrão"},
                      "historico": tun["historico"], "n_configs": len(tun["historico"])},
           "significancia": sig, "pares": pares, "erro_por_porte": por_porte,
           "erro_por_mes": por_mes, "shap": shap,
           "duracao_s": round(time.perf_counter() - t_ini)}
    (config.OUTPUTS / "experimentos.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(modelos).T.to_csv(config.OUTPUTS / "experimentos_resumo.csv")

    pd.set_option("display.width", 150)
    print("\n=== Modelos (teste 2025, 1 mês à frente) ===")
    print(pd.DataFrame(modelos).T.to_string())
    print(f"\nMelhor: {melhor}")
    print("\n=== Ablação ===")
    print(pd.DataFrame(ablacao)[["grupo", "rotulo", "n_atributos", "MAE", "erro_total_pct"]].to_string(index=False))
    print("\n=== Significância (vs melhor) ===")
    print(pd.DataFrame(sig).to_string(index=False))
    print("\n=== Erro por porte ===")
    print(pd.DataFrame(por_porte).to_string(index=False))
    print(f"\nConcluído em {res['duracao_s']}s -> outputs/experimentos.json")


if __name__ == "__main__":
    main()
