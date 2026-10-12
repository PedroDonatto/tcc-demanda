"""Testes com dados sintéticos (não precisa de internet).

Uso:  python -m tests.test_pipeline
"""
import struct

import numpy as np
import pandas as pd

from src import config, datasus, ingest_cnes, ingest_sih
from src.build_dataset import montar_painel
from src.features import colunas_atributos, criar_atributos
from src.train import tabela_metricas, walk_forward


def escrever_dbf(df: pd.DataFrame) -> bytes:
    """Gera um DBF dBase III simples (campos C) para testar o leitor."""
    campos = [(c, max(1, int(df[c].astype(str).str.len().max()))) for c in df]
    tam_header = 32 + 32 * len(campos) + 1
    tam_reg = 1 + sum(t for _, t in campos)
    h = struct.pack("<BBBBIHH20x", 3, 126, 1, 1, len(df), tam_header, tam_reg)
    for nome, t in campos:
        h += struct.pack("<11sc4xBB14x", nome.encode(), b"C", t, 0)
    h += b"\x0d"
    corpo = b""
    for i, (_, row) in enumerate(df.iterrows()):
        corpo += b"*" if i == 1 else b" "  # 2º registro marcado como apagado
        for nome, t in campos:
            corpo += str(row[nome]).ljust(t).encode("latin-1")
    return h + corpo + b"\x1a"


def test_leitor_dbf():
    df = pd.DataFrame({"A": ["1", "2", "3"], "NOME": ["São", "x", "Jundiaí"]})
    out = datasus.ler_dbf_bytes(escrever_dbf(df))
    assert out["A"].tolist() == ["1", "3"], out  # apagado removido
    assert out["NOME"].tolist() == ["São", "Jundiaí"]
    out2 = datasus.ler_dbf_bytes(escrever_dbf(df), ["NOME", "XX"],
                                 ignorar_ausentes=True)
    assert list(out2.columns) == ["NOME"]
    print("ok leitor DBF")


def test_sih():
    rd = pd.DataFrame({
        "N_AIH": ["1", "2", "3", "4", "5", "2"],
        "IDENT": ["1", "1", "5", "1", "1", "1"],
        "MUNIC_RES": ["352590"] * 6,
        "DIAG_PRINC": ["F320", "F412", "F331", "J189", "F399", "F412"],
        "DT_INTER": ["20240105", "20240110", "20240101", "20240101",
                     "20231231", "20240110"],
        "DIAS_PERM": ["10"] * 6, "VAL_TOT": ["100.5"] * 6,
        "CAR_INT": ["02", "01", "02", "02", "02", "01"],
    })
    f = ingest_sih.filtrar_rd(rd)
    assert set(f["N_AIH"]) == {"1", "2", "3", "5"}  # J189 fora
    config.ANO_INI = 2024
    aih = ingest_sih.consolidar([f])
    # 3 = continuação (IDENT 5); 5 = internado em 2023; "2" duplicada
    assert sorted(aih["N_AIH"]) == ["1", "2"], aih["N_AIH"].tolist()
    m = ingest_sih.agregar_mensal(aih)
    assert m.set_index("grupo")["internacoes"].to_dict() == {
        "humor": 1, "ansiedade": 1}
    print("ok filtros SIH")


def dados_sinteticos():
    rng = np.random.default_rng(0)
    mun = pd.DataFrame({
        "munic": [f"35{i:04d}" for i in range(6)],
        "reg_imediata_id": ["A", "A", "B", "B", "C", "C"],
    })
    meses = pd.date_range("2018-01-01", "2023-12-01", freq="MS")
    linhas = []
    for mu in mun["munic"]:
        base = rng.uniform(5, 20)
        for t, mes in enumerate(meses):
            saz = 1 + 0.3 * np.sin(2 * np.pi * mes.month / 12)
            for g, fat in (("humor", 1.0), ("ansiedade", 0.4)):
                lam = base * fat * saz * (1 + 0.01 * t)
                linhas.append({"munic_res": mu, "mes": mes, "grupo": g,
                               "internacoes": rng.poisson(lam),
                               "dias_perm": 0, "val_tot": 0, "urgencias": 1})
    sih = pd.DataFrame(linhas)
    cnes = pd.DataFrame([{"munic": mu, "mes": mes, "leitos_sm_sus": 10,
                          "leitos_tot_sus": 100}
                         for mu in mun["munic"] for mes in meses])
    pop = pd.DataFrame([{"munic": mu, "ano": a, "pop": 50000}
                        for mu in mun["munic"] for a in range(2017, 2024)])
    return sih, cnes, mun, pop


def test_painel_features_treino():
    config.ANO_INI, config.ANO_FIM = 2018, 2023
    sih, cnes, mun, pop = dados_sinteticos()
    painel = montar_painel(sih, cnes, mun, pop, "regiao")
    assert painel["unidade"].nunique() == 3
    assert len(painel) == 3 * 72
    tot = sih[sih.grupo == "humor"]["internacoes"].sum()
    assert painel["y"].sum() == tot

    est = montar_painel(sih, cnes, mun, pop, "estado")
    assert est["unidade"].nunique() == 1 and est["y"].sum() == tot

    # --- vazamento: alterar o FUTURO não pode mudar atributos do passado
    df1 = criar_atributos(painel)
    corte = pd.Timestamp("2022-06-01")
    p2 = painel.copy()
    p2.loc[p2["mes"] >= corte, ["y", "int_ansiedade"]] *= 100
    df2 = criar_atributos(p2)
    cols = colunas_atributos(df1)
    a = df1[df1["mes"] <= corte].set_index(["unidade", "mes"])[cols]
    b = df2[df2["mes"] <= corte].set_index(["unidade", "mes"])[cols]
    pd.testing.assert_frame_equal(a, b)
    assert "y" not in cols and "int_humor" not in cols
    print(f"ok sem vazamento ({len(cols)} atributos)")

    prev, custos, _ = walk_forward(df1, n_teste=6)
    met = tabela_metricas(prev, custos)
    print(met.round(3).to_string(index=False))
    assert {"Ingenuo", "Sazonal_ingenuo", "RandomForest"} <= set(met["modelo"])
    from src.train import grafico
    grafico(prev, df1, "teste_sintetico")
    assert (config.OUTPUTS / "grafico_teste_sintetico.png").exists()
    (config.OUTPUTS / "grafico_teste_sintetico.png").unlink()
    print("ok treino walk-forward + gráfico")


def test_cnes():
    lt = pd.DataFrame({"CODUFMUN": ["352590", "352590", "355030"],
                       "CODLEITO": ["47", "33", "87"],
                       "QT_EXIST": ["10", "50", "5"], "QT_SUS": ["8", "40", "5"]})
    ag = ingest_cnes.agregar_lt(lt, 2024, 1)
    p = ingest_cnes.montar_painel(ag).set_index("munic")
    assert p.loc["352590", "leitos_sm_sus"] == 8
    assert p.loc["352590", "leitos_tot_sus"] == 48
    assert p.loc["355030", "leitos_sm_exist"] == 5
    print("ok CNES")


if __name__ == "__main__":
    test_leitor_dbf()
    test_sih()
    test_cnes()
    test_painel_features_treino()
    print("\nTODOS OS TESTES PASSARAM")
