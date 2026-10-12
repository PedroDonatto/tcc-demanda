"""Banco DuckDB: consulta SQL sobre os arquivos Parquet de data/processed.

Uso:
    python -m src.db                       # cria/atualiza as views e lista
    python -m src.db "SELECT * FROM painel_regiao LIMIT 5"

Cada arquivo data/processed/<nome>.parquet vira uma view <nome> no banco
data/tcc.duckdb. Os dados continuam nos Parquet; o banco só guarda as views,
então dá para abrir o tcc.duckdb no DBeaver ou no Python e usar SQL.
"""
from __future__ import annotations

import sys

import duckdb

from . import config


def conectar() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(config.DB_PATH))
    for arq in sorted(config.PROCESSED.glob("*.parquet")):
        caminho = arq.as_posix().replace("'", "''")
        con.execute(f"CREATE OR REPLACE VIEW {arq.stem} AS "
                    f"SELECT * FROM read_parquet('{caminho}')")
    return con


def main() -> None:
    con = conectar()
    if len(sys.argv) > 1:
        print(con.sql(" ".join(sys.argv[1:])))
    else:
        print(con.sql("SELECT view_name FROM duckdb_views() "
                      "WHERE NOT internal ORDER BY 1"))
    con.close()


if __name__ == "__main__":
    main()
