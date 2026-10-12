"""Acesso aos arquivos do DATASUS: download via FTP e leitura de .dbc/.dbf.

- Download: ftplib (biblioteca padrão), com tentativas e retomada simples.
- .dbc -> .dbf: biblioteca `datasus-dbc` (tem versão pronta para Windows).
- .dbf -> DataFrame: leitor próprio com numpy (muito mais rápido que ler
  registro a registro) que extrai só as colunas pedidas.
"""
from __future__ import annotations

import ftplib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# ------------------------------------------------------------------ FTP
def baixar_ftp(diretorio: str, arquivo: str, destino: Path,
               tentativas: int = 4) -> Path | None:
    """Baixa `diretorio/arquivo` do FTP do DATASUS para `destino`.

    Retorna o caminho salvo, ou None se o arquivo não existe no servidor
    (ex.: competência ainda não publicada).
    """
    destino = Path(destino)
    if destino.exists() and destino.stat().st_size > 0:
        return destino  # já baixado

    tmp = destino.with_suffix(destino.suffix + ".part")
    for t in range(1, tentativas + 1):
        try:
            with ftplib.FTP(config.FTP_HOST, timeout=120) as ftp:
                ftp.login()
                ftp.cwd(diretorio)
                try:
                    ftp.size(arquivo)
                except ftplib.error_perm:
                    # Alguns arquivos estão com extensão maiúscula/minúscula
                    nomes = {n.upper(): n for n in ftp.nlst()}
                    if arquivo.upper() not in nomes:
                        return None
                    arquivo = nomes[arquivo.upper()]
                with open(tmp, "wb") as f:
                    ftp.retrbinary(f"RETR {arquivo}", f.write, blocksize=1 << 16)
            tmp.replace(destino)
            return destino
        except (ftplib.all_errors) as e:  # noqa: PERF203
            espera = 5 * t
            print(f"    ! erro no FTP ({e}); tentativa {t}/{tentativas}, "
                  f"aguardando {espera}s")
            time.sleep(espera)
    raise RuntimeError(f"Falha ao baixar {diretorio}/{arquivo}")


# ------------------------------------------------------------------ DBC/DBF
def dbc_para_dbf_bytes(caminho_dbc: Path) -> bytes:
    import datasus_dbc  # import aqui para os testes rodarem sem a lib

    with open(caminho_dbc, "rb") as f:
        return datasus_dbc.decompress_bytes(f.read())


def ler_dbf_bytes(buf: bytes, colunas: list[str] | None = None,
                  encoding: str = "latin-1",
                  ignorar_ausentes: bool = False) -> pd.DataFrame:
    """Lê um DBF (dBase III) a partir de bytes, devolvendo colunas como texto.

    Todas as colunas voltam como `str` (sem espaços nas pontas); converta os
    tipos depois, só nas colunas que interessam.
    """
    if len(buf) < 32:
        raise ValueError("arquivo DBF vazio ou corrompido")
    n_reg = int.from_bytes(buf[4:8], "little")
    tam_header = int.from_bytes(buf[8:10], "little")
    tam_reg = int.from_bytes(buf[10:12], "little")

    # descritores de campo: blocos de 32 bytes até o terminador 0x0D
    campos = []  # (nome, tamanho)
    pos = 32
    while pos < tam_header and buf[pos] != 0x0D:
        nome = buf[pos:pos + 11].split(b"\x00")[0].decode("ascii").strip()
        tamanho = buf[pos + 16]
        campos.append((nome, tamanho))
        pos += 32

    soma = 1 + sum(t for _, t in campos)
    if soma != tam_reg:
        raise ValueError(f"DBF inconsistente: registro {tam_reg} != campos {soma}")

    dtype = np.dtype([("_del", "S1")] + [(n, f"S{t}") for n, t in campos])
    # o número de registros do cabeçalho às vezes vem errado; usamos o menor
    disponivel = (len(buf) - tam_header) // tam_reg
    n = min(n_reg, disponivel) if n_reg else disponivel
    arr = np.frombuffer(buf, dtype=dtype, count=n, offset=tam_header)
    arr = arr[arr["_del"] != b"*"]  # remove registros apagados

    nomes = [n for n, _ in campos]
    if colunas is None:
        colunas = nomes
    faltando = [c for c in colunas if c not in nomes]
    if faltando and ignorar_ausentes:
        colunas = [c for c in colunas if c in nomes]
    elif faltando:
        raise KeyError(f"colunas ausentes no DBF: {faltando}. Disponíveis: {nomes}")

    dados = {}
    for c in colunas:
        s = pd.Series(arr[c])
        dados[c] = s.str.decode(encoding).str.strip()
    return pd.DataFrame(dados)


def ler_dbc(caminho_dbc: Path, colunas: list[str] | None = None,
            ignorar_ausentes: bool = True) -> pd.DataFrame:
    return ler_dbf_bytes(dbc_para_dbf_bytes(caminho_dbc), colunas,
                         ignorar_ausentes=ignorar_ausentes)


def competencias(ano_ini: int, mes_ini: int, ano_fim: int, mes_fim: int):
    """Gera (ano, mes) de ano_ini/mes_ini até ano_fim/mes_fim, inclusive."""
    a, m = ano_ini, mes_ini
    while (a, m) <= (ano_fim, mes_fim):
        yield a, m
        m += 1
        if m == 13:
            a, m = a + 1, 1
