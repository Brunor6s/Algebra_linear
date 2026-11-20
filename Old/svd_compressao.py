"""
Compressão de imagem com SVD (Decomposição em Valores Singulares)

Ideia: toda matriz A (m x n) pode ser escrita como A = U @ diag(S) @ Vt.
Se guardarmos só os k maiores valores singulares, obtemos uma aproximação A_k
que usa bem menos dados e ainda parece com a original.

Uso:
    python svd_compressao.py minha_foto.jpg
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------
# 1) CONFIGURAÇÃO
# ---------------------------------------------------------------
CAMINHO = sys.argv[1] if len(sys.argv) > 1 else "alisson.jpg"
VALORES_K = [2, 5, 20, 50, 100]  # quantos valores singulares manter


# ---------------------------------------------------------------
# 2) FUNÇÕES
# ---------------------------------------------------------------
def carregar_imagem(caminho):
    """Abre a imagem em tons de cinza e devolve uma matriz numpy (m x n)."""
    img = Image.open(caminho).convert("L")  # "L" = escala de cinza
    return np.array(img, dtype=float)


def reconstruir(U, S, Vt, k):
    """Monta A_k usando apenas os k primeiros valores singulares."""
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def tamanho_comprimido(m, n, k):
    """Quantos números precisamos guardar: U (m*k) + S (k) + Vt (k*n)."""
    return k * (m + n + 1)


def energia_retida(S, k):
    """Fração da 'informação' mantida (soma dos quadrados dos valores singulares)."""
    return np.sum(S[:k] ** 2) / np.sum(S ** 2)


# ---------------------------------------------------------------
# 3) PROGRAMA PRINCIPAL
# ---------------------------------------------------------------
A = carregar_imagem(CAMINHO)
m, n = A.shape
print(f"Imagem: {m} x {n} pixels  ->  {m * n} números\n")

# A decomposição em si: uma única linha
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Tabela de resultados
print(f"{'k':>5} {'números guardados':>20} {'economia':>10} {'energia':>9}")
for k in VALORES_K:
    guardado = tamanho_comprimido(m, n, k)
    economia = 1 - guardado / (m * n)
    print(f"{k:>5} {guardado:>20} {economia:>10.1%} {energia_retida(S, k):>9.1%}")

# ---------------------------------------------------------------
# 4) GRÁFICO 1: original vs. aproximações
# ---------------------------------------------------------------
fig, eixos = plt.subplots(1, len(VALORES_K) + 1, figsize=(3 * (len(VALORES_K) + 1), 3.5))

eixos[0].imshow(A, cmap="gray")
eixos[0].set_title("Original")

for eixo, k in zip(eixos[1:], VALORES_K):
    Ak = reconstruir(U, S, Vt, k)
    eixo.imshow(Ak, cmap="gray", vmin=0, vmax=255)
    eixo.set_title(f"k = {k}\n{1 - tamanho_comprimido(m, n, k) / (m * n):.0%} menos dados")

for eixo in eixos:
    eixo.axis("off")
plt.tight_layout()
plt.savefig("comparacao_svd.png", dpi=150)

# ---------------------------------------------------------------
# 5) GRÁFICO 2: por que funciona? (valores singulares caem rápido)
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.semilogy(S)
plt.xlabel("índice i")
plt.ylabel("valor singular σᵢ (escala log)")
plt.title("Poucos valores singulares concentram quase toda a informação")
plt.grid(True)
plt.tight_layout()
plt.savefig("valores_singulares.png", dpi=150)

plt.show()
