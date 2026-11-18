"""
Solvers Fight - 4 métodos de álgebra linear resolvendo A x = b.

Métodos (todos vistos na disciplina):
  1) Regra de Cramer          x_i = det(A_i) / det(A)
  2) Matriz inversa           x = A^-1 b
  3) Eliminação de Gauss      escalonamento + substituição de volta
  4) Decomposição LU          A = L U  ->  L y = b  ->  U x = y

O que o script faz:
  PARTE 1: confere que os 4 métodos dão a MESMA resposta num sistema 3x3.
  PARTE 2: aumenta o tamanho n do sistema e mede TEMPO e ERRO de cada método.
  PARTE 3: gera um GIF animado ("corrida") e um gráfico final.

Uso:
    python solvers_fight.py
"""

import time
import numpy as np
import matplotlib

matplotlib.use("Agg")  # troque por "TkAgg" para ver a janela ao vivo
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(3)

# ---------------------------------------------------------------
# 1) OS 4 MÉTODOS  (cada um recebe A, b e devolve x)
# ---------------------------------------------------------------
# OBS: os 4 métodos são escritos "à mão" no mesmo estilo (escalonamento com
# operações por linha). Assim a comparação é justa: mede o custo do ALGORITMO,
# e não de quem usa código compilado pronto (np.linalg.inv / np.linalg.det).

def determinante(A):
    """Determinante por escalonamento: produto dos pivôs (com troca de sinal)."""
    n = len(A)
    M = A.astype(float).copy()
    sinal = 1.0
    for k in range(n):
        p = np.argmax(np.abs(M[k:, k])) + k
        if M[p, k] == 0:
            return 0.0
        if p != k:
            M[[k, p]] = M[[p, k]]
            sinal = -sinal                           # trocar linhas muda o sinal
        fatores = M[k + 1:, k] / M[k, k]
        M[k + 1:] -= np.outer(fatores, M[k])
    return sinal * np.prod(np.diag(M))


def cramer(A, b):
    """x_i = det(A com a coluna i trocada por b) / det(A)  -> n+1 determinantes."""
    n = len(b)
    det_A = determinante(A)
    x = np.empty(n)
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b            # troca a coluna i por b
        x[i] = determinante(Ai) / det_A
    return x


def inversa(A, b):
    """Calcula A^-1 por Gauss-Jordan em [A | I] e depois x = A^-1 b."""
    n = len(b)
    M = np.hstack([A.astype(float), np.eye(n)])
    for k in range(n):
        p = np.argmax(np.abs(M[k:, k])) + k
        M[[k, p]] = M[[p, k]]
        M[k] /= M[k, k]                              # pivô vira 1
        fatores = M[:, k].copy()
        fatores[k] = 0
        M -= np.outer(fatores, M[k])                 # zera acima E abaixo do pivô
    A_inv = M[:, n:]
    return A_inv @ b


def gauss(A, b):
    """Escalona a matriz aumentada [A | b] e resolve de baixo para cima."""
    n = len(b)
    M = np.hstack([A, b.reshape(-1, 1)]).astype(float)
    for k in range(n):
        p = np.argmax(np.abs(M[k:, k])) + k       # pivô: maior valor da coluna
        M[[k, p]] = M[[p, k]]                      # troca de linhas
        fatores = M[k + 1:, k] / M[k, k]
        M[k + 1:] -= np.outer(fatores, M[k])       # zera abaixo do pivô
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):                 # substituição de volta
        x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x


def fatorar_lu(A):
    """Devolve P, L, U tais que A[P] = L @ U (LU com troca de linhas)."""
    n = len(A)
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.arange(n)
    for k in range(n):
        p = np.argmax(np.abs(U[k:, k])) + k
        U[[k, p]] = U[[p, k]]
        P[[k, p]] = P[[p, k]]
        L[[k, p], :k] = L[[p, k], :k]
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:] -= np.outer(L[k + 1:, k], U[k])
    return P, L, U


def lu(A, b):
    """Resolve L y = b (para frente) e depois U x = y (para trás)."""
    n = len(b)
    P, L, U = fatorar_lu(A)
    pb = b[P]
    y = np.zeros(n)
    for i in range(n):                              # substituição para frente
        y[i] = pb[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):                  # substituição para trás
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


METODOS = {"Cramer": cramer, "Inversa": inversa, "Gauss": gauss, "LU": lu}
CORES = {"Cramer": "#ff5a6e", "Inversa": "#1fa3ff", "Gauss": "#3ddc97", "LU": "#ffb02e"}

# ---------------------------------------------------------------
# 2) PARTE 1: todos dão a mesma resposta?
# ---------------------------------------------------------------
A3 = np.array([[2.0, 1.0, -1.0],
               [-3.0, -1.0, 2.0],
               [-2.0, 1.0, 2.0]])
b3 = np.array([8.0, -11.0, -3.0])        # solução exata: x = (2, 3, -1)

print("PARTE 1 - sistema 3x3 (solução exata: 2, 3, -1)")
for nome, f in METODOS.items():
    print(f"  {nome:8s} -> {np.round(f(A3, b3), 6)}")
print()

# ---------------------------------------------------------------
# 3) PARTE 2: corrida - tempo e erro para n crescente
# ---------------------------------------------------------------
TAMANHOS = list(range(5, 155, 10))        # n = 5, 15, ..., 145


def medir(f, A, b, repeticoes=2):
    """Menor tempo em `repeticoes` execuções + erro ||A x - b||."""
    melhor = 1e9
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        x = f(A, b)
        melhor = min(melhor, time.perf_counter() - t0)
    erro = np.linalg.norm(A @ x - b)
    return melhor, max(erro, 1e-17)


tempos = {m: [] for m in METODOS}
erros = {m: [] for m in METODOS}
print("PARTE 2 - corrida (tempo em ms | erro residual)")
for n in TAMANHOS:
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    linha = f"  n={n:3d} "
    for nome, f in METODOS.items():
        t, e = medir(f, A, b)
        tempos[nome].append(t)
        erros[nome].append(e)
        linha += f"| {nome} {t * 1000:7.2f} ms "
    print(linha)

# ---------------------------------------------------------------
# 4) PARTE 3: gráfico animado (estilo "fight")
# ---------------------------------------------------------------
FUNDO, PAINEL, TEXTO = "#0b1220", "#0f1a2e", "#aab6d3"
fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=FUNDO)
fig.subplots_adjust(top=0.70, bottom=0.12, left=0.07, right=0.97, wspace=0.22)

fig.text(0.04, 0.93, "LINEAR ALGEBRA", color="#8fb4ff", fontsize=9, family="monospace")
fig.text(0.04, 0.85, "Solvers Fight", color="white", fontsize=26, weight="bold")
fig.text(0.04, 0.80, "Cramer  vs  Inversa  vs  Gauss  vs  LU   |   resolvendo A x = b",
         color=TEXTO, fontsize=10)
txt_n = fig.text(0.97, 0.93, "", color="#8fb4ff", fontsize=10, ha="right", family="monospace")

linhas_t, linhas_e = {}, {}
for ax, titulo, ylabel in [(ax_t, "Tempo de resolução", "tempo (s)  [escala log]"),
                           (ax_e, "Precisão (erro ||Ax − b||)", "erro  [escala log]")]:
    ax.set_facecolor(PAINEL)
    ax.set_yscale("log")
    ax.set_xlim(0, TAMANHOS[-1] + 5)
    ax.set_title(titulo, color="white", fontsize=11, loc="left")
    ax.set_xlabel("tamanho do sistema (n incógnitas)", color=TEXTO, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXTO, fontsize=9)
    ax.tick_params(colors=TEXTO, labelsize=8)
    ax.grid(True, color="#22304d", lw=0.6)
    for s in ax.spines.values():
        s.set_color("#22304d")

todos_t = [v for m in tempos for v in tempos[m]]
todos_e = [v for m in erros for v in erros[m]]
ax_t.set_ylim(min(todos_t) / 2, max(todos_t) * 3)
ax_e.set_ylim(min(todos_e) / 10, max(todos_e) * 10)

for nome, cor in CORES.items():
    linhas_t[nome], = ax_t.plot([], [], color=cor, lw=2.4, marker="o", ms=3.5, label=nome)
    linhas_e[nome], = ax_e.plot([], [], color=cor, lw=2.4, marker="o", ms=3.5, label=nome)
ax_t.legend(facecolor=PAINEL, edgecolor="#22304d", labelcolor="white", fontsize=9, loc="upper left")


def atualizar(i):
    i = min(i, len(TAMANHOS) - 1)
    xs = TAMANHOS[: i + 1]
    for nome in METODOS:
        linhas_t[nome].set_data(xs, tempos[nome][: i + 1])
        linhas_e[nome].set_data(xs, erros[nome][: i + 1])
    txt_n.set_text(f"n = {TAMANHOS[i]}")
    return []


frames = list(range(len(TAMANHOS))) + [len(TAMANHOS) - 1] * 15   # segura no final
anim = FuncAnimation(fig, atualizar, frames=frames, interval=120, blit=False)
anim.save("solvers_fight.gif", writer=PillowWriter(fps=8), dpi=80)
atualizar(len(TAMANHOS) - 1)
fig.savefig("solvers_fight.png", dpi=130, facecolor=FUNDO)

# ranking final
n_final = TAMANHOS[-1]
print(f"\nRANKING em n = {n_final} (mais rápido primeiro):")
for nome in sorted(METODOS, key=lambda m: tempos[m][-1]):
    print(f"  {nome:8s} {tempos[nome][-1] * 1000:8.2f} ms   erro {erros[nome][-1]:.1e}")
print("\nArquivos gerados: solvers_fight.gif e solvers_fight.png")