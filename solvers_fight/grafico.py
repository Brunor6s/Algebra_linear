import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# PARTE 3: carrega os resultados produzidos por solvers.py.
dados = np.load("resultados.npz")
TAMANHOS = dados["TAMANHOS"]

METODOS = ["Cramer", "Inversa", "Gauss", "LU"]
CORES = {
    "Cramer": "#ff5a6e",
    "Inversa": "#1fa3ff",
    "Gauss": "#3ddc97",
    "LU": "#ffb02e"
}

tempos = {nome: dados[f"tempos_{nome}"] for nome in METODOS}
erros = {nome: dados[f"erros_{nome}"] for nome in METODOS}


# Configuração da aparência dos gráficos.
FUNDO = "#0b1220"
PAINEL = "#0f1a2e"
TEXTO = "#aab6d3"

fig, (ax_t, ax_e) = plt.subplots(
    1, 2,
    figsize=(11, 5.2),
    facecolor=FUNDO
)

fig.subplots_adjust(
    top=0.70,
    bottom=0.12,
    left=0.07,
    right=0.97,
    wspace=0.22
)

fig.text(
    0.04, 0.93, "LINEAR ALGEBRA",
    color="#8fb4ff", fontsize=9, family="monospace"
)
fig.text(
    0.04, 0.85, "Solvers Fight",
    color="white", fontsize=26, weight="bold"
)
fig.text(
    0.04, 0.80,
    "Cramer  vs  Inversa  vs  Gauss  vs  LU   |   resolvendo A x = b",
    color=TEXTO, fontsize=10
)

texto_n = fig.text(
    0.97, 0.93, "",
    color="#8fb4ff", fontsize=10,
    ha="right", family="monospace"
)

linhas_t = {}
linhas_e = {}

for ax, titulo, ylabel in [
    (ax_t, "Tempo de resolução", "tempo (s)  [escala log]"),
    (ax_e, "Precisão (erro ||Ax − b||)", "erro  [escala log]")
]:
    ax.set_facecolor(PAINEL)
    ax.set_yscale("log")
    ax.set_xlim(0, TAMANHOS[-1] + 5)
    ax.set_title(titulo, color="white", fontsize=11, loc="left")
    ax.set_xlabel(
        "tamanho do sistema (n incógnitas)",
        color=TEXTO, fontsize=9
    )
    ax.set_ylabel(ylabel, color=TEXTO, fontsize=9)
    ax.tick_params(colors=TEXTO, labelsize=8)
    ax.grid(True, color="#22304d", lw=0.6)

    for s in ax.spines.values():
        s.set_color("#22304d")


# Define os limites verticais com base nos dados medidos.
todos_t = [v for nome in METODOS for v in tempos[nome]]
todos_e = [v for nome in METODOS for v in erros[nome]]

ax_t.set_ylim(min(todos_t) / 2, max(todos_t) * 3)
ax_e.set_ylim(min(todos_e) / 10, max(todos_e) * 10)


# Cria as linhas que serão atualizadas durante a animação.
for nome in METODOS:
    cor = CORES[nome]

    linhas_t[nome], = ax_t.plot(
        [], [], color=cor, lw=2.4,
        marker="o", ms=3.5, label=nome
    )
    linhas_e[nome], = ax_e.plot(
        [], [], color=cor, lw=2.4,
        marker="o", ms=3.5, label=nome
    )

ax_t.legend(
    facecolor=PAINEL,
    edgecolor="#22304d",
    labelcolor="white",
    fontsize=9,
    loc="upper left"
)


def atualizar(i):
    i = min(i, len(TAMANHOS) - 1)
    xs = TAMANHOS[:i + 1]

    for nome in METODOS:
        linhas_t[nome].set_data(xs, tempos[nome][:i + 1])
        linhas_e[nome].set_data(xs, erros[nome][:i + 1])

    texto_n.set_text(f"n = {TAMANHOS[i]}")
    return []


# Salva uma imagem PNG com o resultado completo.
atualizar(len(TAMANHOS) - 1)
fig.savefig("solvers_fight.png", dpi=130, facecolor=FUNDO)


# Prepara os gráficos vazios para começar a animação do primeiro tamanho.
def iniciar():
    for nome in METODOS:
        linhas_t[nome].set_data([], [])
        linhas_e[nome].set_data([], [])
    texto_n.set_text("")
    return list(linhas_t.values()) + list(linhas_e.values()) + [texto_n]


# Exibe a animação em uma janela, sem criar um arquivo GIF.
frames = list(range(len(TAMANHOS))) + [len(TAMANHOS) - 1] * 15

animacao = FuncAnimation(
    fig,
    atualizar,
    frames=frames,
    init_func=iniciar,
    interval=120,
    blit=False,
    repeat=False
)

print("Imagem estática salva: solvers_fight.png")
print("A animação será exibida em uma janela. Feche a janela para encerrar.")
plt.show()
