import time
import numpy as np

np.random.seed(3)

def determinante(A):
    n = len(A)
    M = A.astype(float).copy()
    sinal = 1.0

    for k in range(n):
        p = np.argmax(np.abs(M[k:, k])) + k

        if M[p, k] == 0:
            return 0.0

        if p != k:
            M[[k, p]] = M[[p, k]]
            sinal = -sinal

        fatores = M[k + 1:, k] / M[k, k]
        M[k + 1:] -= np.outer(fatores, M[k])

    return sinal * np.prod(np.diag(M))


def cramer(A, b):
    n = len(b)
    det_A = determinante(A)
    x = np.empty(n)

    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = determinante(Ai) / det_A

    return x


def inversa(A, b):
    n = len(b)
    M = np.hstack([A.astype(float), np.eye(n)])

    for k in range(n):
        p = np.argmax(np.abs(M[k:, k])) + k
        M[[k, p]] = M[[p, k]]
        M[k] /= M[k, k]

        fatores = M[:, k].copy()
        fatores[k] = 0
        M -= np.outer(fatores, M[k])

    A_inv = M[:, n:]
    return A_inv @ b


def gauss(A, b):
    n = len(b)
    M = np.hstack([A, b.reshape(-1, 1)]).astype(float)

    for k in range(n):
        p = np.argmax(np.abs(M[k:, k])) + k
        M[[k, p]] = M[[p, k]]

        fatores = M[k + 1:, k] / M[k, k]
        M[k + 1:] -= np.outer(fatores, M[k])

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]

    return x


def fatorar_lu(A):
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
    n = len(b)
    P, L, U = fatorar_lu(A)
    pb = b[P]

    y = np.zeros(n)
    for i in range(n):
        y[i] = pb[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    return x


METODOS = {
    "Cramer": cramer,
    "Inversa": inversa,
    "Gauss": gauss,
    "LU": lu
}


# PARTE 1: verifica se os quatro métodos encontram a mesma solução.

A3 = np.array([
    [2.0, 1.0, -1.0],
    [-3.0, -1.0, 2.0],
    [-2.0, 1.0, 2.0]
])

b3 = np.array([8.0, -11.0, -3.0])


# PARTE 2: mede o tempo e o erro dos métodos.

def medir(f, A, b, repeticoes=2):
    melhor = 1e9

    for _ in range(repeticoes):
        t0 = time.perf_counter()
        x = f(A, b)
        melhor = min(melhor, time.perf_counter() - t0)

    erro = np.linalg.norm(A @ x - b)
    return melhor, max(erro, 1e-17)


tempos = {m: [] for m in METODOS}
erros = {m: [] for m in METODOS}

TAMANHOS = list(range(5, 155, 10))

for n in TAMANHOS:
    A = np.random.randn(n, n)
    b = np.random.randn(n)

    for nome, f in METODOS.items():
        t, e = medir(f, A, b)

        tempos[nome].append(t)
        erros[nome].append(e)


# Salva os resultados em .pnz
dados = {"TAMANHOS": np.array(TAMANHOS)}

for nome in METODOS:
    dados[f"tempos_{nome}"] = np.array(tempos[nome])
    dados[f"erros_{nome}"] = np.array(erros[nome])

np.savez("resultados.npz", **dados)



# EXIBIÇÃO DOS RESULTADOS NO TERMINAL.
print("PARTE 1 - sistema 3x3 (solução: 2, 3, -1)")

for nome, f in METODOS.items():
    print(f"  {nome:8s} -> {np.round(f(A3, b3), 6)}")

print()

print("PARTE 2 - corrida (tempo em ms | erro residual)")

for i, n in enumerate(TAMANHOS):
    linha = f"  n={n:3d} "

    for nome in METODOS:
        t = tempos[nome][i]
        e = erros[nome][i]

        linha += f"| {nome} {t * 1000:7.2f} ms "

    print(linha)

# Mostra o ranking final.
n_final = TAMANHOS[-1]

print(f"\nRANKING em n = {n_final}:")

for nome in sorted(METODOS, key=lambda m: tempos[m][-1]):
    print(
        f"  {nome:8s} {tempos[nome][-1] * 1000:8.2f} ms"
        f"   erro {erros[nome][-1]:.1e}"
    )