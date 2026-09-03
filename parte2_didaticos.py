"""
parte2_didaticos.py

Exercicio 2.2 - Previsao x realidade na biblioteca de bisseccao.

Funcao-teste padrao da Parte 2: f(x) = x^3 - 9x + 3, raiz em (0,1).
"""

import math
import pandas as pd

from metodos import bisseccao


def previsao_bisseccao(a0, b0, eps):
    """Numero minimo de iteracoes previsto pela teoria da bisseccao.

    A cada iteracao o intervalo eh dividido ao meio, entao apos n
    iteracoes o erro maximo eh (b0-a0)/2**n. Queremos (b0-a0)/2**n < eps,
    ou seja n > log2((b0-a0)/eps).
    """
    return math.ceil((math.log(b0 - a0) - math.log(eps)) / math.log(2))


def exercicio_2_2():
    f = lambda x: x**3 - 9 * x + 3
    a0, b0 = 0.0, 1.0
    epsilons = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

    linhas = []
    for eps in epsilons:
        previsto = previsao_bisseccao(a0, b0, eps)
        raiz, hist = bisseccao(f, a0, b0, eps=eps)
        # historico[k] eh 0-indexado -> o numero de iteracoes efetivamente
        # executadas eh (indice k do ultimo elemento) + 1
        real = hist[-1]["k"] + 1
        linhas.append({
            "eps": eps,
            "previsto": previsto,
            "real": real,
            "raiz": raiz,
        })

    df = pd.DataFrame(linhas)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    exercicio_2_2()
