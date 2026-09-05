"""
parte2_didaticos.py

Exercicio 2.2 - Previsao x realidade na biblioteca de bisseccao.

Funcao-teste padrao da Parte 2: f(x) = x^3 - 9x + 3, raiz em (0,1).
"""

import math
import pandas as pd
import numpy as np

from metodos import bisseccao, newton, secante, newton


def previsao_bisseccao(a0, b0, eps):
    """Numero minimo de iteracoes previsto pela teoria da bisseccao.

    A cada iteracao o intervalo eh dividido ao meio, entao apos n
    iteracoes o erro maximo eh (b0-a0)/2**n. Queremos (b0-a0)/2**n < eps,
    ou seja n > log2((b0-a0)/eps).
    """
    return math.ceil((math.log(b0 - a0) - math.log(eps)) / math.log(2))

def tabelar_sinais(f, a, b, n):
    """
    Avalia f em n pontos igualmente espaçados em [a,b]
    e retorna os intervalos onda há mudança de sinal

    Caso um ponto da malha seja praticamente uma raiz,
    retorna o intervalo degenerado (x, x).
    """

    if n < 2:
        raise ValueError("n deve ser pelo menos 2")
    
    pontos = np.linspace(a, b, n)
    valores = [f(x) for x in pontos]
    intervalos = []
    tol_zero = 1e-12 #Usada para detectar raiz que caiu exatamente na malha

    for i in range(n):
        if abs(valores[i]) < tol_zero:
            intervalos.append(
                (float(pontos[i]), float(pontos[i]))
            )

    for i in range(n - 1):
        f_esquerda = valores[i]
        f_direita = valores[i + 1]

        # Se um dos extremos já foi identificado como raiz
        if (
            abs(f_esquerda) < tol_zero
            or abs(f_direita) < tol_zero
        ):
            continue

        # Sinais diferentes??
        if(f_esquerda > 0) != (f_direita > 0):
            intervalos.append(
                (float(pontos[i]), float(pontos[i + 1]))
            )
    intervalos.sort(key=lambda intervalo: intervalo[0])

    return intervalos

def exercicio_2_1():
    print("Ex 2.1 a)")
    def f(x):
        return x**3 - 9*x + 3

    for n in [21, 11, 6, 4]:
        intervalos = tabelar_sinais(f, -5, 5, n)

        print(f"n = {n}")
        print("intervalos:", intervalos)
        print("quantidade:", len(intervalos))
        print()

    print("Ex 2.1 b)")
    def g(x):
        return (x - 1.05) * (x - 1.15) * (x - 3)

    for n in [9, 17, 41, 401]:
        intervalos = tabelar_sinais(g, 0, 4, n)

        print(f"n = {n}")
        print("intervalos:", intervalos)
        print("quantidade:", len(intervalos))
        print()

    # Ex 2.1 c)
    """
    O tabelamento grosseiro falha para \(g(x)\) porque as raízes \(1.05\) e \(1.15\) estão muito próximas 
    e podem ficar contidas dentro do mesmo subintervalo da malha. Como a função cruza o eixo duas vezes 
    nesse intervalo, ela retorna ao sinal original e os valores nos extremos apresentam o mesmo sinal. 
    Assim, a ausência de mudança de sinal não implica ausência de raízes. Para \(f(x)=x^3-9x+3\), nas quatro 
    malhas utilizadas as raízes ficam distribuídas em subintervalos distintos, de modo que cada uma produz 
    uma mudança de sinal detectável. Se a localização das raízes fosse desconhecida, uma estratégia adequada 
    seria começar com uma malha ampla e refiná-la progressivamente, principalmente nas regiões suspeitas, eventualmente 
    combinando o tabelamento com a inspeção gráfica."""

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



def exercicio_2_3():
    """Custo real: avaliacoes de funcao.

    Encontra a raiz em (0,1) com eps=1e-8 pelos tres metodos e compara
    iteracoes x avaliacoes de f (e de f' no caso do Newton).

    Chutes iniciais: bisseccao usa o intervalo [0,1] (o mesmo do
    isolamento feito em 2.1). Newton parte do ponto medio x0=0.5 do
    mesmo intervalo. A secante parte dos dois extremos x0=0, x1=1,
    para manter a comparacao justa (mesma informacao de partida que a
    bisseccao).
    """
    f = lambda x: x**3 - 9 * x + 3
    df = lambda x: 3 * x**2 - 9
    eps = 1e-8

    raiz_b, hist_b = bisseccao(f, 0, 1, eps=eps)
    raiz_n, hist_n = newton(f, df, 0.5, eps=eps)
    raiz_s, hist_s = secante(f, 0, 1, eps=eps)

    linhas = [
        {
            "metodo": "Bisseccao",
            "iteracoes": hist_b[-1]["k"] + 1,
            "avaliacoes_f": hist_b[-1]["avaliacoes_f"],
            "avaliacoes_df": 0,
            "raiz": raiz_b,
        },
        {
            "metodo": "Newton",
            "iteracoes": hist_n[-1]["k"] + 1,
            "avaliacoes_f": hist_n[-1]["avaliacoes_f"],
            "avaliacoes_df": hist_n[-1]["avaliacoes_df"],
            "raiz": raiz_n,
        },
        {
            "metodo": "Secante",
            "iteracoes": hist_s[-1]["k"] + 1,
            "avaliacoes_f": hist_s[-1]["avaliacoes_f"],
            "avaliacoes_df": 0,
            "raiz": raiz_s,
        },
    ]

    df_tab = pd.DataFrame(linhas)
    print(df_tab.to_string(index=False))
    return df_tab


if __name__ == "__main__":
    exercicio_2_1()
    print()
    exercicio_2_2()
    print()
    exercicio_2_3()
