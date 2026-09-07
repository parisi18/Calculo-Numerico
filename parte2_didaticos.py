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

    print('Ex 2.4')
    f = lambda x: x**3 -9*x + 3
    df = lambda x: 3*x**2 - 9
    eps = 1e-8
    a = 0
    b = 1
    qsi = 0.3376089559658377

    print('Newton')
    
    raiz_newton, hist_newton = newton(f, df, a, eps)

    pk_newton = []
    e_newton = []

    for i in range(len(hist_newton)):
        e_newton.append(abs(hist_newton[i]['x'] - qsi))

    for i in range(1, len(e_newton) - 1):
        pk_newton.append(math.log(e_newton[i+1]/e_newton[i], math.e)/math.log(e_newton[i]/e_newton[i-1], math.e))

    print()
    print(f'erros: {e_newton}')
    print(f'p valores: {pk_newton}')

    print('\nSecante')
    
    raiz_secante, hist_secante = secante(f, a, b, eps)

    pk_secante = []
    e_secante = []

    for i in range(len(hist_secante)):
        e_secante.append(abs(hist_secante[i]['x'] - qsi))

    for i in range(1, len(e_secante) - 1):
        pk_secante.append(math.log(e_secante[i+1]/e_secante[i], math.e)/math.log(e_secante[i]/e_secante[i-1], math.e))

    print()
    print(f'erros: {e_secante}')
    print(f'p valores: {pk_secante}')

    print("""
No Método de Newton, obteve-se pk aproximadamente 2, confirmando a convergência quadrática teórica. 
No Método da Secante, a sequência de valores pk estabiliza-se em torno de 1.73, aproximando-se do valor teórico esperado de 
aproximadamente 1.618.
""")

    print('Ex 2.5 a)')
    f = lambda x: x**3 -2*x + 2
    df = lambda x: 3*x**2 - 2
    x0 = 0
    
    raiz_newton, hist_newton = newton(f, df, x0, eps, max_iter=10)

    print('Não, conforme apontado pelo waring ele não converge.')
    print("""
A ausência de convergência ocorre porque o método entra em um ciclo infinito de oscilação (ciclo limite de período 2), alternando 
indefinidamente entre os valores 0 e 1. 

Na primeira iteração (x0 = 0): A derivada é negativa (f'(0) = -2), indicando que a reta tangente é decrescente. 
Ao traçar essa tangente a partir do ponto (0, f(0)), ela intercepta o eixo x mais à frente, no ponto x1 = 1.

Na segunda iteração (x1 = 1): A derivada muda de sinal e passa a ser positiva (f'(1) = 1), indicando uma reta tangente crescente. 
Ao traçar essa nova tangente a partir de (1, f(1)), ela intercepta o eixo x exatamente de volta no ponto 0.

Geometricamente, as retas tangentes nesses dois pontos funcionam como "espelhos" que jogam a estimativa de um lado para o outro 
permanentemente. Como o algoritmo fica preso saltando entre 0 e 1, ele nunca atinge a raiz real da função. Ou seja, isso aconteceria para
10 ou infinitas iterações.
""")

    print('\nEx 2.5 b)')
    f = lambda x: math.atan(x)
    df = lambda x: 1/(1 + x**2)

    x0 = 2
    try:
        raiz_newton, hist_newton = newton(f, df, x0, eps)
        print(f'raiz encontrada: {raiz_newton}')
    except:
        print(f'para x0 = {x0} metodo diverge.')

    x0 = 1
    raiz_newton, hist_newton = newton(f, df, x0, eps)
    print(f'raiz encontrada: {raiz_newton}')
    print(f'para x0 = {x0} metodo converge.')

    x0_valores = np.arange(0, 2.0001, 0.001)
    limite_encontrado = None

    for x0 in x0_valores:
        try:
            # Chama o seu método de Newton existente
            # Ajuste o nome da função e o retorno conforme o seu código (ex: se retorna só a raiz ou tupla)
            raiz, hist = newton(f, df, x0, eps=1e-6)
            
            # Opcional: Verifique se o resultado explodiu numericamente (NaN ou infinito)
            if np.isnan(raiz) or np.isinf(raiz) or abs(raiz) > 1e3:
                limite_encontrado = x0
                break
                
        except (ValueError):
            # Se o método falhou por divisão por zero ou estouro, achamos o limite!
            limite_encontrado = x0
            break

    print(f"Limite encontrado em x0 = {limite_encontrado:.4f}")

    print("""
Para x0 = 2, a divergência acontece devido ao tamanho excessivo do salto (overshoot) gerado por uma inclinação muito fraca. 
Como a função arctan(x) possui assíntotas horizontais, à medida que x se afasta da origem (zero da função), a curva se aplaina e a 
inclinação da reta tangente se torna quase horizontal. Isso faz com que o ponto de intersecção com o eixo x se torne cada vez mais 
distante da origem, esse ponto vai ser utilizado na proxima iteração para definir o próximo salto, assim, o método entra nesse ciclo de 
saltos cada vez mais longe do zero, fazendo-o divergir.
""")

    print('\nEx 2.5 c)')
    f = lambda x: x**3 - 9*x + 3
    df = lambda x: 3*(x**2) - 9

    x0 = math.sqrt(3)
    try:
        raiz_newton, hist_newton = newton(f, df, x0)
        print(f'raiz encontrada: {raiz_newton}, {df(x0)}')
    except:
        print("""
O método divergiu, pois, geometricamente, escolher x0 como raiz quadrada de três significa que a reta tangente ao gráfico nesse ponto é 
perfeitamente horizontal (paralela ao eixo $x$). Como uma linha horizontal nunca intercepta o eixo x, o algoritmo perde a referência 
geométrica de onde projetar o próximo passo, impossibilitando totalmente a continuidade do método.
        """)