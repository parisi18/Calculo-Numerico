"""
parte3_problemas.py

Resolucao do Problema A (Reservatorio esferico) da Parte 3 do TC1.

O codigo segue explicitamente os 5 passos exigidos no enunciado:
  1) deducao de f(x);
  2) Fase I - isolamento (tabelamento e/ou grafico);
  3) justificativa do metodo e do chute inicial;
  4) resultado com unidade fisica e algarismos significativos coerentes;
  5) verificacao substituindo a raiz no problema original.
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem tela, so para salvar as figuras
import matplotlib.pyplot as plt

from metodos import bisseccao


# =======================================================================
# Funcao auxiliar geral de Fase I (isolamento por tabelamento de sinais)
# =======================================================================

def tabelar_sinais(f, a, b, n):
    """Mesma ideia do exercicio 2.1: varre [a,b] com n pontos e devolve
    os subintervalos onde f muda de sinal (candidatos a raiz)."""
    xs = np.linspace(a, b, n)
    ys = [f(x) for x in xs]
    intervalos = []
    for i in range(len(xs) - 1):
        if ys[i] == 0:
            intervalos.append((xs[i], xs[i]))
        elif (ys[i] > 0) != (ys[i + 1] > 0):
            intervalos.append((xs[i], xs[i + 1]))
    return intervalos


# =======================================================================
# Problema A - Reservatorio esferico
# =======================================================================

def problema_A():
    print("### Problema A - Reservatorio esferico ###\n")

    R = 3.0        # m
    V_alvo = 40.0  # m^3

    # 1) f(h): volume da calota menos o volume desejado
    V = lambda h: math.pi * h**2 * (3 * R - h) / 3.0
    f = lambda h: V(h) - V_alvo
    df = lambda h: math.pi * (2 * R * h - h**2)  # dV/dh

    V_max = V(2 * R)
    print(f"Volume total da esfera (h=2R=6.0 m): {V_max:.4f} m^3 "
          f"(o alvo de 40 m^3 cabe dentro do reservatorio: OK)\n")

    # --- A.1: altura fisica (0 <= h <= 2R), erro < 1 mm --------------
    print("A.1) Fase I - tabelamento de sinais de f(h)=V(h)-40 em [0, 6] m:")
    ivs_fis = tabelar_sinais(f, 0.0, 2 * R, 13)
    print(f"  Intervalos com mudanca de sinal: {ivs_fis}")
    a, b = ivs_fis[0]
    print(
        f"\n  Metodo escolhido: bisseccao, pelo intervalo fisico [{a:.3f},"
        f" {b:.3f}] ja isolado e por V(h) ser monotona crescente em [0,2R]\n"
        f"  (garante convergencia robusta sem risco de sair do dominio fisico)."
    )
    eps_h = 1e-4  # bem abaixo de 1 mm = 1e-3 m
    h1, hist1 = bisseccao(f, a, b, eps=eps_h)
    print(f"\n  Resultado: h = {h1:.4f} m  "
          f"(criterio: |f(x)|<{eps_h:g} ou passo<{eps_h:g}; "
          f"{hist1[-1]['k']+1} iteracoes)")
    print(f"  Verificacao: V({h1:.4f}) = {V(h1):.6f} m^3  "
          f"(alvo = {V_alvo} m^3, residuo = {f(h1):.2e})\n")

    # --- A.2: as tres raizes reais da cubica --------------------------
    print("A.2) Tabelamento amplo para achar as 3 raizes reais da cubica:")
    ivs_todas = tabelar_sinais(f, -5.0, 12.0, 350)
    print(f"  Intervalos encontrados: {ivs_todas}")

    raizes = []
    for (ai, bi) in ivs_todas:
        r, h = bisseccao(f, ai, bi, eps=1e-8)
        raizes.append(r)
    raizes.sort()
    print(f"\n  Raizes da cubica: {[f'{r:.6f}' for r in raizes]} m")
    print(
        "\n  Interpretacao:\n"
        "  - A raiz negativa (h<0) e uma raiz puramente matematica do\n"
        "    polinomio: nao existe altura negativa, e a restricao fisica\n"
        "    0 <= h a elimina.\n"
        "  - A raiz com h>2R=6 m e outra raiz espuria: a formula V(h) so\n"
        "    descreve o volume da calota enquanto 0<=h<=2R (altura nao\n"
        "    pode exceder o diametro da esfera); alem disso, algebricamente\n"
        "    V(h) volta a CRESCER sem limite para h>2R, o que fisicamente\n"
        "    nao tem sentido (o reservatorio esta cheio em h=2R e nao ha\n"
        "    'mais volume' para h maior). A restricao 0<=h<=2R a elimina.\n"
        "  - Apenas a raiz dentro de [0, 2R] (a do item A.1) e fisicamente\n"
        "    valida."
    )

    # --- A.3: tabela h x V e grafico -----------------------------------
    print("\nA.3) Tabela h x V (resolvendo h para cada V, por bisseccao):")
    Vs = list(range(10, 111, 10))
    hs = []
    for Vi in Vs:
        fi = lambda h, Vi=Vi: V(h) - Vi
        hi, _ = bisseccao(fi, 0.0, 2 * R, eps=1e-8)
        hs.append(hi)
    tab_A3 = pd.DataFrame({"V (m^3)": Vs, "h (m)": [f"{x:.4f}" for x in hs]})
    print(tab_A3.to_string(index=False))

    plt.figure(figsize=(6, 4))
    plt.plot(Vs, hs, marker="o")
    plt.xlabel("V (m^3)")
    plt.ylabel("h (m)")
    plt.title("Problema A - Curva de nivel h(V)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("problema_A_curva_hV.png", dpi=120)
    plt.close()
    print("  [grafico salvo em problema_A_curva_hV.png]")
    print(
        "\n  Por que a curva e mais 'achatada' no meio: dV/dh = pi*h*(2R-h)\n"
        "  e maxima em h=R (meio da esfera), ou seja, para um mesmo\n"
        "  incremento de h o volume varia MAIS perto do meio. Isso equivale\n"
        "  a dizer que, para um mesmo incremento de V, dh = dV/(dV/dh) e\n"
        "  MENOR no meio (curva mais achatada/plana) e MAIOR perto das\n"
        "  pontas (h perto de 0 ou de 2R), onde a esfera 'estreita' e um\n"
        "  pequeno volume extra ja faz o nivel subir bastante.\n"
    )


# =======================================================================

if __name__ == "__main__":
    problema_A()
