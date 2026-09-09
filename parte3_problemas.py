"""
parte3_problemas.py

Resolucao dos Problemas A (Reservatorio esferico), C (van der Waals) e D
(Taxa Interna de Retorno) da Parte 3 do TC1.

Cada problema segue, na ordem, os 5 passos obrigatorios do enunciado:
  PASSO 1 - Deducao de f(x) (a funcao cuja raiz resolve o problema)
  PASSO 2 - Fase I: isolamento explicito da raiz (tabelamento e/ou grafico)
  PASSO 3 - Justificativa da escolha do metodo e do chute/intervalo inicial
  PASSO 4 - Resultado, com unidade fisica e algarismos significativos
            coerentes com os dados de entrada
  PASSO 5 - Verificacao: substituicao da raiz de volta no problema original

Apos os 5 passos, cada problema traz uma secao de "analise complementar"
com os itens especificos pedidos no enunciado (ex.: A.2/A.3, C.2/C.3/C.4,
D.3/D.4).
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem tela, so para salvar as figuras
import matplotlib.pyplot as plt

from metodos import bisseccao, newton, secante


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


def cabecalho_passo(n, titulo):
    print(f"\n--- PASSO {n}: {titulo} ---")


# =======================================================================
# Problema A - Reservatorio esferico
# =======================================================================

def problema_A():
    print("### Problema A - Reservatorio esferico ###")

    R = 3.0        # m (raio do reservatorio)
    V_alvo = 40.0  # m^3 (volume desejado)

    # --------------------------------------------------------------
    cabecalho_passo(1, "Deducao de f(h)")
    # --------------------------------------------------------------
    print(
        "  O volume de liquido ate a altura h e V(h) = pi*h^2*(3R-h)/3.\n"
        "  Queremos V(h) = 40 m^3, portanto a raiz procurada e a de\n"
        "      f(h) = V(h) - 40 = pi*h^2*(3R-h)/3 - 40 = 0"
    )
    V = lambda h: math.pi * h**2 * (3 * R - h) / 3.0
    f = lambda h: V(h) - V_alvo

    V_max = V(2 * R)
    print(f"  Volume total da esfera (h=2R=6.0 m): {V_max:.4f} m^3 "
          f"(o alvo de 40 m^3 cabe dentro do reservatorio: OK)")

    # --------------------------------------------------------------
    cabecalho_passo(2, "Fase I - isolamento da raiz")
    # --------------------------------------------------------------
    print("  Tabelamento de sinais de f(h) em [0, 2R] = [0, 6] m (13 pontos):")
    ivs_fis = tabelar_sinais(f, 0.0, 2 * R, 13)
    print(f"  Intervalos com mudanca de sinal: {ivs_fis}")
    a, b = ivs_fis[0]
    print(f"  -> raiz fisica isolada em [{a:.3f}, {b:.3f}] m")

    # --------------------------------------------------------------
    cabecalho_passo(3, "Justificativa do metodo e do chute inicial")
    # --------------------------------------------------------------
    print(
        f"  Metodo escolhido: BISSECCAO, usando o intervalo [{a:.3f}, {b:.3f}]\n"
        f"  ja isolado no Passo 2. Justificativa: dV/dh = pi*h*(2R-h) > 0 para\n"
        f"  0<h<2R, ou seja, V(h) e estritamente MONOTONA CRESCENTE em todo o\n"
        f"  dominio fisico - a bisseccao e garantidamente robusta aqui (nao ha\n"
        f"  risco de o intervalo escolhido nao conter a raiz fisica), o que\n"
        f"  compensa sua convergencia mais lenta."
    )

    # --------------------------------------------------------------
    cabecalho_passo(4, "Resultado")
    # --------------------------------------------------------------
    eps_h = 1e-4  # bem abaixo da exigencia de 1 mm = 1e-3 m
    h1, hist1 = bisseccao(f, a, b, eps=eps_h)
    print(f"  h = {h1:.3f} m")
    print(f"  (criterio de parada: |f(x)|<{eps_h:g} ou passo<{eps_h:g} m; "
          f"{hist1[-1]['k']+1} iteracoes)")

    # --------------------------------------------------------------
    cabecalho_passo(5, "Verificacao")
    # --------------------------------------------------------------
    print(f"  V({h1:.3f}) = {V(h1):.6f} m^3  "
          f"(alvo = {V_alvo} m^3, residuo = {f(h1):.2e} m^3)")
    print("  A altura obtida reproduz o volume desejado dentro da tolerancia.")

    # ================================================================
    # Analise complementar (itens especificos do enunciado A.2 e A.3)
    # ================================================================
    print("\n=== Analise complementar ===")

    # --- A.2: as tres raizes reais da cubica --------------------------
    print("\nA.2) Tabelamento amplo para achar as 3 raizes reais da cubica:")
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
        "  - Apenas a raiz dentro de [0, 2R] (a do Passo 4) e fisicamente\n"
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
        "  pequeno volume extra ja faz o nivel subir bastante."
    )


# =======================================================================
# Problema B - Perda de carga em tubulação
# =======================================================================


def problema_B():
    D = 0.1
    e = 4.5e-5
    Re = 2e5
    L = 500
    Q = 0.05

    f = lambda x: x + 2*math.log10((e/(3.7*D)) + ((2.51*x)/Re))
    df = lambda x: 1 + (2/math.log(10, math.e))*((2.51/Re)/((e/(3.7*D)) + ((2.51*x)/Re)))
    intervalos = tabelar_sinais(f, 0, 20, 15)
    a = intervalos[0][0]
    b = intervalos[0][1]
    print('B1:')
    print(f'Intervalos que possuem mudanças de raiz: {intervalos}\nAplicar secante')
    raiz, hist = secante(f, a, b)
    print(f'x = {raiz}, f = {1/(raiz**2)}')
    print(f'iterações: {hist[-1]["k"] + 1}')

    f_valor = 1/(raiz**2)

    print()
    print('B2:')
    print(f'Intervalos que possuem mudanças de raiz: {intervalos}\nAplicar os três métodos.')
    print('Secante')
    raiz, hist = secante(f, a, b)
    print(f'x = {raiz}, f = {1/(raiz**2)}')
    print(f'iterações: {hist[-1]["k"] + 1}')
    print('Newton')
    raiz, hist = newton(f, df, a)
    print(f'x = {raiz}, f = {1/(raiz**2)}')
    print(f'iterações: {hist[-1]["k"] + 1}')
    print('Bissecção')
    raiz, hist = bisseccao(f, a, b)
    print(f'x = {raiz}, f = {1/(raiz**2)}')
    print(f'iterações: {hist[-1]["k"] + 1}')

    print()
    print('B3:')
    f0 = 0.25/((math.log10((e/(3.7*D)) + (5.74/(Re**0.9))))**2)
    
    print(f'f0 = {f0}')

    print('Secante')
    raiz, hist = secante(f, f0, b)
    print(f'x = {raiz}, f = {1/(raiz**2)}')
    print(f'iterações: {hist[-1]["k"] + 1}')
    print('Newton')
    raiz, hist = newton(f, df, f0)
    print(f'x = {raiz}, f = {1/(raiz**2)}')
    print(f'iterações: {hist[-1]["k"] + 1}')

    f0 = 0.05

    print(f'f0 = {f0}')

    print('Secante')
    raiz, hist = secante(f, f0, b)
    print(f'x = {raiz}, f = {1/(raiz**2)}')
    print(f'iterações: {hist[-1]["k"] + 1}')
    print('Newton')
    raiz, hist = newton(f, df, f0)
    print(f'x = {raiz}, f = {1/(raiz**2)}')
    print(f'iterações: {hist[-1]["k"] + 1}')

    print()
    print('Nenhuma iteração foi economizada em relação ao chute arbitrário')

    print()
    print("B4:")
    A = (math.pi * D**2)/4
    V = Q/A
    h_f = f_valor * (L/D) * ((V**2)/(2*9.81))

    print(f'h_f = {h_f}')
    
    print()
    print("B5:")
    h_f_new = 0.02 * (L/D) * ((V**2)/(2*9.81))

    print(f'h_f = {h_f_new}, erro = {(abs(h_f - h_f_new)/h_f)*100:.2f}%')

    print('Se vale a pena ou não dependeria da aplicação mas o erro percentual parece sim, muito baixo.')
# =======================================================================

# =======================================================================
# Problema C - Equacao de van der Waals
# =======================================================================

def problema_C():
    print("\n\n### Problema C - Equacao de van der Waals (CO2) ###")

    Rg = 8.314       # J/(mol K)
    a_vdw = 0.3640   # Pa m^6/mol^2
    b_vdw = 4.267e-5  # m^3/mol
    T = 300.0        # K
    P = 5.0e6        # Pa
    Tc = 304.2       # K (temperatura critica do CO2)

    # --------------------------------------------------------------
    cabecalho_passo(1, "Deducao de f(v)")
    # --------------------------------------------------------------
    print(
        "  Equacao de van der Waals: (P + a/v^2)(v - b) = RT.\n"
        "  Multiplicando por v^2 e rearranjando, obtem-se a forma\n"
        "  polinomial (cubica em v) cuja raiz e o volume molar procurado:\n"
        "      f(v) = P*v^3 - (P*b + R*T)*v^2 + a*v - a*b = 0"
    )
    f = lambda v: P * v**3 - (P * b_vdw + Rg * T) * v**2 + a_vdw * v - a_vdw * b_vdw
    df = lambda v: 3 * P * v**2 - 2 * (P * b_vdw + Rg * T) * v + a_vdw

    v_ideal = Rg * T / P
    print(f"  Referencia de gas ideal (v=RT/P): v_ideal = {v_ideal:.4e} m^3/mol")

    # --------------------------------------------------------------
    cabecalho_passo(2, "Fase I - isolamento da raiz")
    # --------------------------------------------------------------
    print(
        "  Os valores de v sao da ordem de 1e-4 m^3/mol: um tabelamento\n"
        "  linear seria ou grosseiro demais ou exigiria pontos demais.\n"
        "  Por isso usamos espacamento LOGARITMICO em [1e-5, 1e-2] m^3/mol\n"
        "  (400 pontos):"
    )
    vs_log = np.logspace(-5, -2, 400)
    ys = [f(v) for v in vs_log]
    ivs = []
    for i in range(len(vs_log) - 1):
        if (ys[i] > 0) != (ys[i + 1] > 0):
            ivs.append((vs_log[i], vs_log[i + 1]))
    print(f"  Intervalos com mudanca de sinal: {ivs}")
    print(
        f"  -> apenas UMA raiz real positiva e detectada nessa faixa, "
        f"isolada em\n     [{ivs[0][0]:.4e}, {ivs[0][1]:.4e}] m^3/mol, "
        f"proxima de v_ideal (fase vapor)."
    )

    # --------------------------------------------------------------
    cabecalho_passo(3, "Justificativa do metodo e do chute inicial")
    # --------------------------------------------------------------
    print(
        "  Metodo escolhido: NEWTON. Justificativa: f'(v) = 3Pv^2 -\n"
        "  2(Pb+RT)v + a e um polinomio simples, barato e imediato de\n"
        "  derivar analiticamente (sem risco de erro de derivacao), e\n"
        f"  Newton converge quadraticamente quando o chute e bom.\n"
        f"  Chute inicial: x0 = v_ideal = {v_ideal:.4e} m^3/mol - razoavel\n"
        f"  porque, na faixa de P moderada do problema, o comportamento\n"
        f"  real do gas nao deve se afastar muito do modelo ideal, e o\n"
        f"  Passo 2 confirmou que a raiz fisica esta de fato proxima desse\n"
        f"  valor."
    )

    # --------------------------------------------------------------
    cabecalho_passo(4, "Resultado")
    # --------------------------------------------------------------
    v_root, hist = newton(f, df, v_ideal, eps=1e-6 * v_ideal)
    print(f"  v = {v_root:.3e} m^3/mol  (3 algarismos significativos, "
          f"coerente com a e b dados com 3-4 algarismos)")
    print(f"  ({hist[-1]['k']+1} iteracoes)")
    erro_pct_ideal = abs(v_ideal - v_root) / v_root * 100
    print(f"  Erro percentual do modelo de gas ideal em relacao a esse "
          f"resultado: {erro_pct_ideal:.1f} %")

    # --------------------------------------------------------------
    cabecalho_passo(5, "Verificacao")
    # --------------------------------------------------------------
    lado_esquerdo = (P + a_vdw / v_root**2) * (v_root - b_vdw)
    lado_direito = Rg * T
    print(f"  Substituindo na equacao original (P + a/v^2)(v - b):")
    print(f"    lado esquerdo = {lado_esquerdo:.4f} J/mol")
    print(f"    lado direito (RT) = {lado_direito:.4f} J/mol")
    print(f"  Residuo do polinomio: f(v) = {f(v_root):.3e} (~0). "
          f"Raiz verificada.")

    # ================================================================
    # Analise complementar (itens especificos do enunciado C.2, C.3, C.4)
    # ================================================================
    print("\n=== Analise complementar ===")

    # C.2) Numero de raizes reais (ja obtido no Passo 2, reexibido aqui)
    print(
        f"\nC.2) Como visto no Passo 2, nas condicoes de T=300K e P=5MPa "
        f"existe\n  apenas 1 raiz real positiva no intervalo investigado "
        f"[1e-5, 1e-2] m^3/mol,\n  ou seja, v = {v_root:.3e} m^3/mol e a "
        f"unica solucao fisica relevante."
    )

    # C.3) discussao fisica
    print(
        "\nC.3) A pressao de saturacao do CO2 a 300K e da ordem de ~6,7 MPa;\n"
        "  como P=5 MPa esta ABAIXO da saturacao, o CO2 esta em fase VAPOR\n"
        "  pura (regiao de uma so fase), o que explica a raiz UNICA "
        "encontrada.\n"
        "  De forma geral, para T<Tc=304,2K e P ACIMA da pressao de "
        "saturacao,\n"
        "  a cubica de van der Waals PODE ter ate 3 raizes reais positivas\n"
        "  (regiao de coexistencia liquido-vapor da isoterma):\n"
        "    - a MENOR raiz = volume molar do LIQUIDO saturado;\n"
        "    - a MAIOR raiz = volume molar do VAPOR saturado;\n"
        "    - a raiz INTERMEDIARIA nao tem significado fisico (regiao com\n"
        "      dP/dv>0, mecanicamente instavel - artefato da curva suave de\n"
        "      van der Waals, que na isoterma real tem um patamar horizontal\n"
        "      em vez dessa 'corcova').\n"
        "  Partindo de um chute proximo do gas ideal (fase vapor), Newton\n"
        "  sempre converge para a raiz MAIOR (fase vapor) quando ela existe."
    )

    # C.4) isoterma T=300K, P de 1 a 10 MPa
    print("\nC.4) Isoterma P x v em T=300K (van der Waals x gas ideal):")
    Ps = np.arange(1.0, 10.01, 0.5) * 1e6
    v_vdw_list, v_id_list = [], []
    for Pi in Ps:
        fi = lambda v, Pi=Pi: Pi * v**3 - (Pi * b_vdw + Rg * T) * v**2 + a_vdw * v - a_vdw * b_vdw
        dfi = lambda v, Pi=Pi: 3 * Pi * v**2 - 2 * (Pi * b_vdw + Rg * T) * v + a_vdw
        v0_i = Rg * T / Pi  # chute = valor ideal NESTE P (evita chute ruim herdado)
        vi, _ = newton(fi, dfi, v0_i, eps=1e-6 * v0_i)
        v_vdw_list.append(vi)
        v_id_list.append(v0_i)

    plt.figure(figsize=(6, 4))
    plt.plot(np.array(v_vdw_list) * 1e6, Ps / 1e6, marker="o", label="van der Waals")
    plt.plot(np.array(v_id_list) * 1e6, Ps / 1e6, marker="x", label="gas ideal")
    plt.xlabel("v (cm^3/mol, =1e-6 m^3/mol)")
    plt.ylabel("P (MPa)")
    plt.title("Problema C - Isoterma T=300K")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("problema_C_isoterma.png", dpi=120)
    plt.close()
    print("  [grafico salvo em problema_C_isoterma.png]")
    print(
        "  O gas ideal superestima v em relacao ao van der Waals nessa\n"
        "  faixa de P (atracao molecular do termo a/v^2 'puxa' as\n"
        "  moleculas para mais perto, reduzindo o volume real ocupado)."
    )


# =======================================================================
# Problema D - Taxa Interna de Retorno (TIR)
# =======================================================================

def problema_D():
    print("\n\n### Problema D - Taxa Interna de Retorno ###")

    fluxo1 = [-1000, 300, 350, 400, 450]  # mil R$, anos 0 a 4

    def vpl(i, fluxos):
        return sum(c / (1 + i) ** k for k, c in enumerate(fluxos))

    def dvpl(i, fluxos):
        return sum(-k * c / (1 + i) ** (k + 1) for k, c in enumerate(fluxos))

    # --------------------------------------------------------------
    cabecalho_passo(1, "Deducao de f(i)")
    # --------------------------------------------------------------
    print(
        "  A TIR e, por definicao, a taxa i que zera o VPL. A raiz\n"
        "  procurada e a de\n"
        "      f(i) = VPL(i) = soma_{k=0}^{4} C_k/(1+i)^k = 0\n"
        "  Para uso em Newton, tambem precisamos de\n"
        "      f'(i) = soma_{k=0}^{4} [-k*C_k/(1+i)^(k+1)]"
    )
    f1 = lambda i: vpl(i, fluxo1)
    df1 = lambda i: dvpl(i, fluxo1)

    # --------------------------------------------------------------
    cabecalho_passo(2, "Fase I - isolamento da raiz")
    # --------------------------------------------------------------
    print("  Tabelamento de VPL(i) em [0, 0.5] (51 pontos):")
    ivs = tabelar_sinais(f1, 0.0, 0.5, 51)
    print(f"  Intervalo com mudanca de sinal: {ivs}")
    a, b = ivs[0]
    print(
        f"  -> raiz isolada em [{a:.2f}, {b:.2f}]. VPL(i) e monotonicamente\n"
        f"  decrescente em [0, 0.5] (fluxo de caixa CONVENCIONAL - uma unica\n"
        f"  troca de sinal em C_k), cruzando o eixo uma unica vez."
    )

    # --------------------------------------------------------------
    cabecalho_passo(3, "Justificativa do metodo e do chute inicial")
    # --------------------------------------------------------------
    print(
        "  Metodo escolhido: NEWTON. Justificativa: f'(i) e uma soma simples\n"
        "  de potencias de (1+i), barata de calcular analiticamente, e a\n"
        "  curva VPL(i) e suave e sem inflexoes bruscas na regiao isolada -\n"
        "  condicoes ideais para a convergencia quadratica de Newton.\n"
        "  Chute inicial: i0 = 0.10 (10%), taxa de referencia de mercado\n"
        "  tipicamente usada como primeira estimativa e ja razoavelmente\n"
        "  proxima da regiao isolada no Passo 2."
    )

    # --------------------------------------------------------------
    cabecalho_passo(4, "Resultado")
    # --------------------------------------------------------------
    tir1, hist1 = newton(f1, df1, 0.10, eps=1e-6)
    print(f"  TIR = {tir1*100:.4f} %  (6 algarismos significativos, "
          f"coerente com a tolerancia 1e-6 exigida)")
    print(f"  ({hist1[-1]['k']+1} iteracoes)")

    # --------------------------------------------------------------
    cabecalho_passo(5, "Verificacao")
    # --------------------------------------------------------------
    print(f"  VPL({tir1*100:.4f}%) = {f1(tir1):.3e} mil R$ (~0). "
          f"Raiz verificada.")

    # ================================================================
    # Analise complementar (itens especificos do enunciado D.2, D.3, D.4)
    # ================================================================
    print("\n=== Analise complementar ===")

    # D.2) grafico
    print("\nD.2) Grafico de VPL(i) em [0, 0.5] com a raiz destacada:")
    is_ = np.linspace(0, 0.5, 200)
    plt.figure(figsize=(6, 4))
    plt.plot(is_ * 100, [f1(i) for i in is_])
    plt.axhline(0, color="k", lw=0.8)
    plt.axvline(tir1 * 100, color="r", ls="--", label=f"TIR={tir1*100:.2f}%")
    plt.xlabel("i (%)")
    plt.ylabel("VPL (mil R$)")
    plt.title("Problema D - VPL(i), projeto 1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("problema_D_vpl_projeto1.png", dpi=120)
    plt.close()
    print("  [grafico salvo em problema_D_vpl_projeto1.png]")

    # D.3) decisao com custo de capital
    print("\nD.3) Decisao de aceitar ou rejeitar o projeto:")
    for custo in [0.15, 0.20]:
        vpl_custo = f1(custo)
        decisao = "ACEITAR" if vpl_custo > 0 else "REJEITAR"
        comparacao = ">" if tir1 > custo else "<"
        print(f"  custo de capital = {custo*100:.0f}%: "
              f"VPL = {vpl_custo:.4f} mil R$  -> {decisao} "
              f"(TIR={tir1*100:.2f}% {comparacao} custo de capital)")

    # D.4) segundo projeto - multiplas TIRs
    print("\nD.4) Projeto 2 (fluxo nao-convencional), fluxos:")
    fluxo2 = [-1000, 2500, -1540]
    f2 = lambda i: vpl(i, fluxo2)
    print(f"  {fluxo2}")

    ivs2 = tabelar_sinais(f2, -0.3, 1.0, 400)
    print(f"  Intervalos com mudanca de sinal: {ivs2}")
    tirs2 = []
    for (ai, bi) in ivs2:
        r, h = bisseccao(f2, ai, bi, eps=1e-8)
        tirs2.append(r)
    print(f"  As DUAS TIRs do projeto 2: "
          f"{[f'{t*100:.4f}%' for t in tirs2]}")
    for t in tirs2:
        print(f"    Verificacao: VPL({t*100:.4f}%) = {f2(t):.3e} (~0)")

    is2 = np.linspace(-0.3, 1.0, 400)
    plt.figure(figsize=(6, 4))
    plt.plot(is2 * 100, [f2(i) for i in is2])
    plt.axhline(0, color="k", lw=0.8)
    plt.axvline(tirs2[0] * 100, color="r", ls="--",
                label=f"TIR1={tirs2[0]*100:.1f}%")
    plt.axvline(tirs2[1] * 100, color="darkred", ls="--",
                label=f"TIR2={tirs2[1]*100:.1f}%")
    plt.xlabel("i (%)")
    plt.ylabel("VPL (mil R$)")
    plt.title("Problema D - VPL(i), projeto 2 (duas raizes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("problema_D_vpl_projeto2.png", dpi=120)
    plt.close()
    print("  [grafico salvo em problema_D_vpl_projeto2.png]")

    print(
        "\n  Pergunta critica: se o analista rodar Newton com um UNICO\n"
        "  chute inicial (ex.: i0=0.5), ele encontra APENAS uma das duas\n"
        "  TIRs (10% ou 40%, dependendo do chute) e pode concluir\n"
        "  erroneamente que o projeto tem uma unica taxa de retorno bem\n"
        "  definida. Isso e perigoso porque a REGRA DA TIR ('aceite se\n"
        "  TIR > custo de capital') so faz sentido quando ha uma unica\n"
        "  raiz: com fluxo de caixa nao-convencional (mais de uma troca de\n"
        "  sinal, como aqui: -,+,-), pode haver VARIAS taxas que zeram o\n"
        "  VPL, e a decisao de aceitar/rejeitar passa a depender de QUAL\n"
        "  TIR foi encontrada - um analista que pula a Fase I (tabelamento/\n"
        "  grafico) pode reportar 'TIR=40%, superior ao custo de capital de\n"
        "  15%, ACEITAR' sem perceber que a 10% o projeto TAMBEM zera e que\n"
        "  o VPL pode ser negativo para custos de capital entre as duas\n"
        "  raizes - levando a uma decisao de investimento errada."
    )




# =======================================================================
# Problema E - Equacao de Kepler
# =======================================================================

import math

from metodos import newton


def problema_E():
    import math

from metodos import newton


def problema_E():
    print("\n\n### Problema E — Equação de Kepler ###\n")

    # ============================================================
    # DEFINIÇÕES GERAIS
    # ============================================================

    eps = 1e-8
    max_iter = 200

    # Equação de Kepler escrita como f(E) = 0
    def f(E, e, M):
        return E - e * math.sin(E) - M

    # Derivada de f
    def df(E, e):
        return 1 - e * math.cos(E)


    # ============================================================
    # E.1 — COMETA HALLEY
    # ============================================================

    print("\n--- E.1 — Cometa Halley ---\n")

    e = 0.967
    M = 0.2
    E0 = 1.0

    # Funções específicas do caso para passar ao método de Newton
    funcao = lambda E: f(E, e, M)
    derivada = lambda E: df(E, e)

    # Fase I
    print("Fase I:")
    print("f(1.0) =", funcao(1.0))
    print("f(1.1) =", funcao(1.1))

    raiz, historico = newton(
        funcao,
        derivada,
        E0,
        eps=eps,
        max_iter=max_iter
    )

    print("\nResultado E.1:")
    print("e =", e)
    print("M =", M)
    print("E0 =", E0)
    print("Raiz E =", raiz)
    print("Número de iterações =", len(historico))
    print("Critério de parada: eps =", eps)

    print("\nVerificação:")
    print("E - e*sin(E) =", raiz - e * math.sin(raiz))

    print("\nHistórico:")
    for passo in historico:
        print(passo)


    # ============================================================
    # E.2 — ROBUSTEZ DO MÉTODO DE NEWTON
    # ============================================================

    print("\n\n--- E.2 — Robustez do método de Newton ---\n")

    casos = [
        ("(i)",   0.10, 0.5),
        ("(ii)",  0.90, 0.1),
        ("(iii)", 0.99, 0.01),
    ]

    resultados_e2 = {}

    for nome_caso, e, M in casos:

        E0 = M

        funcao = lambda E, e=e, M=M: f(E, e, M)
        derivada = lambda E, e=e: df(E, e)

        raiz, historico = newton(
            funcao,
            derivada,
            E0,
            eps=eps,
            max_iter=max_iter
        )

        resultados_e2[nome_caso] = {
            "e": e,
            "M": M,
            "E0": E0,
            "raiz": raiz,
            "iteracoes": len(historico),
        }

        print(f"\nCaso {nome_caso}")
        print("e =", e)
        print("M =", M)
        print("E0 =", E0)

        print("\nResultado:")
        print("Raiz E =", raiz)
        print("Número de iterações =", len(historico))
        print("Resíduo |f(E)| =", abs(funcao(raiz)))
        print("f'(E0) =", derivada(E0))

        print("\nVerificação:")
        print("E - e*sin(E) =", raiz - e * math.sin(raiz))

        print("\nHistórico:")
        for passo in historico:
            print(passo)

    print("\nResumo E.2:")
    print(f"{'Caso':<8} {'E0':<18} {'Raiz':<20} {'Iterações'}")

    for nome_caso, dados in resultados_e2.items():
        print(
            f"{nome_caso:<8}"
            f"{dados['E0']:<18.10f}"
            f"{dados['raiz']:<20.12f}"
            f"{dados['iteracoes']}"
        )


    # ============================================================
    # E.3 — CHUTE INICIAL MELHORADO
    # ============================================================

    print("\n\n--- E.3 — Chute inicial melhorado ---\n")

    e = 0.99
    M = 0.01

    E0 = M + e * math.sin(M) #chute melhorado

    funcao = lambda E: f(E, e, M)
    derivada = lambda E: df(E, e)

    raiz, historico = newton(
        funcao,
        derivada,
        E0,
        eps=eps,
        max_iter=max_iter
    )

    print("e =", e)
    print("M =", M)
    print("E0 = M + e*sin(M) =", E0)

    print("\nResultado E.3:")
    print("Raiz E =", raiz)
    print("Número de iterações =", len(historico))
    print("Resíduo |f(E)| =", abs(funcao(raiz)))

    print("\nVerificação:")
    print("E - e*sin(E) =", raiz - e * math.sin(raiz))

    print("\nHistórico:")
    for passo in historico:
        print(passo)

    # Comparação com E.2 (iii)
    caso_anterior = resultados_e2["(iii)"]

    print("\nComparação E.2 (iii) x E.3:")
    print(f"{'Chute inicial':<28} {'E0':<18} {'Iterações'}")

    print(
        f"{'E0 = M':<28}"
        f"{caso_anterior['E0']:<18.10f}"
        f"{caso_anterior['iteracoes']}"
    )

    print(
        f"{'E0 = M + e*sin(M)':<28}"
        f"{E0:<18.10f}"
        f"{len(historico)}"
    )

        # ============================================================
    # E.4 — COMPARAÇÃO COM O MÉTODO DA BISSECÇÃO
    # ============================================================

    print("\n\n--- E.4 — Comparação com a bissecção ---\n")

    # Mesmo caso (iii)
    e = 0.99
    M = 0.01

    funcao = lambda E: f(E, e, M)

    # Intervalo exigido pelo enunciado
    a = 0.0
    b = math.pi

    # Fase I
    print("Fase I:")
    print("a =", a)
    print("b =", b)
    print("f(a) =", funcao(a))
    print("f(b) =", funcao(b))
    print("f(a) * f(b) =", funcao(a) * funcao(b))

    # Método da bissecção
    raiz_bisseccao, historico_bisseccao = bisseccao(
        funcao,
        a,
        b,
        eps=eps,
        max_iter=max_iter
    )

    print("\nResultado da bissecção:")
    print("Raiz E =", raiz_bisseccao)
    print("Número de iterações =", len(historico_bisseccao))
    print("Resíduo |f(E)| =", abs(funcao(raiz_bisseccao)))
    print("Critério de parada: eps =", eps)

    print("\nVerificação:")
    print(
        "E - e*sin(E) =",
        raiz_bisseccao - e * math.sin(raiz_bisseccao)
    )

    print("\nHistórico da bissecção:")
    for passo in historico_bisseccao:
        print(passo)

    # ------------------------------------------------------------
    # Comparação com Newton — caso (iii) da E.2
    # ------------------------------------------------------------

    caso_iii = resultados_e2["(iii)"]

    print("\nComparação Newton x Bissecção:")
    print(
        f"{'Método':<15}"
        f"{'Raiz':<20}"
        f"{'Iterações'}"
    )

    print(
        f"{'Newton':<15}"
        f"{caso_iii['raiz']:<20.12f}"
        f"{caso_iii['iteracoes']}"
    )

    print(
        f"{'Bissecção':<15}"
        f"{raiz_bisseccao:<20.12f}"
        f"{len(historico_bisseccao)}"
    )

# =======================================================================
# Problema F (BONUS) - Deflexao de viga
# =======================================================================

def problema_F():
    print("\n\n### Problema F (BONUS) - Deflexao de viga ###")

    L = 600.0        # cm (comprimento da viga)
    E_mod = 50000.0  # kN/cm^2 (modulo de elasticidade)
    I = 30000.0      # cm^4 (momento de inercia)
    w0 = 2.5         # kN/cm (intensidade da carga distribuida)

    C = w0 / (120 * L * E_mod * I)  # constante multiplicativa de y(x)
    y = lambda x: C * (-x**5 + 2 * L**2 * x**3 - L**4 * x)
    dy = lambda x: C * (-5 * x**4 + 6 * L**2 * x**2 - L**4)
    d2y = lambda x: C * (-20 * x**3 + 12 * L**2 * x)  # derivada de dy, p/ Newton

    # --------------------------------------------------------------
    cabecalho_passo(1, "Deducao de f(x)")
    # --------------------------------------------------------------
    print(
        "  O ponto de deflexao maxima e onde dy/dx = 0. Derivando y(x)\n"
        "  analiticamente, a raiz procurada e a de\n"
        "      f(x) = dy/dx = C*(-5x^4 + 6L^2 x^2 - L^4) = 0\n"
        "  com C = w0/(120*L*E*I). Dividindo por C e fazendo u = x^2\n"
        "  (equacao biquadrada), obtemos por Bhaskara:\n"
        "      5x^4 - 6L^2 x^2 + L^4 = 0\n"
        "      u = [6L^2 +- sqrt(36L^4 - 20L^4)] / 10 = [6L^2 +- 4L^2] / 10\n"
        "  ou seja, u = L^2 (=> x = L, RAIZ ESPURIA de borda - ver item F.3)\n"
        "  ou u = 0,2 L^2 (=> x = L*sqrt(1/5), candidato interior valido)."
    )
    x_analitico = L * math.sqrt(0.2)
    print(f"  Solucao analitica interior: x = L*sqrt(1/5) = "
          f"{x_analitico:.4f} cm")

    # --------------------------------------------------------------
    cabecalho_passo(2, "Fase I - isolamento da raiz")
    # --------------------------------------------------------------
    print("  Tabelamento de sinais de dy/dx no intervalo ABERTO (0, L) "
          "(60 pontos):")
    ivs = tabelar_sinais(dy, 1e-6, L - 1e-6, 60)
    print(f"  Intervalos com mudanca de sinal: {ivs}")
    a, b = ivs[0]
    print(f"  -> raiz interior isolada em [{a:.3f}, {b:.3f}] cm "
          f"(note que L={L:.0f} cm foi excluido de proposito do "
          f"tabelamento - ver F.3)")

    # --------------------------------------------------------------
    cabecalho_passo(3, "Justificativa do metodo e do chute inicial")
    # --------------------------------------------------------------
    print(
        "  Metodo escolhido: NEWTON. Justificativa: dy/dx e um polinomio\n"
        "  de grau 4, cuja derivada d2y/dx2 = C*(-20x^3 + 12L^2 x) e\n"
        "  simples e barata de obter analiticamente, permitindo\n"
        "  convergencia quadratica.\n"
        f"  Chute inicial: x0 = L/2 = {L/2:.1f} cm (meio do vao) - robusto\n"
        f"  porque esta bem no interior do intervalo isolado no Passo 2 e\n"
        f"  longe da raiz espuria de borda x=L."
    )

    # --------------------------------------------------------------
    cabecalho_passo(4, "Resultado")
    # --------------------------------------------------------------
    x_max, histF1 = newton(dy, d2y, L / 2, eps=1e-6)
    print(f"  x_max = {x_max:.2f} cm  (6 algarismos significativos, "
          f"coerente com os dados de entrada dados com 2-5 algarismos)")
    print(f"  ({histF1[-1]['k']+1} iteracoes)")

    # --------------------------------------------------------------
    cabecalho_passo(5, "Verificacao")
    # --------------------------------------------------------------
    print(f"  dy/dx({x_max:.2f}) = {dy(x_max):.3e} (~0). Raiz verificada.")
    print(f"  (compare com a solucao analitica do Passo 1: "
          f"x = {x_analitico:.2f} cm - praticamente identico)")

    # ================================================================
    # Analise complementar (itens especificos do enunciado F.2, F.3, F.4)
    # ================================================================
    print("\n=== Analise complementar ===")

    # F.2) deflexao maxima
    y_max = y(x_max)
    print(f"\nF.2) Deflexao maxima: y({x_max:.2f}) = {y_max:.4f} cm "
          f"(sinal negativo = deflexao para BAIXO, convencao usual de "
          f"vigas)")

    # F.3) armadilha em x=0 e x=L
    print("\nF.3) Armadilha proposital:")
    print(f"  dy/dx(0) = {dy(0.0):.6e}")
    print(f"  dy/dx(L) = {dy(L):.6e}")
    print(
        "  dy/dx(0) e claramente diferente de zero (extremidade nao e\n"
        "  ponto critico), mas dy/dx(L) e ALGEBRICAMENTE zero tambem (a\n"
        "  extremidade x=L E uma raiz matematica da derivada, como achado\n"
        "  no Passo 1: u=L^2 => x=L)! Um aluno que aplica bisseccao\n"
        "  ingenuamente em [0, L] esbarra exatamente nessa armadilha: com\n"
        "  f(a)=dy/dx(0)<0 e f(b)=dy/dx(L)=0, o teste ROBUSTO por sinal do\n"
        "  metodos.py ((fa>0)==(fb>0), pois 0.0>0 e False, igual ao sinal\n"
        "  de f(a)) detecta corretamente que NAO ha troca de sinal genuina\n"
        "  e recusa o intervalo - ou seja, a implementacao 'salva' o aluno\n"
        "  de uma convergencia para um ponto sem sentido fisico. Testando:"
    )
    try:
        x_bis_errado, hist_bis_err = bisseccao(dy, 0.0, L, eps=1e-6)
        print(f"    bisseccao(dy, 0, L) convergiu para x = "
              f"{x_bis_errado:.4f} cm -> raiz SEM sentido fisico "
              f"(extremidade)")
    except ValueError as e:
        print(f"    bisseccao(dy, 0, L) levantou ValueError: {e}")
    print(
        "  (Se a implementacao usasse o teste classico fa*fb<0, o produto\n"
        "  daria -0.003*0.0=0.0, que tambem falha o teste '<0' - mas em\n"
        "  ponto flutuante f(L) raramente e EXATAMENTE zero, podendo dar um\n"
        "  residuo minusculo de sinal aleatorio; se esse residuo desse\n"
        "  positivo, o teste passaria e a bisseccao convergiria SILENCIO-\n"
        "  SAMENTE para x~L, um resultado sem sentido fisico.)\n"
        "  Correcao: restringir o intervalo para EXCLUIR a extremidade\n"
        "  espuria, por exemplo [0, 0.9L] ou, melhor, usar o intervalo ja\n"
        "  isolado pelo tabelamento no Passo 2 (que exclui x=L),\n"
        "  garantindo que a bisseccao va atras apenas da raiz interior."
    )

    # F.4) derivada numerica (diferencas centradas) e erro vs h
    print("\nF.4) Derivada numerica central: f'(x) ~ (f(x+h)-f(x-h))/(2h)")
    hs = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    erros = []
    for h in hs:
        dy_num = lambda x, h=h: (y(x + h) - y(x - h)) / (2 * h)
        d2y_num = lambda x, h=h: (dy_num(x + h) - dy_num(x - h)) / (2 * h)
        try:
            x_num, hist_num = newton(dy_num, d2y_num, L / 2, eps=1e-6, max_iter=100)
            erro = abs(x_num - x_analitico)
        except (ValueError, ZeroDivisionError):
            x_num, erro = float("nan"), float("nan")
        erros.append(erro)
        print(f"  h={h:.0e}:  x_max_numerico={x_num:.6f} cm  "
              f"erro={erro:.3e} cm")

    plt.figure(figsize=(6, 4))
    plt.loglog(hs, erros, marker="o")
    plt.xlabel("h")
    plt.ylabel("erro em x_max (cm)")
    plt.title("Problema F - Erro da derivada numerica vs h")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("problema_F_erro_vs_h.png", dpi=120)
    plt.close()
    print("  [grafico salvo em problema_F_erro_vs_h.png]")
    print(
        "\n  Por que o erro NAO cai indefinidamente quando h->0: ha DOIS\n"
        "  erros competindo. O erro de TRUNCAMENTO da formula de diferenca\n"
        "  centrada e O(h^2) e diminui com h menor - mas o erro de\n"
        "  ARREDONDAMENTO de ponto flutuante, causado por subtrair dois\n"
        "  numeros f(x+h) e f(x-h) MUITO proximos (cancelamento\n"
        "  catastrofico) e depois dividir por 2h (h tambem minusculo),\n"
        "  CRESCE conforme h->0. Existe um h* otimo (tipicamente perto de\n"
        "  1e-5 a 1e-6 para diferenca centrada em double precision) onde a\n"
        "  soma dos dois erros e minima; para h menor que isso, o erro\n"
        "  volta a AUMENTAR (ou o metodo ate falha, como visto acima para\n"
        "  h<=1e-6). Este e o fenomeno classico de compromisso\n"
        "  truncamento-vs-arredondamento em diferenciacao numerica."
    )


# =======================================================================

if __name__ == "__main__":
    problema_A()
    print("\n" + "=" * 70)
    problema_B()
    print("\n" + "=" * 70)
    problema_C()
    print("\n" + "=" * 70)
    problema_D()
    print("\n" + "=" * 70)
    problema_E()
    print("\n" + "=" * 70)
    problema_F()