"""
metodos.py
"""

import math


def contador(func):
    """Decorator que conta quantas vezes 'func' foi efetivamente chamada."""
    def wrapper(*args, **kwargs):
        wrapper.n += 1
        return func(*args, **kwargs)
    wrapper.n = 0
    return wrapper

def secante(f, x0, x1, eps=1e-8, max_iter=200):
    """Método da secante.

    Retorna (raiz, historico). historico[i] contém k, x, fx, erro e
    'avaliacoes_f' (contagem acumulada).
    """
    g = contador(f)

    f0, f1 = g(x0), g(x1)
    historico = [
        {"k": 0, "x": x0, "fx": f0, "erro": float("nan"), "avaliacoes_f": g.n, "convergiu": False},
    ]
    if abs(f1) < eps:
        historico.append({"k": 1, "x": x1, "fx": f1, "erro": abs(x1 - x0),
                           "avaliacoes_f": g.n, "convergiu": True})
        return x1, historico

    convergiu = False
    for k in range(1, max_iter + 1):
        denom = (f1 - f0)
        if denom == 0.0:
            print(f"[secante] AVISO: denominador nulo (f(x_{k})=f(x_{k-1})) "
                  f"em x={x1:.10g}. Parando.")
            break

        x2 = x1 - f1 * (x1 - x0) / denom
        f2 = g(x2)
        erro = abs(x2 - x1)

        historico.append({
            "k": k, "x": x2, "fx": f2, "erro": erro,
            "avaliacoes_f": g.n, "convergiu": False,
        })

        x0, f0 = x1, f1
        x1, f1 = x2, f2

        if erro < eps or abs(f2) < eps:
            historico[-1]["convergiu"] = True
            convergiu = True
            break

    if not convergiu:
        print(f"[secante] AVISO: não convergiu em {max_iter} iterações "
              f"(eps={eps}). Retornando melhor valor encontrado.")

    return historico[-1]["x"], historico


if __name__ == "__main__":
    # Auto-teste rápido com f(x) = x^3 - 9x + 3, raiz perto de 0.3376 em (0,1)
    f = lambda x: x**3 - 9*x + 3
    df = lambda x: 3*x**2 - 9

    r3, h3 = secante(f, 0, 1)

    print("Secante:  ", r3, "em", len(h3) - 1, "iterações")
