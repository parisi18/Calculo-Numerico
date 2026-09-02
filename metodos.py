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

def bisseccao (f , a , b , eps=1e-8, max_iter =200) :
    """ Retorna (raiz , historico )."""

    g = contador(f)

    fa = g(a)
    fb = g(b)
    
    if(fa * fb >= 0):
        raise(ValueError(f"O intervalo passado não contempla um zero para a função f: f(a) * f(b) = {fa * fb}"))
    hist = []
    for k in range(max_iter):
        x = (a + b)/2
        fx = g(x)
        hist.append({
            "k": k
            , "x": x
            , "fx": fx
            , "erro": (b-a)/2.0
        })
        if (abs(fx) < eps or (b-a)/2.0 < eps):
            print(f"Total de chamadas -> f: {g.n}")
            return (x, hist)
        elif fa * fx < 0:
            b = x
            fb = fx
        else:
            a = x
            fa = fx
    print(f"Aviso: Número máximo de iterações ({max_iter}) atingido sem convergência.")
    print(f"Total de chamadas -> f: {g.n}")
    return (x, hist)

def newton(f, df, x0, eps=1e-8, max_iter=200):
    """Retorna (raiz, historico)."""
    
    f = contador(f)
    df = contador(df)
    
    historico = []
    xk = float(x0)
    fx = f(x0)
    
    for k in range(max_iter):
        dfx = df(xk)
        
        if abs(dfx) < 1e-15:
            raise ValueError(f"Derivada nula ou extremamente próxima de zero em x_{k} = {xk}.")
        
        x_next = xk - (fx / dfx)
        fx_next = f(x_next)
        erro = abs(x_next - xk)
        
        historico.append({
            "k": k,
            "x": x_next,
            "fx": fx_next,
            "erro": erro
        })
        
        if erro < eps or abs(fx_next) < eps:
            print(f"Total de chamadas -> f: {f.n}, df: {df.n}")
            return x_next, historico  
        
        xk = x_next
        fx = fx_next
        
    print(f"Aviso: Número máximo de iterações ({max_iter}) atingido sem convergência.")
    print(f"Total de chamadas -> f: {f.n}, df: {df.n}")
    return xk, historico

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

    r3, h3 = bisseccao(f, 0, 1)

    print("Bisseccao:  ", r3, "em", len(h3) - 1, "iterações")

    r3, h3 = newton(f, df, 0)

    print("Bisseccao:  ", r3, "em", len(h3) - 1, "iterações")
