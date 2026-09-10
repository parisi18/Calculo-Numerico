# Cálculo Numérico — métodos de busca de raízes

Este repositório reúne implementações de métodos numéricos para encontrar raízes de equações não lineares, experimentos didáticos sobre convergência e aplicações em problemas de engenharia, termodinâmica, finanças e mecânica orbital.

O foco dos códigos é comparar três métodos:

- **Bissecção:** robusto quando existe mudança de sinal no intervalo inicial;
- **Newton-Raphson:** rápido quando a derivada está disponível e o chute inicial é adequado;
- **Secante:** evita o cálculo da derivada e utiliza duas aproximações iniciais.

## Estrutura do projeto

| Arquivo | Conteúdo |
| --- | --- |
| `metodos.py` | Biblioteca com bissecção, Newton-Raphson, secante e contagem de avaliações de função. |
| `parte2_didaticos.py` | Experimentos sobre isolamento de raízes, custo computacional, ordem de convergência e falhas dos métodos. |
| `parte3_problemas.py` | Aplicação dos métodos aos problemas A a F. Também gera gráficos em PNG. |
| `relatorio.tex` | Relatório acadêmico em LaTeX com a fundamentação e a discussão dos resultados. |
| `requirements.txt` | Dependências Python usadas nas tabelas e visualizações. |

## Métodos implementados

Todas as rotinas principais retornam uma tupla no formato:

```python
raiz, historico = metodo(...)
```

`raiz` é a última aproximação calculada. `historico` é uma lista de dicionários, um para cada iteração, que permite analisar a convergência e o custo do método.

### Bissecção

```python
bisseccao(f, a, b, eps=1e-8, max_iter=200)
```

Recebe uma função contínua e um intervalo `[a, b]` cujos extremos devem ter sinais opostos. A cada iteração, reduz o intervalo pela metade. Se não houver mudança de sinal, a função lança `ValueError`.

O histórico contém `k`, `x`, `fx`, `erro` e `avaliacoes_f`.

### Newton-Raphson

```python
newton(f, df, x0, eps=1e-8, max_iter=200)
```

Recebe a função, sua derivada e um chute inicial. O método utiliza:

```text
x(k+1) = x(k) - f(x(k)) / f'(x(k))
```

O histórico também registra `avaliacoes_df`. Uma derivada nula ou muito próxima de zero interrompe o método com `ValueError`.

### Secante

```python
secante(f, x0, x1, eps=1e-8, max_iter=200)
```

Utiliza duas aproximações iniciais e substitui a derivada pela inclinação da secante. O histórico inclui o campo `convergiu`, além das informações de iteração, aproximação, resíduo, erro e avaliações de `f`.

### Critério de parada

As implementações encerram quando uma destas condições é atendida:

```text
|f(x)| < eps
```

ou quando o erro estimado — metade do intervalo na bissecção e diferença entre aproximações nos demais métodos — fica abaixo de `eps`. Caso `max_iter` seja atingido, a última aproximação e o histórico ainda são devolvidos, acompanhados de um aviso no terminal.

## Exemplo de uso

O exemplo abaixo calcula a raiz de `f(x) = x³ - 9x + 3` localizada entre 0 e 1:

```python
from metodos import bisseccao, newton, secante

f = lambda x: x**3 - 9*x + 3
df = lambda x: 3*x**2 - 9

raiz_b, hist_b = bisseccao(f, 0, 1)
raiz_n, hist_n = newton(f, df, 0.5)
raiz_s, hist_s = secante(f, 0, 1)

print(f"Bissecção: {raiz_b:.10f} ({len(hist_b)} iterações)")
print(f"Newton:    {raiz_n:.10f} ({len(hist_n)} iterações)")
print(f"Secante:   {raiz_s:.10f} ({len(hist_s)} iterações)")
print("Último passo de Newton:", hist_n[-1])
```

Os três métodos encontram uma raiz próxima de `0.337608956`.

## Conteúdo dos experimentos didáticos

O arquivo `parte2_didaticos.py` explora:

1. **Isolamento por tabelamento:** identifica subintervalos com mudança de sinal e mostra como uma malha grosseira pode ocultar raízes próximas;
2. **Previsão da bissecção:** compara o número teórico de iterações com o número realmente executado para diferentes tolerâncias;
3. **Custo computacional:** compara iterações e avaliações de função dos três métodos;
4. **Ordem empírica de convergência:** verifica a convergência quadrática de Newton e a ordem aproximada da secante;
5. **Falhas de Newton:** apresenta ciclo periódico, divergência causada por chutes ruins e derivada próxima de zero;
6. **Raízes múltiplas:** compara Newton padrão com Newton modificado pela multiplicidade conhecida;
7. **Armadilha do resíduo:** mostra que um resíduo pequeno nem sempre significa uma aproximação próxima da raiz.

## Problemas aplicados

O arquivo `parte3_problemas.py` organiza cada aplicação em dedução da equação, isolamento da raiz, escolha do método, resultado e verificação:

| Problema | Aplicação | Principal resultado estudado |
| --- | --- | --- |
| A | Reservatório esférico | Altura do líquido para um volume especificado e análise das raízes sem significado físico. |
| B | Perda de carga em tubulação | Solução da equação de Colebrook, fator de atrito e perda de carga. |
| C | Equação de van der Waals | Volume molar do CO₂ e comparação com o modelo de gás ideal. |
| D | Taxa Interna de Retorno | Cálculo da TIR, decisão por custo de capital e ocorrência de múltiplas TIRs. |
| E | Equação de Kepler | Anomalia excêntrica e influência da excentricidade e do chute inicial em Newton. |
| F | Deflexão de viga (bônus) | Posição e valor da deflexão máxima, com análise de derivação numérica. |

A execução completa produz os seguintes gráficos no diretório do projeto:

- `problema_A_curva_hV.png`;
- `problema_C_isoterma.png`;
- `problema_D_vpl_projeto1.png`;
- `problema_D_vpl_projeto2.png`;
- `problema_F_erro_vs_h.png`.

## Instalação

É recomendado utilizar um ambiente virtual com Python 3:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

As principais dependências diretas são NumPy, pandas e Matplotlib.

## Execução

Execute os comandos a partir da raiz do repositório:

```bash
# Autoteste dos três métodos
python3 metodos.py

# Exercícios e análises didáticas da Parte 2
python3 parte2_didaticos.py

# Problemas aplicados da Parte 3 e geração dos gráficos
python3 parte3_problemas.py
```

Os programas apresentam no terminal as aproximações, os resíduos, as quantidades de iterações e as discussões numéricas de cada caso.

## Observações numéricas

- A bissecção precisa de um intervalo com mudança genuína de sinal; raízes de multiplicidade par podem não ser detectadas dessa forma.
- Newton-Raphson pode divergir ou entrar em ciclos quando o chute inicial é inadequado.
- A secante pode parar se duas avaliações consecutivas produzirem o mesmo valor e anularem o denominador.
- Um valor pequeno de `|f(x)|` não garante, sozinho, erro pequeno em `x`, especialmente próximo de raízes múltiplas.
- O histórico retornado deve ser usado para avaliar convergência e custo, não apenas a aproximação final.
