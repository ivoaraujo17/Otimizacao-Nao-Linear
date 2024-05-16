import numpy as np
import sympy as sp

def busca_exata(funcao, variaveis, ponto_inicial, direcao, verbose=True):
    """
    Executa a busca exata em uma direcao para encontrar o valor de t que minimiza
    f(x + t*d).
    
    Args:
    funcao: sympy expression
        Funcao a ser minimizada.
    variaveis: list of sympy.Symbol
        Variaveis da funcao.
    ponto_inicial: numpy array
        Ponto inicial x_.
    direcao: numpy array
        Direcao a ser executada a busca exata.
    verbose: bool
        Se True, imprime os resultados parciais.
    
    Returns:
    t_val: float
        Valor de t que minimiza f(x + t*d).
    """

    # Verificações de tipos da entrada
    if not isinstance(funcao, sp.Expr):
        raise TypeError("funcao deve ser do tipo sympy.Expr")
    if not all(isinstance(variavel, sp.Symbol) for variavel in variaveis):
        raise TypeError("variaveis deve ser do tipo sympy.Symbol")
    if not isinstance(ponto_inicial, np.ndarray):
        raise TypeError("ponto_inicial deve ser do tipo numpy.ndarray")
    if not isinstance(direcao, np.ndarray):
        raise TypeError("direcao deve ser do tipo numpy.ndarray")
    if not isinstance(verbose, bool):
        raise TypeError("verbose deve ser do tipo bool")

    # Verifica se a direção é zero
    if np.all(direcao == 0):
        return None

    # Calcula a derivada de f em relação às variáveis
    derivada_funcao = [sp.diff(funcao, var) for var in variaveis]

    # Avalia a derivada no ponto inicial
    derivada_funcao_ponto_inicial = [derivada.subs(dict(zip(variaveis, ponto_inicial))) for derivada in derivada_funcao]

    # Produto escalar da derivada no ponto inicial com a direção
    derivada_funcao_ponto_inicial_direcao = np.dot(derivada_funcao_ponto_inicial, direcao)
    if derivada_funcao_ponto_inicial_direcao >= 0:
        return 0

    # Define o novo ponto x* = x_ + t*d
    t = sp.Symbol('t')
    novo_ponto_x_mais_td = ponto_inicial + t * direcao

    # Substitui as variáveis pelo novo ponto na função
    funcao_aplicada_novo_ponto = funcao.subs(dict(zip(variaveis, novo_ponto_x_mais_td)))

    # Derivada da função aplicada ao novo ponto em relação a t
    derivada_funcao_novo_ponto = sp.diff(funcao_aplicada_novo_ponto, t)

    # Resolve a derivada para encontrar o valor de t que minimiza f(x + t*d)
    t_val = sp.solve(derivada_funcao_novo_ponto, t)
    t_val = t_val[0] if t_val else 0  # Pega a primeira solução, se existir, ou 0 se não houver solução

    if verbose:
        print(f"         f = {funcao}")
        print(f"        f' = {derivada_funcao}")
        print(f"    f'(x_) = {derivada_funcao_ponto_inicial}")
        print(f"f'(x_) * d = {derivada_funcao_ponto_inicial_direcao}")
        print(f"  x_ + t*d = {novo_ponto_x_mais_td}")
        print(f"     f(x*) = {funcao_aplicada_novo_ponto}")
        print(f"    f'(x*) = {derivada_funcao_novo_ponto}")
        print(f"         t = {t_val}")

    return t_val

# Exemplo de uso:
if __name__ == "__main__":
    x, y = sp.symbols('x y')
    func = x**2 + y**2
    vars = [x, y]
    ponto_ini = np.array([1, 1])
    direcao = np.array([-1, -1])

    t_min = busca_exata(func, vars, ponto_ini, direcao)
    print(f"O valor de t que minimiza f(x + t*d) é: {t_min}")
