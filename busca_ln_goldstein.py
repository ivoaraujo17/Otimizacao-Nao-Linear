import numpy as np
import sympy as sp


def print_passo(k, x_mais_td, t, fx_mais_um_menos_eta_t_f_linha_x_d, fx_mais_td, fx_mais_eta_t_f_linha_x_d):
    print(f"    ----------------- Iteração = {k} -----------------")
    print(f"    t = {t}")
    print(f"    x + t*d = {x_mais_td}")
    print(f"    f(x) + (1 - eta)*t*f'(x)*d = {fx_mais_um_menos_eta_t_f_linha_x_d}")
    print(f"    f(x + t*d) = {fx_mais_td}")
    print(f"    f(x) + eta*t*f'(x)*d = {fx_mais_eta_t_f_linha_x_d}")


def busca_goldstein(f, vars, x, d, t, lambda_=0.8, eta=0.1, verbose=True):
    """
    Função que realiza a busca linear de Goldstein
    :param f: função a ser minimizada (sympy.Expr)
    :param vars: lista de variáveis (sympy.Symbol)
    :param x: ponto inicial (np.array)
    :param d: direção de descida (np.array)
    :param t: tamanho inicial do passo (float 0 < t < 1)
    :param lambda_: fator de redução do passo (float 0 < lambda_ < 1)
    :param eta1: parâmetro de aceitação (float 0 < eta1 < 1)

    :return: [True, t, k] se o critério de Goldstein foi satisfeito, onde t é o tamanho do passo e k é o número de iterações
                [False, t, k] caso contrário
    """
    # verifica os tipos dos parametros de entrada
    if not isinstance(f, sp.Expr):
        raise TypeError('A função deve ser uma expressão simbólica (sympy.Expr).')
    if not all(isinstance(v, sp.Symbol) for v in vars):
        raise TypeError('As variáveis devem ser do tipo simbólico (sympy.Symbol).')
    if not isinstance(x, np.ndarray):
        raise TypeError('O ponto inicial deve ser um array numpy (np.array).')
    if not isinstance(d, np.ndarray):
        raise TypeError('A direção deve ser um array numpy (np.array).')
    if not isinstance(t, (int, float)):
        raise TypeError('O tamanho inicial do passo deve ser um número.')
    if not isinstance(lambda_, float) or not (0 < lambda_ < 1):
        raise TypeError('O fator de redução do passo deve ser um número entre 0 e 1.')
    if not isinstance(eta, float) or not (0 < eta < 1):
        raise TypeError('O parâmetro de aceitação eta deve ser um número entre 0 e 1.')

    # Calcula o valor da funcao no ponto inicial
    fx = f.subs(dict(zip(vars, x)))

    # Calcula o gradiente da funcao e aplica no ponto inicial
    grad_f = [sp.diff(f, v) for v in vars]
    grad_fx = [g.subs(dict(zip(vars, x))) for g in grad_f]

    # Calcula o gradiente da f no ponto inicial produto interno com a direção
    grad_fx_d = np.dot(grad_fx, d)

    # Define o ponto x + t*d
    x_mais_td = x + t*d

    # Aplica o ponto x + t*d na funcao
    f_x_mais_td = f.subs(dict(zip(vars, x_mais_td)))
    k = 0

    if verbose:
        print(f"    ----------------- Busca Linear Goldstein ---------------")
        print(f"    f = {f}")
        print(f"    x = {x}")
        print(f"    d = {d}")
        print(f"    t inicial = {t}")
        print(f"    lambda = {lambda_}")
        print(f"    eta = {eta}")
        print(f"    ----------------- Valores Constantes -----------------")
        print(f"    f(x) = {fx}")
        print(f"    f' = {grad_f}")
        print(f"    f'(x) = {grad_fx}")
        print(f"    f'(x)*d = {grad_fx_d}")
        print_passo(k, x_mais_td, t, fx + (1-eta)*t*grad_fx_d, f_x_mais_td, fx + eta*t*grad_fx_d)

    while (fx + (1-eta)*t*grad_fx_d > f_x_mais_td) or (f_x_mais_td > fx + eta*t*grad_fx_d):
        t = lambda_*t
        x_mais_td = x + t*d
        f_x_mais_td = f.subs(dict(zip(vars, x_mais_td)))
        k += 1

        if verbose:
            print_passo(k, x_mais_td, t, fx + (1-eta)*t*grad_fx_d, f_x_mais_td, fx + eta*t*grad_fx_d)

        if t < 1e-10:
            return [False, t, k]

    return [True, t, k]        


if __name__ == '__main__':
    # Teste da busca linear de Goldstein
    x1, x2 = sp.symbols('x1 x2')
    f = x1**2 + x1*x2 + x2**2
    vars = [x1, x2]
    x = np.array([1, 2])
    d = np.array([-1, -1])
    t = 10
    lambda_ = 0.5
    eta = 1e-4

    print(busca_goldstein(f, vars, x, d, t, lambda_, eta))
