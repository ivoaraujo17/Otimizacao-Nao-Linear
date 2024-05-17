import numpy as np
import sympy as sp


def print_passo(k, x_mais_td, t, funcao_x_mais_td, f_x_mais_eta1_t_f_linha_x_d, 
                grad_f_x_mais_td, np_dot_grad_f_x_mais_td_com_d, eta2_f_linha_x_d):
    print(f"    ----------------- Iteração = {k} -----------------")
    print(f"    t = {t}")
    print(f"    x + t*d = {x_mais_td}")
    print(f"    f(x + t*d) = {funcao_x_mais_td}")
    print(f"    f(x_) + eta*t*f'(x)*d = {f_x_mais_eta1_t_f_linha_x_d}")
    print(f"    f'(x + t*d) = {grad_f_x_mais_td}")
    print(f"    f'(x + t*d)*d = {np_dot_grad_f_x_mais_td_com_d}")
    print(f"    n2*f'(x)*d = {eta2_f_linha_x_d}")



def busca_wolfe(f, var, x, d, t_inicial, lambda_=0.8, eta1=1e-2, eta2=1e-1, verbose=True):
    """
    Realiza a busca de passo pela condição de Wolfe.

    Parâmetros:
    - f: A função que estamos otimizando (em formato sympy).
    - var: A lista de variáveis da função (sympy.Symbol).
    - x: O ponto inicial para a busca (np.array).
    - d: A direção de descida a ser seguida (np.array).
    - t_inicial: O tamanho inicial do passo (float, 0 < t_inicial < 1).
    - lambda_: O fator de redução do passo (float, 0 < lambda_ < 1).
    - eta1: Parâmetro de aceitação para a condição de Wolfe (float, 0 < eta1 < 1).
    - eta2: Parâmetro de curvatura para a condição de Wolfe (float, 0 < eta2 < 1).
    - verbose: Flag para imprimir informações de depuração (bool).

    Retorna:
    - t: O tamanho do passo encontrado pela busca de linha.
    """

    # Verifica os tipos dos parâmetros de entrada
    if not isinstance(f, sp.Expr):
        raise TypeError('A função deve ser uma expressão simbólica (sympy.Expr).')
    if not all(isinstance(v, sp.Symbol) for v in var):
        raise TypeError('As variáveis devem ser do tipo simbólico (sympy.Symbol).')
    if not isinstance(x, np.ndarray):
        raise TypeError('O ponto inicial deve ser um array numpy (np.array).')
    if not isinstance(d, np.ndarray):
        raise TypeError('A direção deve ser um array numpy (np.array).')
    if not isinstance(t_inicial, (int, float)):
        raise TypeError('O tamanho inicial do passo deve ser um número.')
    if not isinstance(lambda_, float) or not (0 < lambda_ < 1):
        raise TypeError('O fator de redução do passo deve ser um número entre 0 e 1.')
    if not isinstance(eta1, float) or not (0 < eta1 < 1):
        raise TypeError('O parâmetro de aceitação eta1 deve ser um número entre 0 e 1.')
    if not isinstance(eta2, float) or not (0 < eta2 < 1) or eta2 < eta1:
        raise TypeError('O parâmetro de curvatura eta2 deve ser um número entre 0 e 1 e maior que eta1.')
    if not isinstance(verbose, bool):
        raise TypeError('O parâmetro verbose deve ser um booleano.')
    
    # Calcula o valor da funcao no ponto inicial
    f_ponto_inicial = f.subs(dict(zip(var, x)))

    # Calcula o gradiente da funcao e aplica no ponto inicial
    gradiente_f = [sp.diff(f, v) for v in var]
    gradiente_f_ponto_inicial = [g.subs(dict(zip(var, x))) for g in gradiente_f]

    # Calcula o gradiente da f no ponto inicial produto interno com a direção
    gradiente_f_ponto_inicial_d = np.dot(gradiente_f_ponto_inicial, d)

    # Define o ponto x + t*d
    x_mais_td = x + t_inicial*d

    # Aplica o ponto x + t*d na funcao
    f_x_mais_td = f.subs(dict(zip(var, x_mais_td)))

    # Aplica o ponto x + t*d no gradiente
    gradiente_f_x_mais_td = [g.subs(dict(zip(var, x_mais_td))) for g in gradiente_f]

    t = t_inicial
    k = 0
    if verbose:
        print(f"    ----------------- Busca Linear Wolfe -----------------")
        print(f"    Parametros:")
        print(f"    f = {f}")
        print(f"    x = {x}")
        print(f"    d = {d}")
        print(f"    t inicial = {t_inicial}")
        print(f"    lambda = {lambda_}")
        print(f"    eta1 = {eta1}")
        print(f"    eta2 = {eta2}")
        print(f"    ----------------- Valores Constantes -----------------")
        print(f"    f(x) = {f_ponto_inicial}")
        print(f"    f' = {gradiente_f}")
        print(f"    f'(x) = {gradiente_f_ponto_inicial}")
        print(f"    f'(x)*d = {gradiente_f_ponto_inicial_d}")
        print_passo(k, x_mais_td, t, f_x_mais_td, f_ponto_inicial + eta1*t*gradiente_f_ponto_inicial_d,
                    gradiente_f_x_mais_td, np.dot(gradiente_f_x_mais_td, d), eta2*gradiente_f_ponto_inicial_d)

    while f_x_mais_td > f_ponto_inicial + eta1*t*gradiente_f_ponto_inicial_d or \
        np.dot(gradiente_f_x_mais_td,d) < eta2*gradiente_f_ponto_inicial_d:
        # Atualiza o tamanho do passo
        t = lambda_*t
        # Atualiza o ponto x + t*d
        x_mais_td = x + t*d
        # Aplica o novo ponto x + t*d na funcao
        f_x_mais_td = f.subs(dict(zip(var, x_mais_td)))
        # Aplica o novo ponto x + t*d no gradiente
        gradiente_f_x_mais_td = [g.subs(dict(zip(var, x_mais_td))) for g in gradiente_f]
        # Atualiza o número de iterações
        k += 1
        if verbose:
            print_passo(k, x_mais_td, t, f_x_mais_td, f_ponto_inicial + eta1*t*gradiente_f_ponto_inicial_d,
                        gradiente_f_x_mais_td, np.dot(gradiente_f_x_mais_td, d), eta2*gradiente_f_ponto_inicial_d)
        # Verifica se o tamanho do passo é muito pequeno
        if t < 1e-8:
            raise ValueError("Erro de Bracketing.")
    
    return t


if __name__ == "__main__":
    # Teste da busca de wolfe
    x1, x2 = sp.symbols('x1 x2')
    f = x1**2 + x1*x2 + x2**2
    var = [x1, x2]
    x = np.array([1, 2])
    d = np.array([-1, -1])
    t_inicial = 10
    lambda_ = 0.5
    eta1 = 1e-4
    eta2 = 0.9
    verbose = True

    resultado = busca_wolfe(f, var, x, d, t_inicial, lambda_, eta1, eta2, verbose)
    print(resultado)



