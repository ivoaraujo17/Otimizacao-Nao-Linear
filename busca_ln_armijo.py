import numpy as np
import sympy as sp


def print_passo(k, x_mais_td, t, funcao_x_mais_td, f_x_mais_eta_t_f_linha_x_d):
    print(f"\n----------------- Iteração = {k} -----------------")
    print(f"t = {t}")
    print(f"x + t*d = {x_mais_td}")
    print(f"f(x + t*d) = {funcao_x_mais_td}")
    print(f"f(x) + eta*t*f'(x)*d = {f_x_mais_eta_t_f_linha_x_d}")



def busca_linear_armijo(funcao, variaveis, ponto_inicial, direcao, t_inicial, lamda=0.8, eta=0.1, verbose=True):
    """
    Busca linear com critério de Armijo para encontrar o tamanho do passo t
    :param f: função a ser minimizada
    :param variaveis: lista de variáveis
    :param ponto_inicial: ponto inicial
    :param direcao: direção de descida
    :param t_inicial: tamanho inicial do passo
    :param lamda: fator de redução do passo
    :param eta: fator de velocidade de descida

    :return: [True, t, k] se o critério de Armijo foi satisfeito, onde t é o tamanho do passo e k é o número de iterações
                [False, t, k] caso contrário
    """

    # verifica os tipos dos parametros de entrada
    if not isinstance(funcao, sp.Expr):
        raise TypeError('A função deve ser uma expressão simbólica')
    if not all(isinstance(var, sp.Symbol) for var in variaveis):
        raise TypeError('As variáveis devem ser do tipo simbólico')
    if not isinstance(ponto_inicial, np.ndarray):
        raise TypeError('O ponto inicial deve ser um array numpy')
    if not isinstance(direcao, np.ndarray):
        raise TypeError('A direção deve ser um array numpy')
    if not isinstance(t_inicial, (int, float)):
        raise TypeError('O tamanho inicial do passo deve ser um número')
    if not isinstance(lamda, float) and 0 < lamda < 1:
        raise TypeError('O fator de redução do passo deve ser um número entre 0 e 1')
    if not isinstance(eta, float) and 0 < eta < 1:
        raise TypeError('O fator de velocidade de descida deve ser um número entre 0 e 1')

    # Calcula o valor da função e do gradiente no ponto inicial
    funcao_ponto_inicial = funcao.subs(dict(zip(variaveis, ponto_inicial)))

    # Calcula o gradiente da função e aplica no ponto inicial
    derivada_funcao = [sp.diff(funcao, var) for var in variaveis]
    derivada_funcao_ponto_inicial = [derivada.subs(dict(zip(variaveis, ponto_inicial))) for derivada in derivada_funcao]
    derivada_funcao_ponto_inicial_d = np.dot(derivada_funcao_ponto_inicial, direcao)
    # define o ponto x + t*d
    x_mais_td = ponto_inicial + t_inicial*direcao
    
    # Aplica o ponto x + t*d na função
    funcao_x_mais_td = funcao.subs(dict(zip(variaveis, x_mais_td)))
    
    t = t_inicial
    k=0
    if verbose:
        print("\n----------------- Busca Linear Armijo -----------------")
        print(f"f = {funcao}")
        print(f"x = {ponto_inicial}")
        print(f"d = {direcao}")
        print(f"t inicial = {t_inicial}")
        print(f"lambda = {lamda}")
        print(f"eta = {eta}")
        print("-----------------")
        print(f"f(x) = {funcao_ponto_inicial}")
        print(f"f' = {derivada_funcao}")
        print(f"f'(x) = {derivada_funcao_ponto_inicial}")
        print(f"f'(x)*d = {derivada_funcao_ponto_inicial_d}")
        print_passo(k, x_mais_td, t, funcao_x_mais_td, funcao_ponto_inicial + eta*t*np.dot(derivada_funcao_ponto_inicial, direcao))
    
    while funcao_x_mais_td > funcao_ponto_inicial + eta*t*np.dot(derivada_funcao_ponto_inicial, direcao):
        # Atualiza o valor de t
        t *= lamda

        # Atualiza o ponto x + t*d
        x_mais_td = ponto_inicial + t*direcao
        
        # Aplica o ponto x + t*d na função
        funcao_x_mais_td = funcao.subs(dict(zip(variaveis, x_mais_td)))
        
        # Atualiza o número de iterações e verifica se o tamanho do passo é muito pequeno
        k += 1
        if verbose:
            print_passo(k, x_mais_td, t, funcao_x_mais_td, funcao_ponto_inicial + eta*t*np.dot(derivada_funcao_ponto_inicial, direcao))

        if t < 1e-8:
            print('Erro no Backtracking')
            return [False, t, k]

    return [True, t, k]


if __name__ == "__main__":
    x, y = sp.symbols('x y')
    func = 0.5*(x -2)**2 + (y-1)**2
    variaveis = [x, y]

    ponto_ini = np.array([1, 0])
    direcao = np.array([3, 1])
    t_inicial = 1

    t_min = busca_linear_armijo(func, variaveis, ponto_ini, direcao, t_inicial, eta=1/4, lamda=0.8)
    print(f"O valor de t que minimiza f(x + t*d) é: {t_min}")