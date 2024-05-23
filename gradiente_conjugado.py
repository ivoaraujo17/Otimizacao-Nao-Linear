import numpy as np
import sympy as sp


def print_passo(k, tk, xk, grad_f_xk, beta, d):
    print(f"----------- Passo {k} -----------")
    print(f"    tk = {tk}")
    print(f"    xk = {xk}")
    print(f"    Novo grad(f(xk)) = {grad_f_xk}")
    print(f"    beta = {beta}")
    print(f"    Nova direção = {d}")
    print(f"-------------------------------")


def gradiente_conjugado(f, var, x0, max_iter, tol=1e-4, verbose=False):
    """
    Método do gradiente conjugado.
    params: f: função a ser minimizada
            var: variáveis da função
            x0: ponto inicial
            max_iter: número máximo de iterações
            tol: tolerância
    """
    # Calcula o gradiente da função
    grad_f = [sp.diff(f, var) for var in var]
    # Calcula o gradiente da função no ponto x0
    grad_f_xk = np.array([float(grad.subs(list(zip(var, x0)))) for grad in grad_f])
    # Direção inicial = -gradiente da função no ponto x0
    d = -grad_f_xk
    # Calcula a matriz Hessiana da função
    A = np.array([[float(sp.diff(grad_f[i], var[j]).subs(list(zip(var, x0)))) for j in range(len(var))] for i in range(len(var))])
    # xk = ponto inicial x0
    xk = x0
    if verbose:
        print(f"----------- Gradiente Conjugado -----------")
        print(f"f = {f}")
        print(f"x0 = {x0}")
        print(f"Gradiente da função: {grad_f}")
        print(f"Gradiente no ponto x0 = {grad_f_xk}")
        print(f"Direção d0 = {d}")
        print(f"Matriz Hessiana A = {A}\n")
    k = 0
    while np.linalg.norm(grad_f_xk) > tol:
        k += 1
        # gradiente da funcao no ponto xk * direcao (parte de cima do calculo de tk)
        grad_f_xk_vezes_d = np.dot(grad_f_xk.T, d)
        # d transposto vezes A vezes d (parte de baixo do calculo de tk)
        d_transposto_A_d = np.dot(d.T, np.dot(A, d))
        # calcula o tk fazendo a divisao das duas partes
        # gradiente da funcao no ponto xk * direcao / direcao transposta vezes A vezes d
        tk = - (grad_f_xk_vezes_d / d_transposto_A_d)
        # atualiza o ponto
        xk = xk + tk * d
        # atualiza o gradiente no novo ponto
        grad_f_xk = np.array([float(grad.subs(list(zip(var, xk)))) for grad in grad_f])
        # direcao transposta vezes A vezes gradiente no ponto xk (parte de cima do calculo de beta)
        d_transposto_A_grad_f_xk = np.dot(d.T, np.dot(A, grad_f_xk))
        # direcao transposta vezes A vezes direcao (parte de baixo do calculo de beta)
        d_transposto_A_d = np.dot(d.T, np.dot(A, d))
        # calcula o beta fazendo a divisao das duas partes
        # beta = direcao transposta vezes A vezes gradiente no ponto xk / direcao transposta vezes A vezes direcao
        beta = np.dot(d.T, np.dot(A, grad_f_xk)) / np.dot(d.T, np.dot(A, d))
        # nova direcao = -gradiente no ponto xk + beta * direcao
        d = -grad_f_xk + (beta * d)

        if verbose:
            print_passo(k, tk, xk, grad_f_xk, beta, d)
        if k > max_iter:
            raise ValueError("Número máximo de iterações atingido.")
            break
    return xk, k


# Exemplo de uso
if __name__ == "__main__":
    x, y = sp.symbols('x y')
    f = (3 -x)**2 + (4-y)**2 + y**4
    #f = 2*(x**2) + 6*(y**2) + 2*x*y + 2*x + 3*y +3
    var = [x, y]
    x0 = np.array([0, 0])
    max_iter = 10000
    tol = 1e-4
    xk, k = gradiente_conjugado(f, var, x0, max_iter, tol, verbose=True)
    print(xk, k)
    print(f.subs(list(zip(var, xk))))