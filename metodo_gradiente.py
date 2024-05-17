import numpy as np
import sympy as sp
from busca_exata import busca_exata
from busca_ln_armijo import busca_linear_armijo
from busca_ln_wolfe import busca_wolfe
from busca_ln_goldstein import busca_goldstein


class Gradiente:
    def __init__(self, f, var, x0, max_iter, tol=1e-4):
        self.f = f
        self.var = var
        self.x0 = x0
        self.max_iter = max_iter
        self.tolerancia = tol
    
    def gradiente_busca_exata(self):
        f = self.f
        xk = self.x0
        tk = 0
        k = 0

        grad_f = [sp.diff(f, var) for var in self.var]
        grad_f_xk = np.array([float(grad.subs(list(zip(self.var, xk)))) for grad in grad_f])

        while np.linalg.norm(grad_f_xk) > self.tolerancia:
            k += 1
            d = np.array([-grad for grad in grad_f_xk])
            tk = busca_exata(f, self.var, xk, d, verbose=False)
            xk = xk + tk * d
            grad_f_xk = np.array([float(grad.subs(list(zip(self.var, xk)))) for grad in grad_f])
            if k > self.max_iter:
                raise ValueError("Número máximo de iterações atingido.")
                break
        return xk
    
    def gradiente_armijo(self, t0, lambda_, eta, verbose=False):
        """
        Método do gradiente com busca linear de Armijo.

        params: t0: tamanho inicial do passo (int,float)
                lambda_: fator de redução do passo (float: (0,1))
                eta: fator de velocidade de descida (float: (0,1)) 
        """
        f = self.f # Função a ser minimizada
        xk = self.x0 # Ponto inicial
        tk = t0 # Tamanho inicial do passo
        k = 0 # Número de iterações
        # Calcula o gradiente da função e aplica no ponto inicial x0
        grad_f = [sp.diff(f, var) for var in self.var]
        grad_f_xk = np.array([float(grad.subs(list(zip(self.var, xk)))) for grad in grad_f])

        while np.linalg.norm(grad_f_xk) > self.tolerancia:
            k += 1
            d = np.array([-grad for grad in grad_f_xk])
            tk = busca_linear_armijo(f, self.var, xk, d, tk, eta, lambda_, verbose=False)
            if tk[0] == False:
                raise ValueError("Erro no Backtracking.")
                break
            tk = tk[1]
            xk = xk + tk * d
            grad_f_xk = np.array([float(grad.subs(list(zip(self.var, xk)))) for grad in grad_f])
            if k > self.max_iter:
                raise ValueError("Número máximo de iterações atingido.")
                break
        return xk
    
    def gradiente_wolfe(self, t0, lambda_=0.8, eta1=1e-2, eta2=1e-1, verbose=False):
        f = self.f
        xk = self.x0
        tk = t0
        k = 0

        grad_f = [sp.diff(f, var) for var in self.var]
        grad_f_xk = np.array([float(grad.subs(list(zip(self.var, xk)))) for grad in grad_f])

        while np.linalg.norm(grad_f_xk) > self.tolerancia:
            k += 1
            d = np.array([-grad for grad in grad_f_xk])
            tk = busca_wolfe(f, self.var, xk, d, tk, lambda_, eta1, eta2, verbose=False)
            if tk[0] == False:
                raise ValueError("Erro no Backtracking.")
                break
            tk = tk[1]
            xk = xk + tk * d
            grad_f_xk = np.array([float(grad.subs(list(zip(self.var, xk)))) for grad in grad_f])
            if k > self.max_iter:
                raise ValueError("Número máximo de iterações atingido.")
                break
        return xk
    
    def gradiente_goldstein(self, t0, lambda_=0.8, eta=0.1, verbose=False):
        f = self.f
        xk = self.x0
        tk = t0
        k = 0

        grad_f = [sp.diff(f, var) for var in self.var]
        grad_f_xk = np.array([float(grad.subs(list(zip(self.var, xk)))) for grad in grad_f])

        while np.linalg.norm(grad_f_xk) > self.tolerancia:
            k += 1
            d = np.array([-grad for grad in grad_f_xk])
            tk = busca_goldstein(f, self.var, xk, d, tk, lambda_, eta, verbose=False)
            if tk[0] == False:
                raise ValueError("Erro no Backtracking.")
                break
            tk = tk[1]
            xk = xk + tk * d
            grad_f_xk = np.array([float(grad.subs(list(zip(self.var, xk)))) for grad in grad_f])
            if k > self.max_iter:
                raise ValueError("Número máximo de iterações atingido.")
                break
        return xk


if __name__ == "__main__":
    x, y = sp.symbols('x y')
    f = (4-x)**2 + (2-y)**2
    var = [x, y]
    x0 = np.array([2, 3])
    tol = 1e-4
    max_iter = 1000

    grad = Gradiente(f, var, x0, max_iter)
    x_min = grad.gradiente_busca_exata()
    print(f"x* (busca exata): {x_min}")

    x_min = grad.gradiente_armijo(1, 0.8, 1/4)
    print(f"x* (armijo): {x_min}")

    x_min = grad.gradiente_wolfe(1, 0.8, 1/4, 1/2)
    print(f"x* (wolfe): {x_min}")

    x_min = grad.gradiente_goldstein(1, 0.8, 1/4)
    print(f"x* (goldstein): {x_min}")