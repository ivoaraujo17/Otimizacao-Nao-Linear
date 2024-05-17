import numpy as np
import sympy as sp
from busca_exata import busca_exata


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


if __name__ == "__main__":
    x, y = sp.symbols('x y')
    f = (4-x)**2 + (2-y)**2
    var = [x, y]
    x0 = np.array([2, 3])
    tol = 1e-4
    max_iter = 1000

    grad = Gradiente(f, var, x0, max_iter)
    x_min = grad.gradiente_busca_exata()
    print(f"O ponto que minimiza a função é: {x_min}")