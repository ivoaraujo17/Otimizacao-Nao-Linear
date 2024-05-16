import unittest
import numpy as np
import sympy as sp
from busca_exata import busca_exata

class TestBuscaExata(unittest.TestCase):

    def test_funcao_quadratica(self):
        x, y = sp.symbols('x y')
        func = x**2 + y**2
        vars = [x, y]
        ponto_ini = np.array([1, 1])
        direcao = np.array([-1, -1])
        
        t_val = busca_exata(func, vars, ponto_ini, direcao, verbose=False)
        expected_t_val = 1  # Esperamos que t_val seja 1, pois (1,1) -> (0,0) ao longo da direção (-1,-1)
        self.assertAlmostEqual(t_val, expected_t_val, places=5)

    def test_funcao_linear(self):
        x = sp.Symbol('x')
        func = 3*x
        vars = [x]
        ponto_ini = np.array([2])
        direcao = np.array([-1])
        
        t_val = busca_exata(func, vars, ponto_ini, direcao, verbose=False)
        expected_t_val = 0  # Esperamos que t_val seja 0, pois a função é linear e não tem mínimo na direção negativa
        self.assertEqual(t_val, expected_t_val)

    def test_funcao_cubica(self):
        x = sp.Symbol('x')
        func = x**3 - 3*x**2 + 2*x
        vars = [x]
        ponto_ini = np.array([0])
        direcao = np.array([1])
        
        t_val = busca_exata(func, vars, ponto_ini, direcao, verbose=False)
        expected_t_val = 0  # O ponto de mínimo na direção positiva a partir de x=0
        self.assertAlmostEqual(t_val, expected_t_val, places=5)

    def test_t_val_none(self):
        x = sp.Symbol('x')
        func = x**3
        vars = [x]
        ponto_ini = np.array([0])
        direcao = np.array([0])
        
        t_val = busca_exata(func, vars, ponto_ini, direcao, verbose=False)
        self.assertIsNone(t_val)

if __name__ == '__main__':
    unittest.main()
