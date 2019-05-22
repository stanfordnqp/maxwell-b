import numpy as np
import unittest
import bicg

n = 10
A0 = np.random.randn(n, n)
b0 = np.random.randn(n)

class TestBicg(unittest.TestCase):
    def test_asymm(self):
        A = A0
        b = b0

        def multA(x, y):
            y[:] = np.dot(A, x)

        def multAT(x, y):
            y[:] = np.dot(A.T, x)

        ops = {'multA': multA, 'multAT': multAT}

        x, err, success = bicg.solve_asymm(b, **ops)
        self.assertTrue(success)

    def test_symm(self):
        A = np.dot(A0.T, A0) # Make A symmetric.
        b = b0

        def multA(x, y):
            y[:] = np.dot(A, x)

        ops = {'multA': multA}

        x, err, success = bicg.solve_symm(b, **ops)
        self.assertTrue(success)

    def test_zlumped(self):
        A = np.dot(A0.T, A0) # Make A symmetric.
        b = b0

        def alpha_step(rho_k, rho_k_1, p, r, v):
            p[:] = r + (rho_k / rho_k_1) * p
            v[:] = np.dot(A, p)
            return rho_k / np.dot(p, v) # Return alpha.

        def rho_step(alpha, p, r, v, x):
            x[:] = x + alpha * p
            r[:] = r - alpha * v

            # Return rho and err.
            return np.dot(r, r), np.sqrt(np.dot(np.conj(r), r)) 

        def zeros():
            return np.zeros_like(b)
        ops = {'rho_step': rho_step, 'alpha_step': alpha_step, 'zeros': zeros}

        x, err, success = bicg.solve_symm_lumped(b, **ops)
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()
        
        
        

