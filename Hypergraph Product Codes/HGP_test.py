from HGPCode import HGP
import numpy as np

A = np.array([[1, 1, 0, 1],
              [1, 0, 1, 1]])
B = np.array([[1, 0, 1, 1, 0, 1],
              [0, 1, 1, 0, 1, 1]])

HGP_code = HGP(A, B)
print(f"HGP Code Parameters [[n, k]]: [[{HGP_code.n}, {HGP_code.k}]]")
print("HGP Code Parity Check Matrices:")
print("H_X:")
print((HGP_code.H_X.toarray()).astype(int))
print("H_Z:")   
print((HGP_code.H_Z.toarray()).astype(int))
