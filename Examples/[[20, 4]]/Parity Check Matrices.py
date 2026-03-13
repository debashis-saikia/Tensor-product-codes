from LPCode import LPC
from SyndromeCircuit import SyndromeCircuit
from algebra import (FiniteGroup, RingElement, RingMatrix, GroupAlgebraElement,cyclic_group)
import numpy as np

L = 2
G = cyclic_group(L)
g = GroupAlgebraElement([1,0], G)
g_inv = GroupAlgebraElement([0,1], G)
print("\nChecking commutativity")
assert (g * g_inv).coeffs.tolist() == (g_inv * g).coeffs.tolist()
print("Group is Abelian")
x = RingElement([1,0, 0])
zero = RingElement([0,0,0])
A = RingMatrix([[x, zero]])
B = RingMatrix([[zero],[x]])
print("\nAbstract AB = 0 :", (A @ B).is_zero())
generator_images = [g, g_inv, g*g_inv]
print("\nConstructing Lifted Product Code...")
code = LPC(A, B, generator_images)
H_X, H_Z = code.parity_checks()
print("LP construction successful")
print("\nMatrix dimensions")
print("H_X shape:", H_X.shape)
print("H_Z shape:", H_Z.shape)
css = (H_X @ H_Z.T) % 2
print("\nCSS orthogonality:",
      np.all(css == 0))
n = code.n
k = code.compute_k()
print("\nQuantum Code Parameters")
print(f"[[ n = {n}, k = {k} ]]")

print("H_X:")
print(H_X)
print("H_Z:")
print(H_Z)
#save_matrices = input("\nDo you want to save the parity check matrices? (y/n): ")
#if save_matrices.lower() == 'y':
#    code.save_parity_matrices()
#    print("Parity check matrices saved.")

