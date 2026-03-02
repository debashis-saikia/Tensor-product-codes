import numpy as np
from algebra import (
    FiniteGroup,
    RingElement,
    RingMatrix,
    GroupAlgebraElement,
    cyclic_group
)
from LPCode import LPC

L = 5
G = cyclic_group(L)

print("\nTesting Abelian cyclic group C_", L)

g = GroupAlgebraElement(
    [0,1,0,0,0], G
)

g_inv = GroupAlgebraElement(
    [0,0,0,0,1], G
)

print("\nChecking commutativity")

print("g*g_inv =", g * g_inv)
print("g_inv*g =", g_inv * g)

assert (g * g_inv).coeffs.tolist() == \
       (g_inv * g).coeffs.tolist()
print("Group is Abelian")

x = RingElement([1,0,0])
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
assert np.all(css == 0)
print("CSS condition satisfied")

n = code.n

rank_X = np.linalg.matrix_rank(H_X % 2)
rank_Z = np.linalg.matrix_rank(H_Z % 2)

k = n - rank_X - rank_Z
rate = k / n

print("\nQuantum Code Parameters")
print(f"[[ n = {n}, k = {k} ]]")
print("Rate =", rate)