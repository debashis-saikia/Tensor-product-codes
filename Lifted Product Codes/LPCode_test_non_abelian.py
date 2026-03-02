import numpy as np
from algebra import (
    FiniteGroup,
    RingElement,
    RingMatrix,
    GroupAlgebraElement
)
from LPCode import LPC

S3_table = [
    [0,1,2,3,4,5],
    [1,0,5,4,3,2],
    [2,4,0,5,1,3],
    [3,5,4,0,2,1],
    [4,2,3,1,5,0],
    [5,3,1,2,0,4],
]

G = FiniteGroup(S3_table)

s = GroupAlgebraElement([0,1,0,0,0,0], G)
t = GroupAlgebraElement([0,0,0,0,1,0], G)

print("Checking non-commutativity")
print("s*t =", s*t)
print("t*s =", t*s)

if (s*t).coeffs.tolist() != (t*s).coeffs.tolist():
    print("Non-abelian group.")
else:
    print("Abelian Group.")


x = RingElement([1,0,0])
zero = RingElement([0,0,0])

A = RingMatrix([[x, zero]])
B = RingMatrix([[zero], [x]])

print("Abstract AB = 0 :", (A @ B).is_zero())

generator_images = [s, t, s*t]


print("\nConstructing Lifted Product Code...")

code = LPC(A, B, generator_images)

H_X, H_Z = code.parity_checks()

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
print("\nRate k/n =", rate)
