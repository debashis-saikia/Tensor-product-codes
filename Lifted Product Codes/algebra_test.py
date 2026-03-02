import numpy as np
from algebra import (
    FiniteGroup,
    RingElement,
    RingMatrix,
    GroupAlgebraElement,
    GroupAlgebraMatrix,
    RingLifter,
    lift_to_binary
)
# ordering:
# 0:e
# 1:(12)
# 2:(13)
# 3:(23)
# 4:(123)
# 5:(132)
S3_table = [
    [0,1,2,3,4,5],
    [1,0,5,4,3,2],
    [2,4,0,5,1,3],
    [3,5,4,0,2,1],
    [4,2,3,1,5,0],
    [5,3,1,2,0,4],
]
G = FiniteGroup(S3_table)
# generators
s = GroupAlgebraElement([0,1,0,0,0,0], G)
t = GroupAlgebraElement([0,0,0,0,1,0], G)
print("\nChecking non-commutativity:")
print("s*t =", s * t)
print("t*s =", t * s)
assert (s * t).coeffs.tolist() != (t * s).coeffs.tolist()
print("Group is non-abelian")
x = RingElement([1,0,0])
zero = RingElement([0,0,0])
# A B = 0 structure
A = RingMatrix([[x, zero]])
B = RingMatrix([[zero],[x]])
print("\nAbstract AB=0:", (A @ B).is_zero())
st = s * t
generator_images = [s, t, st]
lifter = RingLifter(generator_images)
GA = GroupAlgebraMatrix([
    [lifter.lift(e) for e in row]
    for row in A.data
])
GB = GroupAlgebraMatrix([
    [lifter.lift(e) for e in row]
    for row in B.data
])
print("\nTesting sided multiplication")
C = GA.matmul(GB, mode="right-left")
print("GA ⊗ GB zero?:", C.is_zero())
HA = lift_to_binary(GA, side="right")
HB = lift_to_binary(GB, side="left")
print("\nBinary matrix sizes")
print("HA:", HA.shape)
print("HB:", HB.shape)
css = (HA @ HB) % 2
print("\nCSS condition satisfied:",
      np.all(css == 0))