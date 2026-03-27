import numpy as np

class FiniteGroup:
    def __init__(self, mul_table):
        self.mul = mul_table
        self.n = len(mul_table)

def cyclic_group(L):
    table = [[(i + j) % L for j in range(L)] for i in range(L)]
    return FiniteGroup(table)

def dihedral_group(n):
    size = 2*n
    table = [[0]*size for _ in range(size)]

    for i in range(size):
        for j in range(size):
            if i < n:
                if j < n:
                    table[i][j] = (i+j)%n
                else:
                    table[i][j] = n + ((i+j-n)%n)
            else:
                if j < n:
                    table[i][j] = n + ((i-n-j)%n)
                else:
                    table[i][j] = (i-n-j)%n

    return FiniteGroup(table)

class GroupAlgebraElement:
    def __init__(self, coeffs, group):
        self.coeffs = np.array(coeffs, dtype=np.uint8) % 2
        self.group = group
        self.n = group.n

    def __add__(self, other):
        return GroupAlgebraElement(self.coeffs ^ other.coeffs, self.group)

    def right_mul(self, other):
        """ a * b """
        G = self.group
        result = np.zeros(self.n, dtype=np.uint8)
        for i, a in enumerate(self.coeffs):
            if a:
                for j, b in enumerate(other.coeffs):
                    if b:
                        k = G.mul[i][j]
                        result[k] ^= 1
        return GroupAlgebraElement(result, G)

    def left_mul(self, other):
        """ b * a """
        return other.right_mul(self)

    def is_zero(self):
        return not self.coeffs.any()
    
class GroupAlgebraMatrix:
    def __init__(self, data):
        self.data = np.array(data, dtype=object)
        self.group = self.data[0,0].group
        self.L = self.group.n

    def shape(self):
        return self.data.shape

    def zero(self):
        return GroupAlgebraElement([0]*self.L, self.group)
    
def tensor_product_R(A: GroupAlgebraMatrix, B: GroupAlgebraMatrix):
    """
    Construct formal tensor over R:
    returns two boundary maps in algebra form
    """

    mA, nA = A.data.shape
    mB, nB = B.data.shape

    # ∂2: R^{nA*nB} → R^{mA*nB} ⊕ R^{nA*mB}
    d2_top = [[A.data[i,j] for j in range(nA) for _ in range(nB)] for i in range(mA)]
    d2_bot = [[B.data[i,j] for _ in range(nA) for j in range(nB)] for i in range(mB)]

    # ∂1: R^{mA*nB} ⊕ R^{nA*mB} → R^{mA*mB}
    # constructed later in binary form

    return d2_top, d2_bot

def right_regular_matrix(ga):
    G = ga.group
    L = G.n
    M = np.zeros((L, L), dtype=np.uint8)

    for i in range(L):
        for k, c in enumerate(ga.coeffs):
            if c:
                j = G.mul[i][k]
                M[j, i] ^= 1
    return M


def left_regular_matrix(ga):
    G = ga.group
    L = G.n
    M = np.zeros((L, L), dtype=np.uint8)

    for i in range(L):
        for k, c in enumerate(ga.coeffs):
            if c:
                j = G.mul[k][i]
                M[j, i] ^= 1
    return M

def lift_matrix(GA: GroupAlgebraMatrix, side="right"):
    rows, cols = GA.data.shape
    L = GA.L

    H = np.zeros((rows*L, cols*L), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if side == "right":
                block = right_regular_matrix(GA.data[i,j])
            else:
                block = left_regular_matrix(GA.data[i,j])

            H[i*L:(i+1)*L, j*L:(j+1)*L] = block

    return H

class RingElement:
    # basis: [x, y, xy]
    def __init__(self, coeffs):
        self.coeffs = np.array(coeffs, dtype=np.uint8) % 2

    def __add__(self, other):
        return RingElement(self.coeffs ^ other.coeffs)

    def __repr__(self):
        names = ["x", "y", "xy"]
        terms = [n for n, c in zip(names, self.coeffs) if c]
        return " + ".join(terms) if terms else "0"
    
def multiply(a, b):
    ax, ay, axy = a.coeffs
    bx, by, bxy = b.coeffs
    cx = 0
    cy = 0
    cxy = (ax & by)

    return RingElement([cx, cy, cxy])


class RingMatrix:
    def __init__(self, data):
        self.data = np.array(data, dtype=object)

    def __matmul__(self, other):
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError("Dimension mismatch")

        zero = RingElement([0] * len(self.data[0,0].coeffs))

        result = [[zero for _ in range(other.data.shape[1])]
                  for _ in range(self.data.shape[0])]

        for i in range(self.data.shape[0]):
            for j in range(other.data.shape[1]):
                s = zero
                for k in range(self.data.shape[1]):
                    s = s + multiply(self.data[i, k], other.data[k, j])
                result[i][j] = s

        return RingMatrix(result)

    def is_zero(self):
        return all(
            all((e.coeffs == 0).all() for e in row)
            for row in self.data
        )

    def __repr__(self):
        return "\n".join(["\t".join(map(str, row)) for row in self.data])

