import numpy as np

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


class FiniteGroup:
    def __init__(self, multiplication_table):
        self.mul = multiplication_table
        self.n = len(multiplication_table)

    def right_mul(self, i, j):
        return self.mul[i][j]

    def left_mul(self, i, j):
        return self.mul[i][j]
    

class GroupAlgebraElement:

    def __init__(self, coeffs, group: FiniteGroup):
        self.coeffs = np.array(coeffs, dtype=np.uint8) % 2
        self.group = group
        self.n = group.n

    def __add__(self, other):
        return GroupAlgebraElement(
            self.coeffs ^ other.coeffs,
            self.group
        )
    def __mul__(self, other):

        result = np.zeros(self.n, dtype=np.uint8)

        for i, a in enumerate(self.coeffs):
            if a:
                for j, b in enumerate(other.coeffs):
                    if b:
                        k = self.group.mul[i][j]
                        result[k] ^= 1

        return GroupAlgebraElement(result, self.group)

    def right_mul(self, other):
        # a · b
        return self * other

    def left_mul(self, other):
        # b · a
        return other * self

    def matmul(self, other, mode="right-left"):

        if mode == "right-left":
            return self.right_mul(other)

        elif mode == "left-right":
            return self.left_mul(other)

        else:
            raise ValueError(
                "mode must be 'right-left' or 'left-right'"
            )

    def __repr__(self):
        terms = [
            f"g^{i}" for i, c in enumerate(self.coeffs) if c
        ]
        return " + ".join(terms) if terms else "0"
    
class GroupAlgebraMatrix:
    """
    Matrix over group algebra F₂[G].

    Supports sided multiplication required for
    non-commutative lifted product codes.

    A acts as RIGHT module
    B acts as LEFT module
    """

    def __init__(self, data):
        self.data = np.array(data, dtype=object)

        if self.data.size == 0:
            raise ValueError("Matrix cannot be empty")

        self.group = self.data[0, 0].group
        self.L = self.data[0, 0].n

    def _zero(self):
        return GroupAlgebraElement(
            [0] * self.L,
            self.group
        )

    def matmul(self, other, mode="right-left"):
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError("Dimension mismatch")

        rows = self.data.shape[0]
        cols = other.data.shape[1]
        inner = self.data.shape[1]

        result = [
            [self._zero() for _ in range(cols)]
            for _ in range(rows)
        ]

        for i in range(rows):
            for j in range(cols):

                acc = self._zero()

                for k in range(inner):

                    term = self.data[i, k].matmul(
                        other.data[k, j],
                        mode=mode
                    )

                    acc = acc + term

                result[i][j] = acc

        return GroupAlgebraMatrix(result)

    def __matmul__(self, other):
        raise RuntimeError(
            "Use matmul(..., mode=...) explicitly "
            "for non-commutative lifted products."
        )

    def is_zero(self):
        for row in self.data:
            for e in row:
                if not (e.coeffs == 0).all():
                    return False
        return True

    def __repr__(self):
        return "\n".join(
            "\t".join(str(e) for e in row)
            for row in self.data
        )
    

class RingLifter:
    def __init__(self, generator_images):
        self.images = generator_images
        self.L = generator_images[0].n   

    def lift(self, ring_element):
        result = GroupAlgebraElement([0]*self.L, self.images[0].group)
        for c, img in zip(ring_element.coeffs, self.images):
            if c:
                result = result + img
        return result


    
def cyclic_permutation(k, L):
    P = np.zeros((L, L), dtype=np.uint8)
    for i in range(L):
        P[i, (i + k) % L] = 1
    return P

def lift_ga_element(ga):
    M = np.zeros((ga.n, ga.n), dtype=np.uint8)
    for k, c in enumerate(ga.coeffs):
        if c:
            M ^= cyclic_permutation(k, ga.n)
    return M

def cyclic_group(L):
    table = [[(i + j) % L for j in range(L)] for i in range(L)]
    return FiniteGroup(table)

def right_regular_matrix(ga):
    G = ga.group
    L = G.n
    M = np.zeros((L, L), dtype=np.uint8)

    for i in range(L):  # basis element g_i
        for k, c in enumerate(ga.coeffs):
            if c:
                j = G.mul[i][k]  # g_i * g_k
                M[j, i] ^= 1

    return M

def left_regular_matrix(ga):
    G = ga.group
    L = G.n
    M = np.zeros((L, L), dtype=np.uint8)

    for i in range(L):  # basis element g_i
        for k, c in enumerate(ga.coeffs):
            if c:
                j = G.mul[k][i]  # g_k * g_i
                M[j, i] ^= 1

    return M

def lift_to_binary(GA, side="right"):
    rows, cols = GA.data.shape
    L = GA.data[0,0].n

    H = np.zeros((rows*L, cols*L), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if side == "right":
                block = right_regular_matrix(GA.data[i,j])
            elif side == "left":
                block = left_regular_matrix(GA.data[i,j])
            else:
                raise ValueError("side must be 'right' or 'left'")

            H[i*L:(i+1)*L, j*L:(j+1)*L] = block

    return H