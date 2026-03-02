import numpy as np

class RingElement:
    def __init__(self, coeffs):
        self.coeffs = np.array(coeffs, dtype=np.uint8) % 2

    def __add__(self, other):
        return RingElement(self.coeffs ^ other.coeffs)

    def __repr__(self):
        names = ["x", "y"]
        terms = [n for n,c in zip(names, self.coeffs) if c]
        return " + ".join(terms) if terms else "0"

def multiply(a, b):
    ax, ay = a.coeffs
    bx, by = b.coeffs

    c = (ax & by) ^ (ay & bx)

    return RingElement([c, c])

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



class GroupAlgebraElement:
    def __init__(self, coeffs, group_size):
        self.coeffs = np.array(coeffs, dtype=np.uint8) % 2
        self.n = group_size

    def __add__(self, other):
        return GroupAlgebraElement(self.coeffs ^ other.coeffs, self.n)

    def __mul__(self, other):
        result = np.zeros(self.n, dtype=np.uint8)
        for i, a in enumerate(self.coeffs):
            if a:
                for j, b in enumerate(other.coeffs):
                    if b:
                        result[(i + j) % self.n] ^= 1
        return GroupAlgebraElement(result, self.n)

    def __repr__(self):
        terms = [f"g^{i}" for i,c in enumerate(self.coeffs) if c]
        return " + ".join(terms) if terms else "0"
    
class GroupAlgebraMatrix:
    def __init__(self, data):
        self.data = np.array(data, dtype=object)

    def __matmul__(self, other):
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError("Dimension mismatch")

        L = self.data[0,0].n   
        zero = GroupAlgebraElement([0]*L, L)

        result = [[zero for _ in range(other.data.shape[1])]
                  for _ in range(self.data.shape[0])]

        for i in range(self.data.shape[0]):
            for j in range(other.data.shape[1]):
                s = zero
                for k in range(self.data.shape[1]):
                    s = s + (self.data[i,k] * other.data[k,j])
                result[i][j] = s

        return GroupAlgebraMatrix(result)

    def is_zero(self):
        return all(
            all((e.coeffs == 0).all() for e in row)
            for row in self.data
        )

class RingLifter:
    def __init__(self, generator_images):
        self.images = generator_images
        self.L = generator_images[0].n   

    def lift(self, ring_element):
        result = GroupAlgebraElement([0]*self.L, self.L)
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


def lift_to_binary(GA):
    rows, cols = GA.data.shape
    L = GA.data[0,0].n   

    H = np.zeros((rows*L, cols*L), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            H[i*L:(i+1)*L, j*L:(j+1)*L] = lift_ga_element(GA.data[i,j])

    return H



'''#Example usage:
ZERO = RingElement([0,0])
x = RingElement([1,0])
y = RingElement([0,1])

A = RingMatrix([[x, y]])
B = RingMatrix([[y], [x]])

C = A @ B
print(C)
print("AB = 0 ?", C.is_zero())

L = 5
g    = GroupAlgebraElement([0,1,0,0,0], L)
ginv = GroupAlgebraElement([0,0,0,0,1], L)

lifter = RingLifter([g, ginv])

GA = GroupAlgebraMatrix([[lifter.lift(e) for e in row] for row in A.data])
GB = GroupAlgebraMatrix([[lifter.lift(e) for e in row] for row in B.data])

print("Lifted AB = 0 ?", (GA @ GB).is_zero())

HA = lift_to_binary(GA)
HB = lift_to_binary(GB)

print("Binary AB = 0 ?", np.all((HA @ HB) % 2 == 0))'''





