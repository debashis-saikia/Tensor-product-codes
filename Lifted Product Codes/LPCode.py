import code
import numpy as np
from algebra import (
    RingElement, RingMatrix,
    GroupAlgebraElement, RingLifter,
    GroupAlgebraMatrix,
)
from HGPCode import HGP


class LPC:
    """
    Lifted Product Codes (Panteleev-Kalachev Codes) can be viewed as a natural generalization of Hypergraph Product Codes (HGPs). 
    Indeed, these codes were first introduced as 'Generalized Hypergraph Product codes' by Panteleev and Kalachev.

    In a Hypergraph Product Code, one starts with two classical parity-check matrices A and B defined over the binary field F₂. 
    These matrices define a chain complex whose tensor-product construction yields the CSS parity check matrices H_X and H_Z.

    In a Lifted Product Code, this idea is generalized by replacing the binary
    parity-check matrices with RingArrays (matrices over a ring), typically a
    group algebra F_q[G] of a finite group G over a finite field F_q. 
    
    References:
        . https://errorcorrectionzoo.org/c/lifted_product
        . P. Panteleev and G. Kalachev, “Degenerate Quantum LDPC Codes With Good Finite Length Performance”, Quantum 5, 585 (2021) arXiv:1904.02703 DOI
        . P. Panteleev and G. Kalachev, “Asymptotically Good Quantum and Locally Testable Classical LDPC Codes”, (2022) arXiv:2111.03654
        . P. Panteleev and G. Kalachev, “Quantum LDPC Codes With Almost Linear Minimum Distance”, IEEE Transactions on Information Theory 68, 213 (2022) arXiv:2012.04068 DOI

    Outputs:
    - H_X, H_Z
    - n (number of physical qubits)
    - k (number of logical qubits)
    """

    def __init__(self, A_ring: RingMatrix, B_ring: RingMatrix,
                 generator_images: list[GroupAlgebraElement]):
        """
        Parameters
        ----------
        A_ring, B_ring : RingMatrix
            Abstract ring matrices satisfying AB = 0

        generator_images : list[GroupAlgebraElement]
            Images of ring generators (e.g. [g, g^{-1}])
        """

        if not (A_ring @ B_ring).is_zero():
            raise ValueError("Abstract condition AB = 0 is not satisfied.")

        self.A_ring = A_ring
        self.B_ring = B_ring

        self.lifter = RingLifter(generator_images)

        self.GA = GroupAlgebraMatrix([
            [self.lifter.lift(e) for e in row] for row in A_ring.data
        ])

        self.GB = GroupAlgebraMatrix([
            [self.lifter.lift(e) for e in row] for row in B_ring.data
        ])

        if not (self.GA @ self.GB).is_zero():
            raise ValueError("Lifted condition AB = 0 failed in group algebra.")


        self.HA = lift_to_binary(self.GA)
        self.HB = lift_to_binary(self.GB)

        if not np.all((self.HA @ self.HB) % 2 == 0):
            raise ValueError("Binary CSS condition HA HB = 0 failed.")

        self.hgp = HGP(A=self.HA, B=self.HB)

        self.H_X = self.hgp.H_X
        self.H_Z = self.hgp.H_Z

        self.n = self.hgp.n
        self.k = self.hgp.k

    def parameters(self):
        """Return code parameters [[n, k]]."""
        return self.n, self.k

    def parity_checks(self):
        """Return H_X and H_Z."""
        return self.H_X, self.H_Z


'''# Example usage:

x = RingElement([1, 0])
y = RingElement([0, 1])

A = RingMatrix([[x, y]])
B = RingMatrix([[y], [x]])

L = 4
g = GroupAlgebraElement([0,1,0,0], L)
g_inv = GroupAlgebraElement([0,0,0,1], L)

code = LPC(A, B, [g, g_inv])

print("n, k =", code.parameters())

H_X, H_Z = code.parity_checks()
print("H_X shape:", H_X.shape)
print("H_Z shape:", H_Z.shape)'''



