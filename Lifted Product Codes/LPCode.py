import os

import numpy as np
from algebra import (GroupAlgebraMatrix, lift_matrix)

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

    Lifted Product Code construction:

    Builds CSS matrices from lifted chain complexes:

        R^{nA nB}
            | d2
        R^{mA nB} ⊕ R^{nA mB}
            | d1
        R^{mA mB}

    with
        H_X = d1
        H_Z = d2^T
    """
    def __init__(self, A: GroupAlgebraMatrix, B: GroupAlgebraMatrix):

        # Lift AFTER algebra is defined
        HA = lift_matrix(A, side="right")
        HB = lift_matrix(B, side="left")

        mA, nA = HA.shape
        mB, nB = HB.shape

        IA = np.eye(nA, dtype=np.uint8)
        IB = np.eye(nB, dtype=np.uint8)
        ImA = np.eye(mA, dtype=np.uint8)
        ImB = np.eye(mB, dtype=np.uint8)

        # ∂2
        A_kron = np.kron(HA, IB)
        B_kron = np.kron(IA, HB)

        self.d2 = np.vstack([A_kron, B_kron]) % 2

        # ∂1
        left_block  = np.kron(ImA, HB)
        right_block = np.kron(HA, ImB)

        self.d1 = np.hstack([left_block, right_block]) % 2

        self.HX = self.d1
        self.HZ = self.d2.T % 2

        if not np.all((self.HX @ self.HZ.T) % 2 == 0):
            raise ValueError("CSS condition failed")

        self.n = self.HX.shape[1]

    def rank(self, M):
        A = M.copy()
        r, c = A.shape
        rank = 0
        col = 0

        for i in range(r):
            while col < c and not A[i:, col].any():
                col += 1
            if col == c:
                break

            pivot = i + np.argmax(A[i:, col])
            A[[i, pivot]] = A[[pivot, i]]

            for j in range(r):
                if j != i and A[j, col]:
                    A[j] ^= A[i]

            rank += 1
            col += 1

        return rank

    def k(self):
        return self.n - self.rank(self.HX) - self.rank(self.HZ)
