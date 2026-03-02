import numpy as np
from algebra import (
    FiniteGroup, RingElement, RingMatrix,
    GroupAlgebraElement, RingLifter,
    GroupAlgebraMatrix, left_regular_matrix, lift_to_binary, cyclic_group, right_regular_matrix
)

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
    def __init__(self,
                 A_ring: RingMatrix,
                 B_ring: RingMatrix,
                 generator_images):

        lifter = RingLifter(generator_images)

        GA = GroupAlgebraMatrix([
            [lifter.lift(e) for e in row]
            for row in A_ring.data
        ])

        GB = GroupAlgebraMatrix([
            [lifter.lift(e) for e in row]
            for row in B_ring.data
        ])

        HA = lift_to_binary(GA, side="right")
        HB = lift_to_binary(GB, side="left")

        self.HA = HA
        self.HB = HB

        mA, nA = HA.shape
        mB, nB = HB.shape

        IA = np.eye(nA, dtype=np.uint8)
        IB = np.eye(nB, dtype=np.uint8)
        ImA = np.eye(mA, dtype=np.uint8)
        ImB = np.eye(mB, dtype=np.uint8)


        A_kron = np.kron(HA, IB)     # mA*nB × nA*nB
        B_kron = np.kron(IA, HB)     # nA*mB × nA*nB

        self.d2 = np.vstack([
            A_kron,
            B_kron
        ]) % 2

        left_block = np.kron(ImA, HB)   # mA*mB × mA*nB
        right_block = np.kron(HA, ImB)  # mA*mB × nA*mB

        self.d1 = np.hstack([
            left_block,
            right_block
        ]) % 2

        self.H_X = self.d1
        self.H_Z = self.d2.T % 2

        if not np.all((self.H_X @ self.H_Z.T) % 2 == 0):
            raise ValueError("CSS condition failed")

        self.n = self.H_X.shape[1]
    def compute_k(self):

        rx = np.linalg.matrix_rank(self.H_X % 2)
        rz = np.linalg.matrix_rank(self.H_Z % 2)

        k = self.n - rx - rz
        return k

    def parity_checks(self):
        return self.H_X, self.H_Z

    def parameters(self):
        return self.n, self.compute_k()

    def __repr__(self):
        return (
            f"Lifted Product Code\n"
            f"H_X shape: {self.H_X.shape}\n"
            f"H_Z shape: {self.H_Z.shape}\n"
            f"parameters: n={self.n}, k={self.compute_k()}"
        )