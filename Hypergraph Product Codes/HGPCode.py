import numpy as np
import scipy as sp
import galois
from typing import Optional
from scipy.sparse import csr_matrix as csr


class HGP:
    def __init__(self, A : Optional[np.ndarray] = None, B : Optional[np.ndarray] = None):
        """
        Initialize the Hypergraph Product Code with given classical parity check matrices.

        A : Classical parity check matrix of the first code. :-> np.ndarray
        B : Classical parity check matrix of the second code. :-> np.ndarray

        H_X, H_Z : Parity check matrices for the Hypergraph Code :-> np.ndarray
        ---------------------------------------------------------------------------------

        Hypergraph Product Codes are a class of quantum error-correcting codes constructed from two classical linear codes. 
        They are defined by their parity check matrices H_X and H_Z, which are derived from the classical parity check matrices A and B.
        The CSS condition H_X * H_Z^T = 0 must hold for the code to be valid.

        References:
        - https://ieeexplore.ieee.org/document/6671468
        - https://errorcorrectionzoo.org/c/hypergraph_product

        """
        if A is None and B is None:
            raise ValueError("At least one of the matrices A or B must be provided.")
        if A is None:
            A = B.T
        if B is None:
            B = A.T

        self.A = A
        self.B = B
        self.mA, self.mB = self.A.shape[0], self.B.shape[0]
        self.nA, self.nB = self.A.shape[1], self.B.shape[1]

        # Constructing parity check matrices H_X and H_Z
        self.H_X, self.H_Z = self._get_parity_check_matrix(self.A, self.B)

        # Verify CSS condition
        self._css_condition()

        # Getting parameters of the code [[n, k]]
        self.n, self.k =  self._get_n_(), self._get_k_()

    def _css_condition(self) -> bool:
        """
        Check if the CSS condition H_X * H_Z^T = 0 holds.

        Returns:
        bool: True if the CSS condition holds, False otherwise.
        """
        product = self.H_X.dot(self.H_Z.T).toarray()
        
        if np.all(product % 2 == 0):
            return True
        else:
            raise ValueError("CSS condition does not hold.")
        
        return None
    
    def _get_n_(self)-> int:
        """
        Number of physical qubits in the Hypergraph Product Code.
        """
        n =  self.mA * self.mB + self.nA * self.nB

        if n != self.H_X.shape[1] or n != self.H_Z.shape[1]:  
            raise ValueError(f"Computed n={n} does not match parity check matrix shape.")
        return n

    def _get_k_(self)-> int:
        """
        Number of logical qubits in the Hypergraph Product Code.
        """
        GF2 = galois.GF(2)

        rank_A = np.linalg.matrix_rank(GF2(self.A))
        rank_B = np.linalg.matrix_rank(GF2(self.B))
        rank_AT = np.linalg.matrix_rank(GF2(self.A.T))
        rank_BT = np.linalg.matrix_rank(GF2(self.B.T))

        kA = self.nA - rank_A
        kB = self.nB - rank_B

        kAT = self.mA - rank_AT
        kBT = self.mB - rank_BT

        return kA * kB + kAT * kBT
    

    @staticmethod
    def _get_parity_check_matrix(
            A : np.ndarray,
            B : np.ndarray
        )-> tuple[csr, csr]:
        """
        Constructs the parity check matrices H_X and H_Z for the Hypergraph Product Code.
        """
        mA, nA = A.shape
        mB, nB = B.shape

        A_csr = sp.sparse.csr_matrix(A)
        B_csr = sp.sparse.csr_matrix(B)

        # H_X parity check matrix construction
        HX_left = sp.sparse.kron(A_csr, sp.sparse.eye(nB, format='csr'))
        HX_right = sp.sparse.kron(sp.sparse.eye(mA, format='csr'), B_csr.T)
        HX = sp.sparse.hstack((HX_left, HX_right), format='csr')

        # H_Z parity check matrix construction
        HZ_left = sp.sparse.kron(sp.sparse.eye(nA, format='csr'), B_csr)
        HZ_right = sp.sparse.kron(A_csr.T, sp.sparse.eye(mB, format='csr'))
        HZ = sp.sparse.hstack((HZ_left, HZ_right), format='csr')

        return HX.tocsr(), HZ.tocsr()
