import numpy as np
import os


class SyndromeCircuit:

    def __init__(self, H_X, H_Z):

        if isinstance(H_X, str):
            if not os.path.exists(H_X):
                raise FileNotFoundError(f"{H_X} not found")
            self.Hx = np.load(H_X)
        else:
            self.Hx = np.array(H_X)

        if isinstance(H_Z, str):
            if not os.path.exists(H_Z):
                raise FileNotFoundError(f"{H_Z} not found")
            self.Hz = np.load(H_Z)
        else:
            self.Hz = np.array(H_Z)

        self.m = self.Hx.shape[0]
        self.n = self.Hx.shape[1]
        self.split = self.n // 2

        self.depth = 5

        self.qX = [f"qX_{i}" for i in range(self.m)]
        self.qL = [f"qL_{i}" for i in range(self.split)]
        self.qR = [f"qR_{i}" for i in range(self.split)]
        self.qZ = [f"qZ_{i}" for i in range(self.m)]

        self.operations = []

        self._extract_edges()

    def _extract_edges(self):

        A = self.Hx[:, :self.split]
        B = self.Hx[:, self.split:]

        self.A_edges = []
        self.B_edges = []

        for i in range(self.m):
            colsA = np.where(A[i] == 1)[0]
            colsB = np.where(B[i] == 1)[0]

            self.A_edges.append(colsA.tolist())
            self.B_edges.append(colsB.tolist())

    def initX(self, q):
        self.operations.append(("InitX", q))

    def initZ(self, q):
        self.operations.append(("InitZ", q))

    def cnot(self, control, target):
        self.operations.append(("CNOT", control, target))

    def measX(self, q):
        self.operations.append(("MeasX", q))

    def measZ(self, q):
        self.operations.append(("MeasZ", q))

    def idle(self, q):
        self.operations.append(("Idle", q))

    def syndrome_cycle(self):

        for i in range(self.m):
            self.initX(self.qX[i])

        for i in range(self.m):
            for j in self.A_edges[i]:
                self.cnot(self.qX[i], self.qL[j])
                self.cnot(self.qR[j], self.qZ[i])

        for i in range(self.m):
            for j in self.B_edges[i]:
                self.cnot(self.qX[i], self.qR[j])
                self.cnot(self.qL[j], self.qZ[i])

        for i in range(self.m):
            self.measZ(self.qZ[i])

        for i in range(self.m):
            self.measX(self.qX[i])
            self.initZ(self.qZ[i])

    def show_operations(self):
        for op in self.operations:
            print(op)

    def export_qasm(self, filename="syndrome_circuit.qasm"):

        qubits = self.qX + self.qL + self.qR + self.qZ
        index_map = {q: i for i, q in enumerate(qubits)}

        with open(filename, "w") as f:

            f.write("OPENQASM 2.0;\n")
            f.write('include "qelib1.inc";\n')

            f.write(f"qreg q[{len(qubits)}];\n")
            f.write(f"creg c[{len(qubits)}];\n")

            for op in self.operations:

                if op[0] == "InitX":
                    q = index_map[op[1]]
                    f.write(f"h q[{q}];\n")

                elif op[0] == "InitZ":
                    q = index_map[op[1]]
                    f.write(f"reset q[{q}];\n")

                elif op[0] == "CNOT":
                    c = index_map[op[1]]
                    t = index_map[op[2]]
                    f.write(f"cx q[{c}],q[{t}];\n")

                elif op[0] == "MeasZ":
                    q = index_map[op[1]]
                    f.write(f"measure q[{q}] -> c[{q}];\n")

                elif op[0] == "MeasX":
                    q = index_map[op[1]]
                    f.write(f"h q[{q}];\n")
                    f.write(f"measure q[{q}] -> c[{q}];\n")

        print("QASM file written:", filename)

    def export_stim(self, filename="syndrome_circuit.stim"):

        qubits = self.qX + self.qL + self.qR + self.qZ
        index_map = {q: i for i, q in enumerate(qubits)}

        with open(filename, "w") as f:

            for op in self.operations:

                if op[0] == "InitX":
                    q = index_map[op[1]]
                    f.write(f"RX {q}\n")

                elif op[0] == "InitZ":
                    q = index_map[op[1]]
                    f.write(f"R {q}\n")

                elif op[0] == "CNOT":
                    c = index_map[op[1]]
                    t = index_map[op[2]]
                    f.write(f"CX {c} {t}\n")

                elif op[0] == "MeasZ":
                    q = index_map[op[1]]
                    f.write(f"M {q}\n")

                elif op[0] == "MeasX":
                    q = index_map[op[1]]
                    f.write(f"MX {q}\n")

        print("Stim file written:", filename)