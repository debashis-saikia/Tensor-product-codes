import numpy as np
import os

class SyndromeCircuit:

    def __init__(self, H_X, H_Z):

        if isinstance(H_X, str):
            if not os.path.exists(H_X):
                raise FileNotFoundError(f"{H_X} not found")
            self.HX = np.load(H_X)
        else:
            self.HX = np.array(H_X)

        if isinstance(H_Z, str):
            if not os.path.exists(H_Z):
                raise FileNotFoundError(f"{H_Z} not found")
            self.HZ = np.load(H_Z)
        else:
            self.HZ = np.array(H_Z)

        self.HX = self.HX
        self.HZ = self.HZ

        self.mX, self.n = self.HX.shape
        self.mZ, _ = self.HZ.shape

        self.operations = []
        self.Xerr = np.zeros(self.n, dtype=int)
        self.Zerr = np.zeros(self.n, dtype=int)

    def CNOT(self, c, t):

        self.operations.append(("CNOT", c, t))

        # propagate X errors
        if c < self.n and t < self.n:
            self.Xerr[t] ^= self.Xerr[c]

        # propagate Z errors
        if c < self.n and t < self.n:
            self.Zerr[c] ^= self.Zerr[t]

    def InitZ(self, q):
        self.operations.append(("InitZ", q))

    def InitX(self, q):
        self.operations.append(("InitX", q))
    
    def MeasZ(self, q):

        check = q - (self.n + self.mX)

        syndrome = 0
        for j in range(self.n):
            if self.HZ[check, j]:
                syndrome ^= self.Xerr[j]

        print("Z syndrome:", syndrome)

 

    def MeasX(self, q):

        check = q - self.n

        syndrome = 0
        for j in range(self.n):
            if self.HX[check, j]:
                syndrome ^= self.Zerr[j]

        print("X syndrome:", syndrome)
    
    def data(self, j):
        return j

    def ancX(self, i):
        return self.n + i

    def ancZ(self, i):
        return self.n + self.mX + i
    
    def measure_Z_checks(self):

        for i in range(self.mZ):

            a = self.ancZ(i)

            self.InitZ(a)

            for j in range(self.n):
                if self.HZ[i, j] == 1:
                    self.CNOT(self.data(j), a)

            self.MeasZ(a)

    def measure_X_checks(self):

        for i in range(self.mX):

            a = self.ancX(i)

            self.InitX(a)

            for j in range(self.n):
                if self.HX[i, j] == 1:
                    self.CNOT(a, self.data(j))

            self.MeasX(a)

    def syndrome_cycle(self):

        self.measure_Z_checks()
        self.measure_X_checks()

    def run(self, cycles):

        for _ in range(cycles):
            self.syndrome_cycle()

    def export_to_qasm(self, filename="syndrome.qasm"):

        n_total = self.n + self.mX + self.mZ

        qasm = []
        qasm.append("OPENQASM 2.0;")
        qasm.append('include "qelib1.inc";')
        qasm.append(f"qreg q[{n_total}];")
        qasm.append(f"creg c[{n_total}];")

        for op in self.operations:

            name = op[0]

            if name == "CNOT":
                _, c, t = op
                qasm.append(f"cx q[{c}],q[{t}];")

            elif name == "InitZ":
                _, q = op
                qasm.append(f"reset q[{q}];")

            elif name == "InitX":
                _, q = op
                qasm.append(f"reset q[{q}];")
                qasm.append(f"h q[{q}];")

            elif name == "MeasZ":
                _, q = op
                qasm.append(f"measure q[{q}] -> c[{q}];")

            elif name == "MeasX":
                _, q = op
                qasm.append(f"h q[{q}];")
                qasm.append(f"measure q[{q}] -> c[{q}];")

        with open(filename, "w") as f:
            f.write("\n".join(qasm))
        print(f"QASM circuit saved as {filename}")

    def export_to_stim(self):
    
        counter = 1
        while True:
            filename = f"syndrome_circuit_{counter:03d}.stim"
            if not os.path.exists(filename):
                break
            counter += 1

        lines = []

        for op in self.operations:

            name = op[0]

            if name == "CNOT":
                _, c, t = op
                lines.append(f"CX {c} {t}")

            elif name == "InitZ":
                _, q = op
                lines.append(f"R {q}")

            elif name == "InitX":
                _, q = op
                lines.append(f"RX {q}")

            elif name == "MeasZ":
                _, q = op
                lines.append(f"M {q}")

            elif name == "MeasX":
                _, q = op
                lines.append(f"MX {q}")

        with open(filename, "w") as f:
            f.write("\n".join(lines))
        print(f"Stim circuit saved as {filename}")


