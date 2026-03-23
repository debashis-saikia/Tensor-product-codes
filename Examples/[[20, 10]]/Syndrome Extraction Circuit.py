from SyndromeCircuit import SyndromeCircuit
import numpy as np

HX_file = r"C:\Users\...\HX_matrix_[20, 10].npy"
HZ_file = r"C:\Users\...\HZ_matrix_[20, 10].npy"

circuit = SyndromeCircuit(HX_file, HZ_file)
print("HX shape:", circuit.Hx.shape)
print("HZ shape:", circuit.Hz.shape)
print("Depth:", circuit.depth)
circuit.syndrome_cycle()
circuit.export_qasm("[20, 10]_syndrome.qasm")
circuit.export_stim("[20, 10]_syndrome.stim")
