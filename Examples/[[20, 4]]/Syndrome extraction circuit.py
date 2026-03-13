from SyndromeCircuit import SyndromeCircuit
import numpy as np

HX_file = r"C:....\HX_matrix_[20, 4].npy"
HZ_file = r"C:....\HZ_matrix_[20, 4].npy"

circuit = SyndromeCircuit(HX_file, HZ_file)
print("HX shape:", circuit.Hx.shape)
print("HZ shape:", circuit.Hz.shape)
print("Depth:", circuit.depth)
circuit.syndrome_cycle()
circuit.export_qasm("[20, 4]_syndrome.qasm")
circuit.export_stim("[20, 4]_syndrome.stim")
