from SyndromeCircuit import SyndromeCircuit

HX_file = f"C:\\Users\\IISER13\\OneDrive\\Desktop\\QEC_Codes\\HX_matrix_001.npy"
HZ_file = f"C:\\Users\\IISER13\\OneDrive\\Desktop\\QEC_Codes\\HZ_matrix_001.npy"

circuit = SyndromeCircuit(HX_file, HZ_file)

print("HX shape:", circuit.Hx.shape)
print("HZ shape:", circuit.Hz.shape)
print("Depth:", circuit.depth)

circuit.syndrome_cycle()

print("First 20 operations:\n")

for op in circuit.operations[:20]:
    print(op)

circuit.export_qasm("test_syndrome.qasm")
circuit.export_stim("test_syndrome.stim")
