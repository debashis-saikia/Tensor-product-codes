from qiskit import QuantumCircuit

circ = f'C:\\Users\\IISER13\\OneDrive\\Desktop\\Pavel Panteleev\\Tensor Product Codes\\[20, 10]_syndrome.qasm'

qc = QuantumCircuit.from_qasm_file(circ)
qc.draw('mpl', fold = 1,filename='[20, 10]_syndrome.png')