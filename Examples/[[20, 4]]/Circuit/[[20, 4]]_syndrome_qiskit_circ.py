from qiskit import QuantumCircuit

circ = f'C:....\\[20, 4]_syndrome.qasm'

qc = QuantumCircuit.from_qasm_file(circ)

qc.draw('mpl', fold = 1,filename='[20, 4]_syndrome.png')
