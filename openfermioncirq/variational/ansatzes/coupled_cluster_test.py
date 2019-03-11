#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import cirq
import cirq.contrib.acquaintance as cca

from openfermioncirq.variational.ansatzes.coupled_cluster import (
    CoupledClusterOperator, GeneralizedCoupledClusterOperator,
    PairedCoupledClusterOperator,
    UnitaryCoupledClusterAnsatz)

def test_cc_operator():
    operator = CoupledClusterOperator(7, 3)
    operator.operator()

def test_generalized_cc_operator():
    operator = GeneralizedCoupledClusterOperator(5)
    operator.operator()

def test_paired_cc_operator():
    operator = PairedCoupledClusterOperator(4)
    operator.operator()

def test_paired_ucc():
    n_spatial_modes = 4
    cluster_operator = PairedCoupledClusterOperator(n_spatial_modes)
    qubits = cirq.LineQubit.range(cluster_operator.n_spin_modes)
    qubit_pairs = [qubits[2 * i: 2 * (i + 1)] for i in range(n_spatial_modes)]
    swap_network, qubit_order = cca.quartic_paired_acquaintance_strategy(
            qubit_pairs)
    initial_mapping = {p: l.x for p, l in zip(qubits, qubit_order)}
    ansatz = UnitaryCoupledClusterAnsatz(
            cluster_operator, swap_network, initial_mapping)
    gate = list(ansatz._circuit[0])[0]
#   print(gate)
    circuit = cirq.Circuit.from_ops(gate)
#   print(circuit)
#   print(ansatz._circuit.to_text_diagram(qubit_order=qubit_order))
