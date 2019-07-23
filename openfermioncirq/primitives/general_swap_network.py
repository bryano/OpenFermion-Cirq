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

from typing import Dict, List, Sequence, Tuple

import cirq
import cirq.contrib.acquaintance as cca
import numpy as np
import openfermion
import scipy.linalg as la


import openfermioncirq.gates as ofc_gates


class GreedyExecutionStrategy(cca.GreedyExecutionStrategy):
    def get_operations(self,
            indices: Sequence[int],
            qubits: Sequence[cirq.Qid],
            ) -> cirq.OP_TREE:
        index_set = frozenset(indices)
        if index_set not in self.index_set_to_gates:
            return ()

        n_indices = len(index_set)
        abs_to_rel = dict(zip(indices, range(n_indices)))
        gates = self.index_set_to_gates.pop(index_set)
        for gate_indices, gate in sorted(gates.items()):
            if len(index_set) == 1:
                yield gate(*qubits)
            elif isinstance(gate, ofc_gates.FermionicSimulationGate):
                pos = [abs_to_rel[i] for i in gate_indices]
                yield gate.permuted(pos)(*qubits)
            else:
                perm = dict(zip((abs_to_rel[i] for i in gate_indices),
                    (abs_to_rel[j] for j in indices)))
                reverse_perm = dict((j, i) for i, j in perm.items())
                yield cca.LinearPermutationGate(n_indices, perm,
                    ofc_gates.FSWAP)(*qubits)
                yield gate(*qubits)
                yield cca.LinearPermutationGate(n_indices, reverse_perm,
                    ofc_gates.FSWAP)(*qubits)


def trotter_circuit(
        swap_network: cirq.Circuit,
        initial_mapping: Dict[cirq.LineQubit, int],
        hamiltonian: openfermion.InteractionOperator,
        execution_strategy: cca.executor.ExecutionStrategy =
            GreedyExecutionStrategy,
        ) -> cirq.Circuit:
    """Uses a swap network to construct a circuit that approximates exp(i H)
    for the specified H.

    Specifically, for each (unordered) set of modes $I = \{i_1, \ldots, i_k\}$,
    there is at most one gate $\exp(i H_I)$ corresponding to the term  $H_I$ in
    the Hamiltonian that involves exactly those modes.

    Args:
        swap_network: The circuit containing permutation gates and acquaintance
            opportunity gates.
        initial_mapping: The initial mapping from physical qubits to logical
            qubits.
        hamiltonian: The Hamiltonian to Trotterize.
        execution_strategy: The strategy used to replace acquaintance
            opportunity gates with concrete ones. Defaults to greedy, i.e., all
            gates are inserted at the first opportunity.
    """
    assert openfermion.is_hermitian(hamiltonian)
    gates = ofc_gates.fermionic_simulation_gates_from_interaction_operator(
            hamiltonian)
    circuit = swap_network.copy()
    execution_strategy(gates, initial_mapping)(circuit)
    return circuit


def trotter_unitary(
        acquaintance_dag: cirq.CircuitDag,
        hamiltonian: openfermion.InteractionOperator,
        ) -> np.ndarray:
    """Returns the unitary corresponding to the Trotterization of the given
    Hamiltonian according to the Trotter order specified by an acquaintance
    DAG.

    Each term is applied in full at every opportunity.

    Note that there may be be distinct unitaries consistent with the DAG if it
    does not specify a precedence relationship between some pair of
    non-commuting terms. No guarantee is made as to which one this method
    returns.

    Args:
        acquaintance_dag: A CircuitDag whose operations are
            AcquaintanceOperations. The indices of each AcquaintanceOperation
            indicate the part of the Hamiltonian to apply at that point.
        hamiltonian: The operator to Trotterize.
    """
    assert openfermion.is_hermitian(hamiltonian)
    unitary = np.eye(1 << hamiltonian.n_qubits)
    for acquaintance_op in acquaintance_dag.all_operations():
        indices = acquaintance_op.logical_indices
        partial_hamiltonian = hamiltonian.projected(indices, exact=True)
        qubit_operator = openfermion.jordan_wigner(partial_hamiltonian)
        qubit_operator_matrix = openfermion.qubit_operator_sparse(
                qubit_operator, hamiltonian.n_qubits)
        partial_unitary = la.expm(1j * qubit_operator_matrix).toarray()
        unitary = np.dot(partial_unitary, unitary)
    return unitary
