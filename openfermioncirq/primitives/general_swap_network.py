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

from typing import Dict, Sequence, Tuple

import cirq
import cirq.contrib.acquaintance as cca
import numpy as np
import scipy.linalg as la

import openfermion
import openfermioncirq.gates as ofc_gates

from openfermioncirq.gates import (
    fermionic_simulation_gates_from_interaction_operator)

class FermionicSwapNetwork:
    def __init__(self,
            circuit: cirq.Circuit,
            initial_mapping: Dict[cirq.Qid, int],
            qubit_order: Sequence[cirq.Qid]
            ) -> None:
        self.circuit = circuit
        self.initial_mapping = initial_mapping
        self.qubit_order = qubit_order


def untrotterize(n_modes: int, gates: Dict[Tuple[int, ...], cirq.Gate]):
    # assumes gate indices in JW order
    one_body_tensor = np.zeros((n_modes,) * 2)
    two_body_tensor = np.zeros((n_modes,) * 4)

    global_shift = 0

    for indices, gate in gates.items():
        if isinstance(gate, cirq.ZPowGate):
            coeff = gate._exponent * np.pi
            global_shift += gate._exponent * gate._global_shift * np.pi
            one_body_tensor[indices * 2] += coeff
        elif isinstance(gate, ofc_gates.QuadraticFermionicSimulationGate):
            weights = tuple(w * gate._exponent for w in gate.weights)
            global_shift += gate._exponent * gate._global_shift
            one_body_tensor[indices] -= weights[0]
            two_body_tensor[indices * 2] += weights[1]
        elif isinstance(gate, ofc_gates.CubicFermionicSimulationGate):
            weights = tuple(
                    w * gate._exponent for w in gate.weights)
            global_shift += gate._exponent * gate._global_shift
            p, q, r = indices
            two_body_tensor[p, q, p, r] += weights[0]
            two_body_tensor[p, q, q, r] += weights[1]
            two_body_tensor[p, r, q, r] += weights[2]
        elif isinstance(gate, ofc_gates.QuarticFermionicSimulationGate):
            weights = tuple(
                    w * gate._exponent for w in gate.weights)
            global_shift += gate._exponent * gate._global_shift
            p, q, r, s = indices
            two_body_tensor[p, s, q, r] += weights[0]
            two_body_tensor[p, r, q, s] += weights[1]
            two_body_tensor[p, q, r, s] += weights[2]
        else:
            raise NotImplementedError()

    constant = global_shift
    one_body_tensor += one_body_tensor.T - np.diag(np.diag(one_body_tensor))
    two_body_tensor += (
            np.transpose(two_body_tensor, (2, 3, 0, 1)) -
            np.diag(np.diag(
                two_body_tensor.reshape(
                    (n_modes ** 2,) * 2))).reshape((n_modes,) * 4))
    return openfermion.InteractionOperator(
            constant, one_body_tensor, two_body_tensor)


class GreedyExecutionStrategy(cca.GreedyExecutionStrategy):
    def get_operations(self,
            indices: Sequence[int],
            qubits: Sequence[cirq.Qid],
            ) -> cirq.OP_TREE:
        index_set = frozenset(indices)
        n_indices = len(index_set)
        abs_to_rel = dict(zip(indices, range(n_indices)))
        if index_set in self.index_set_to_gates:
            gates = self.index_set_to_gates.pop(index_set)
            for gate_indices, gate in sorted(gates.items()):
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
    assert openfermion.is_hermitian(hamiltonian)
    gates = fermionic_simulation_gates_from_interaction_operator(hamiltonian)
    circuit = swap_network.copy()
    execution_strategy(gates, initial_mapping)(circuit)
    return circuit


def trotter_unitary(
        acquaintance_dag: cirq.CircuitDag,
        hamiltonian: openfermion.InteractionOperator,
        ) -> np.ndarray:
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
