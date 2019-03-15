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

import itertools
from typing import cast, Dict, Sequence, Tuple

import cirq
import cirq.contrib.acquaintance as cca
import numpy as np

import openfermion
import openfermioncirq.gates as ofc_gates

from openfermioncirq.gates import (
        CombinedSwapAndZ, CombinedCXXYYPowGate, CombinedDoubleExcitationGate)


def trotterize(hamiltonian: openfermion.InteractionOperator):
    """

    Returns gates such that $e^{i H} ~ \\prod_a e^{i H_a}$.

    """
    n_qubits = hamiltonian.n_qubits
    one_body_tensor = hamiltonian.one_body_tensor
    two_body_tensor = hamiltonian.two_body_tensor
#   assert np.allclose(one_body_tensor, np.conj(one_body_tensor))
#   assert np.allclose(one_body_tensor, one_body_tensor.T)
#   assert np.allclose(two_body_tensor.reshape((n_qubits ** 2,) * 2),
#           two_body_tensor.reshape((n_qubits ** 2,) * 2).T)

    gates = {} # type: Dict[Tuple[int, ...], cirq.Gate]
    for p in range(n_qubits):
        coeff = one_body_tensor[p, p]
        if coeff:
            gates[(p,)] = cirq.Z**(coeff / np.pi)
    for p, q in itertools.combinations(range(n_qubits), 2):
        tunneling_coeff = one_body_tensor[p, q] / np.pi
        interaction_coeff = (
                - two_body_tensor[p, q, p, q]
                + two_body_tensor[q, p, p, q]
                + two_body_tensor[p, q, q, p]
                - two_body_tensor[q, p, q, p]) / np.pi
        weights = (-tunneling_coeff, -interaction_coeff
                ) # type: Tuple[float, ...]
        if any(weights):
            gates[(p, q)] = CombinedSwapAndZ(
                    cast(Tuple[float, float], weights))
    for i, j, k in itertools.combinations(range(n_qubits), 3):
        weights = tuple(2 * sgn * (
            two_body_tensor[p, q, p, r] -
            two_body_tensor[p, q, r, p] -
            two_body_tensor[q, p, p, r] +
            two_body_tensor[q, p, r, p]) / np.pi
            for sgn, (p, q, r) in zip(
                [1, -1, 1], [(i, j, k), (j, k, i), (k, i, j)]))
        if any(weights):
            gates[(i, j, k)] = CombinedCXXYYPowGate(
                    cast(Tuple[float, float, float], weights))
    for i, j, k, l  in itertools.combinations(range(n_qubits), 4):
        weights = tuple(- 2 * (
            two_body_tensor[p, q, r, s] -
            two_body_tensor[p, q, s, r] -
            two_body_tensor[q, p, r, s] +
            two_body_tensor[q, p, s, r]) / np.pi
            for p, q, r, s in [(i, l, j, k), (i, k, j, l),  (i, j, k, l)])
        if any(weights):
            gates[(i, j, k, l)] = CombinedDoubleExcitationGate(
                    cast(Tuple[float, float, float], weights))
    return gates

def untrotterize(n_modes: int, gates: Dict[Tuple[int, ...], cirq.Gate]):
    one_body_tensor = np.zeros((n_modes,) * 2)
    two_body_tensor = np.zeros((n_modes,) * 4)

    global_shift = 0

    for indices, gate in gates.items():
        if isinstance(gate, cirq.ZPowGate):
            coeff = gate._exponent * np.pi
            global_shift += gate._exponent * gate._global_shift * np.pi
            one_body_tensor[indices * 2] += coeff
        elif isinstance(gate, ofc_gates.CombinedSwapAndZ):
            weights = tuple(-w * gate._exponent for w in gate.weights)
            global_shift += gate._exponent * gate._global_shift
            one_body_tensor[indices] += weights[0]
            two_body_tensor[indices * 2] += weights[1]
        elif isinstance(gate, ofc_gates.CombinedCXXYYPowGate):
            weights = tuple(
                    -0.5 * np.pi * w * gate._exponent for w in gate.weights)
            global_shift += gate._exponent * gate._global_shift
            p, q, r = indices
            two_body_tensor[p, q, p, r] += weights[0]
            two_body_tensor[p, q, q, r] += weights[1]
            two_body_tensor[p, r, q, r] += weights[2]
        elif isinstance(gate, ofc_gates.CombinedDoubleExcitationGate):
            weights = tuple(
                    0.5 * np.pi * w * gate._exponent for w in gate.weights)
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
            qubits: Sequence[cirq.QubitId],
            ) -> cirq.OP_TREE:
        index_set = frozenset(indices)
        n_indices = len(index_set)
        if index_set in self.index_set_to_gates:
            gates = self.index_set_to_gates.pop(index_set)
            index_to_qubit = dict(zip(indices, qubits))
            for gate_indices, gate in sorted(gates.items()):
                yield cca.LinearPermutationGate(n_indices,
                    dict(zip(gate_indices, indices)), ofc_gates.FSWAP)(*qubits)
                yield gate(*qubits)
                yield cca.LinearPermutationGate(n_indices,
                    dict(zip(indices, gate_indices)), ofc_gates.FSWAP)(*qubits)


def trotter_circuit(
        swap_network: cirq.Circuit,
        initial_mapping: Dict[cirq.LineQubit, int],
        hamiltonian: openfermion.InteractionOperator,
        execution_strategy: cca.executor.ExecutionStrategy =
            GreedyExecutionStrategy,
        ) -> cirq.Circuit:

    gates = trotterize(hamiltonian)
    circuit = swap_network.copy()
    execution_strategy(gates, initial_mapping)(circuit)
    return circuit
