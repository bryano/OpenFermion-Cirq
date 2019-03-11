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
from typing import cast, Dict, Tuple

import cirq
import numpy as np

import openfermion
import openfermioncirq.gates as ofc_gates

from openfermioncirq.gates import (
        CombinedSwapAndZ, CombinedCXXYYPowGate, CombinedDoubleExcitationGate)


def trotterize(hamiltonian: openfermion.InteractionOperator):
    """

    Returns gates such that $e^{i H} ~ \prod_a e^{i H_a}$.

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
            gates[(p,)] = cirq.Z**(-coeff / np.pi)
    for p, q in itertools.combinations(range(n_qubits), 2):
        tunneling_coeff = one_body_tensor[p, q]
        interaction_coeff = (
                two_body_tensor[p, q, p, q] +
                two_body_tensor[p, q, q, p] +
                two_body_tensor[q, p, q, p])
        weights = (-tunneling_coeff, -interaction_coeff
                ) # type: Tuple[float, ...]
        if any(weights):
            gates[(p, q)] = CombinedSwapAndZ(
                    cast(Tuple[float, float], weights))
    for i, j, k in itertools.combinations(range(n_qubits), 3):
        weights = tuple(
            two_body_tensor[p, r, q, r] +
            two_body_tensor[r, p, q, r] +
            two_body_tensor[p, r, r, q] +
            two_body_tensor[r, p, r, q]
            for p, q, r in [(i, j, k), (k, i, j), (j, k, i)])
        if any(weights):
            gates[(i, j, k)] = CombinedCXXYYPowGate(
                    cast(Tuple[float, float, float], weights))
    for i, j, k, l  in itertools.combinations(range(n_qubits), 4):
        weights = tuple(
            two_body_tensor[p, q, r, s] +
            two_body_tensor[p, q, s, r] +
            two_body_tensor[q, p, r, s] +
            two_body_tensor[q, p, s, r]
            for p, q, r, s in [(i, j, k, l), (i, k, j, l), (i, l, j, k)])
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
            coeff = -gate._exponent * np.pi
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

