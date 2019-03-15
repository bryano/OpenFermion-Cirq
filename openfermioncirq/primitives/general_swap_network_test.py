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
import pytest

import cirq
import cirq.contrib.acquaintance as cca
import numpy as np
import openfermion
import scipy.linalg as la

import openfermioncirq as ofc
from openfermioncirq.primitives.general_swap_network import (
        trotterize, untrotterize, trotter_circuit)
from openfermioncirq.primitives.normal_order import (
        normal_ordered_interaction_operator)


@pytest.mark.parametrize('hamiltonian',
    [ofc.testing.random.random_interaction_operator_term(1) for _ in range(5)])
def test_trotterize_linear(hamiltonian):
    q = cirq.LineQubit(0)
    gate = cca.acquaint(q)
    device = cca.UnconstrainedAcquaintanceDevice
    swap_network = cirq.Circuit.from_ops(gate, device=device)
    initial_mapping = {q: 0}
    circuit = trotter_circuit(swap_network, initial_mapping, hamiltonian)
    actual_unitary = cirq.unitary(circuit)
    qubit_operator = openfermion.jordan_wigner(hamiltonian)
    qubit_operator_matrix = openfermion.qubit_operator_sparse(qubit_operator)
    expected_unitary = la.expm(1j * qubit_operator_matrix).toarray()
    assert np.allclose(actual_unitary, expected_unitary)


@pytest.mark.parametrize('hamiltonian',
    [ofc.testing.random.random_interaction_operator_term(2) for _ in range(5)])
def test_trotterize_quadratic(hamiltonian):
    qubits = cirq.LineQubit.range(2)
    gate = cca.acquaint(*qubits)
    device = cca.UnconstrainedAcquaintanceDevice
    swap_network = cirq.Circuit.from_ops(gate, device=device)
    qubit_operator = openfermion.jordan_wigner(hamiltonian)
    qubit_operator_matrix = openfermion.qubit_operator_sparse(qubit_operator)
    expected_unitary = la.expm(1j * qubit_operator_matrix).toarray()
    for perm in itertools.permutations(range(2)):
        initial_mapping = dict(zip(qubits, perm))
        circuit = trotter_circuit(swap_network, initial_mapping, hamiltonian)
        actual_unitary = cirq.unitary(circuit)
        assert np.allclose(actual_unitary, expected_unitary)


@pytest.mark.parametrize('constant,potential',
    [(1, 1), (0, 1), (0, 0.3), (-0.5, 0.7)])
def test_untrotterize_linear(constant, potential):
    exponent = potential / np.pi
    global_shift = constant / (exponent * np.pi)
    gates = {
            (0,): cirq.ZPowGate(exponent=exponent, global_shift=global_shift)}
    hamiltonian = untrotterize(2, gates)
    assert np.isclose(hamiltonian.constant, constant)
    assert np.allclose(hamiltonian.one_body_tensor, [[potential, 0], [0, 0]])
    assert np.allclose(hamiltonian.two_body_tensor, np.zeros((2,) * 4))


@pytest.mark.parametrize('constant,tunneling,interaction,scale',
    [(0,0,0,1), (1, 1, 1, 1), (0.2, -0.5, 1.7, 0.3), (-0.1, -0.5, 1, -1)])
def test_untrotterize_quadratic(constant, tunneling, interaction, scale):
    weights = (-tunneling, -interaction)
    gate = ofc.CombinedSwapAndZ(weights, exponent=scale, global_shift=constant)
    gates = {(0, 1): gate}
    hamiltonian = untrotterize(2, gates)
    expected_constant = constant * scale
    assert np.isclose(hamiltonian.constant, expected_constant)
    expected_one_body_tensor = np.zeros((2, 2))
    expected_one_body_tensor[0, 1] = expected_one_body_tensor[1, 0] = tunneling
    expected_one_body_tensor *= scale
    assert np.allclose(hamiltonian.one_body_tensor, expected_one_body_tensor)
    expected_two_body_tensor = np.zeros((2,) * 4)
    expected_two_body_tensor[0, 1, 0, 1] = interaction
    expected_two_body_tensor *= scale
    assert np.allclose(hamiltonian.two_body_tensor, expected_two_body_tensor)


@pytest.mark.parametrize('constant,coeffs,scale',
    [(np.random.standard_normal(),
      np.random.standard_normal(3),
      np.random.standard_normal()) for _ in range(10)])
def test_untrotterize_cubic(constant, coeffs, scale):
    weights = tuple(-2 * c  / np.pi for c in coeffs)
    gate = ofc.CombinedCXXYYPowGate(
            weights=weights, exponent=scale, global_shift=constant)
    gates = {(0, 1, 2): gate}
    hamiltonian = untrotterize(3, gates)
    expected_constant = constant * scale
    assert np.isclose(hamiltonian.constant, expected_constant)
    assert np.allclose(hamiltonian.one_body_tensor, np.zeros((3, 3)))
    expected_two_body_tensor = np.zeros((3,) * 4)
    composite_indices = [((0, 1), (0, 2)), ((0, 1), (1, 2)), ((0, 2), (1, 2))]
    for (pq, rs), w in zip(composite_indices, coeffs):
        expected_two_body_tensor[pq + rs] = w
        expected_two_body_tensor[rs + pq] = w
    expected_two_body_tensor *= scale
    assert np.allclose(hamiltonian.two_body_tensor, expected_two_body_tensor)


@pytest.mark.parametrize('constant,coeffs,scale',
    [(np.random.standard_normal(),
      np.random.standard_normal(3),
      np.random.standard_normal()) for _ in range(10)])
def test_untrotterize_quartic(constant, coeffs, scale):
    weights = tuple(2 * c / np.pi for c in coeffs)
    gate = ofc.CombinedDoubleExcitationGate(
            weights=weights, exponent=scale, global_shift=constant)
    gates = {(0, 1, 2, 3): gate}
    hamiltonian = untrotterize(4, gates)
    expected_constant = constant * scale
    assert np.isclose(hamiltonian.constant, expected_constant)
    assert np.allclose(hamiltonian.one_body_tensor, np.zeros((4, 4)))
    expected_two_body_tensor = np.zeros((4,) * 4)
    composite_indices = [((0, 3), (1, 2)), ((0, 2), (1, 3)), ((0, 1), (2, 3))]
    for (pq, rs), w in zip(composite_indices, coeffs):
        expected_two_body_tensor[pq + rs] = w
        expected_two_body_tensor[rs + pq] = w
    expected_two_body_tensor *= scale
    assert np.allclose(np.mod(hamiltonian.two_body_tensor, 2 * np.pi),
                       np.mod(expected_two_body_tensor, 2 * np.pi))


@pytest.mark.parametrize('hamiltonian',
#       [ofc.testing.random_interaction_operator(5) for _ in range(10)])
        [ofc.testing.random_interaction_operator(2) for _ in range(1)])
def test_untrotterize(hamiltonian):
    hamiltonian.constant = 0
    n_modes = len(hamiltonian.one_body_tensor)
    normal_ordered_hamiltonian = (
            normal_ordered_interaction_operator(hamiltonian))

    gates = trotterize(hamiltonian)

    other_hamiltonian = untrotterize(n_modes, gates)
    other_normal_ordered_hamiltonian = (
            normal_ordered_interaction_operator(other_hamiltonian))
    print(gates)
    print(normal_ordered_hamiltonian)
    print(other_normal_ordered_hamiltonian)
#   assert normal_ordered_hamiltonian == other_normal_ordered_hamiltonian