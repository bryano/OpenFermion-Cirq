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
        trotterize, untrotterize, trotter_circuit, trotter_unitary)


@pytest.mark.parametrize('order,hamiltonian',
    [(order, ofc.testing.random_interaction_operator_term(order))
     for order in (1, 2, 3, 4) for _ in range(5)])
def test_trotterize_term(order, hamiltonian):
    n_orbitals = order
    qubits = cirq.LineQubit.range(n_orbitals)
    device = cca.UnconstrainedAcquaintanceDevice
    qubit_operator = openfermion.jordan_wigner(hamiltonian)
    qubit_operator_matrix = openfermion.qubit_operator_sparse(
            qubit_operator, n_orbitals)
    expected_unitary = la.expm(1j * qubit_operator_matrix).toarray()
    initial_mapping = dict(zip(qubits, range(n_orbitals)))
    for perm in itertools.permutations(range(n_orbitals)):
        ops = [
            cca.LinearPermutationGate(n_orbitals,
                dict(zip(range(n_orbitals), perm)), ofc.FSWAP)(*qubits),
            cca.acquaint(*qubits),
            cca.LinearPermutationGate(n_orbitals,
                dict(zip(perm, range(n_orbitals))), ofc.FSWAP)(*qubits)
            ]
        swap_network = cirq.Circuit.from_ops(ops, device=device)
        circuit = trotter_circuit(swap_network, initial_mapping, hamiltonian)
        actual_unitary = cirq.unitary(circuit)
        assert np.allclose(actual_unitary, expected_unitary)


@pytest.mark.parametrize('order,hamiltonian',
    [(order, openfermion.utils._testing_utils.random_interaction_operator(5))
     for order in (1, 2, 3, 4) for _ in range(2)])
def test_trotterize(order, hamiltonian):
    hamiltonian = hamiltonian.projected(order, True)

    qubits = cirq.LineQubit.range(hamiltonian.n_qubits)
    initial_mapping = dict(zip(qubits, range(hamiltonian.n_qubits)))
    swap_network = cca.complete_acquaintance_strategy(
            qubits, order, ofc.FSWAP, remove_redundancies=True)
    cca.return_to_initial_mapping(swap_network, ofc.FSWAP)
    cca.compress_permutations(swap_network, ofc.FSWAP)

    circuit = trotter_circuit(swap_network, initial_mapping, hamiltonian)

    actual_unitary = circuit.to_unitary_matrix(qubit_order=qubits)
    acquaintance_dag = cca.get_acquaintance_dag(swap_network, initial_mapping)
    expected_unitary = trotter_unitary(acquaintance_dag, hamiltonian)

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
    weights = (-tunneling / np.pi, interaction / np.pi)
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
        expected_two_body_tensor[pq + rs] = -w
        expected_two_body_tensor[rs + pq] = -w
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
        expected_two_body_tensor[pq + rs] = -w
        expected_two_body_tensor[rs + pq] = -w
    expected_two_body_tensor *= scale
    assert np.allclose(np.mod(hamiltonian.two_body_tensor, 2 * np.pi),
                       np.mod(expected_two_body_tensor, 2 * np.pi))


@pytest.mark.parametrize('hamiltonian',
    [openfermion.utils._testing_utils.random_interaction_operator(5)
    for _ in range(10)])
def test_untrotterize(hamiltonian):
    hamiltonian.constant = 0
    n_modes = len(hamiltonian.one_body_tensor)
    normal_ordered_hamiltonian = (
            openfermion.normal_ordered(hamiltonian))

    gates = trotterize(hamiltonian)

    other_hamiltonian = untrotterize(n_modes, gates)
    other_normal_ordered_hamiltonian = (
            openfermion.normal_ordered(other_hamiltonian))
    normal_ordered_hamiltonian %= 2 * np.pi
    other_normal_ordered_hamiltonian %= 2 * np.pi
    assert normal_ordered_hamiltonian == other_normal_ordered_hamiltonian
