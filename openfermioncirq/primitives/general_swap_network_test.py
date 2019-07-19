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
from openfermioncirq.gates.fermionic_simulation import (
        fermionic_simulation_gates_from_interaction_operator,
        interaction_operator_from_fermionic_simulation_gates)
from openfermioncirq.primitives.general_swap_network import (
        GreedyExecutionStrategy,
        trotter_circuit, trotter_unitary)


@pytest.mark.parametrize('order,hamiltonian',
    [(order, ofc.testing.random_interaction_operator_term(order, real=False))
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
    [(order, openfermion.utils._testing_utils.random_interaction_operator(
        5, real=False))
     for order in (1, 2, 3, 4) for _ in range(2)])
def test_trotterize(order, hamiltonian):
    hamiltonian = hamiltonian.projected(order, True)

    qubits = cirq.LineQubit.range(hamiltonian.n_qubits)
    initial_mapping = dict(zip(qubits, range(hamiltonian.n_qubits)))
    swap_network = cca.complete_acquaintance_strategy(
            qubits, order, ofc.FSWAP)
    assert cca.uses_consistent_swap_gate(swap_network, ofc.FSWAP)
    cca.remove_redundant_acquaintance_opportunities(swap_network)
    cca.return_to_initial_mapping(swap_network, ofc.FSWAP)

    circuit = trotter_circuit(swap_network, initial_mapping, hamiltonian)

    actual_unitary = circuit.to_unitary_matrix(qubit_order=qubits)
    acquaintance_dag = cca.inspection_utils.get_acquaintance_dag(
            swap_network, initial_mapping)
    expected_unitary = trotter_unitary(acquaintance_dag, hamiltonian)

    assert np.allclose(actual_unitary, expected_unitary)


@pytest.mark.parametrize('constant,potential',
    [(1, 1), (0, 1), (0, 0.3), (-0.5, 0.7)] +
    [np.random.standard_normal(2) for _ in range(3)])
def test_untrotterize_linear(constant, potential):
    exponent = potential / np.pi
    global_shift = constant / (exponent * np.pi)
    gates = {
            (0,): cirq.ZPowGate(exponent=exponent, global_shift=global_shift)}
    hamiltonian = interaction_operator_from_fermionic_simulation_gates(2, gates)
    assert np.isclose(hamiltonian.constant, constant)
    assert np.allclose(hamiltonian.one_body_tensor, [[potential, 0], [0, 0]])
    assert np.allclose(hamiltonian.two_body_tensor, np.zeros((2,) * 4))


@pytest.mark.parametrize('constant,tunneling,interaction,scale',
    [(0,0,0,1), (1, 1, 1, 1), (0.2, -0.5, 1.7, 0.3), (-0.1, -0.5, 1, -1)] +
    [(np.random.standard_normal(),
      np.random.standard_normal() + 1j * np.random.standard_normal(),
      np.random.standard_normal(),
      np.random.standard_normal())
        for _ in range(3)])
def test_untrotterize_quadratic(constant, tunneling, interaction, scale):
    weights = (-tunneling, interaction)
    gate = ofc.QuadraticFermionicSimulationGate(
            weights, exponent=scale, global_shift=constant)
    gates = {(0, 1): gate}
    hamiltonian = interaction_operator_from_fermionic_simulation_gates(2, gates)
    expected_constant = constant * scale
    assert np.isclose(hamiltonian.constant, expected_constant)
    expected_one_body_tensor = np.zeros((2, 2), dtype=np.complex128)
    expected_one_body_tensor[0, 1] = tunneling
    expected_one_body_tensor[1, 0] = tunneling.conjugate()
    expected_one_body_tensor *= scale
    assert np.allclose(hamiltonian.one_body_tensor, expected_one_body_tensor)
    expected_two_body_tensor = np.zeros((2,) * 4, dtype=np.complex128)
    expected_two_body_tensor[0, 1, 0, 1] = interaction
    expected_two_body_tensor *= scale
    assert np.allclose(hamiltonian.two_body_tensor, expected_two_body_tensor)


@pytest.mark.parametrize('constant,coeffs,scale',
    [(np.random.standard_normal(),
      np.random.standard_normal(3) + 1j * np.random.standard_normal(3),
      np.random.standard_normal()) for _ in range(10)])
def test_untrotterize_cubic(constant, coeffs, scale):
    gate = ofc.CubicFermionicSimulationGate(
            weights=coeffs, exponent=scale, global_shift=constant)
    gates = {(0, 1, 2): gate}
    hamiltonian = interaction_operator_from_fermionic_simulation_gates(3, gates)
    expected_constant = constant * scale
    assert np.isclose(hamiltonian.constant, expected_constant)
    assert np.allclose(hamiltonian.one_body_tensor, np.zeros((3, 3)))
    expected_two_body_tensor = np.zeros((3,) * 4, dtype=np.complex128)
    composite_indices = [((0, 1), (0, 2)), ((0, 1), (1, 2)), ((0, 2), (1, 2))]
    for (pq, rs), w in zip(composite_indices, coeffs):
        expected_two_body_tensor[pq + rs] = w
        expected_two_body_tensor[rs + pq] = w.conjugate()
    expected_two_body_tensor *= scale
    assert np.allclose(hamiltonian.two_body_tensor, expected_two_body_tensor)


@pytest.mark.parametrize('constant,coeffs,scale',
    [(np.random.standard_normal(),
      np.random.standard_normal(3) + 1j * np.random.standard_normal(3),
      np.random.standard_normal()) for _ in range(10)])
def test_untrotterize_quartic(constant, coeffs, scale):
    gate = ofc.QuarticFermionicSimulationGate(
            weights=coeffs, exponent=scale, global_shift=constant)
    gates = {(0, 1, 2, 3): gate}
    hamiltonian = interaction_operator_from_fermionic_simulation_gates(4, gates)
    expected_constant = constant * scale
    assert np.isclose(hamiltonian.constant, expected_constant)
    assert np.allclose(hamiltonian.one_body_tensor, np.zeros((4, 4)))
    expected_two_body_tensor = np.zeros((4,) * 4, dtype=np.complex128)
    composite_indices = [((0, 3), (1, 2)), ((0, 2), (1, 3)), ((0, 1), (2, 3))]
    for (pq, rs), w in zip(composite_indices, coeffs):
        expected_two_body_tensor[pq + rs] = w
        expected_two_body_tensor[rs + pq] = w.conjugate()
    expected_two_body_tensor *= scale
    diff = np.abs(hamiltonian.two_body_tensor - expected_two_body_tensor
            ) / (2 * np.pi)
    assert np.allclose(diff, np.around(diff))


@pytest.mark.parametrize('hamiltonian',
    [openfermion.utils._testing_utils.random_interaction_operator(5, real=False)
    for _ in range(10)])
def test_untrotterize(hamiltonian):
    hamiltonian.constant = 0
    n_modes = len(hamiltonian.one_body_tensor)
    normal_ordered_hamiltonian = (
            openfermion.normal_ordered(hamiltonian))

    gates = fermionic_simulation_gates_from_interaction_operator(hamiltonian)

    other_hamiltonian = interaction_operator_from_fermionic_simulation_gates(
            n_modes, gates)
    other_normal_ordered_hamiltonian = (
            openfermion.normal_ordered(other_hamiltonian))
    diff = normal_ordered_hamiltonian - other_normal_ordered_hamiltonian
    assert np.allclose(diff.constant, 0)
    assert np.allclose(diff.one_body_tensor,
            np.zeros_like(diff.one_body_tensor))
    abs_two_body_diff = np.abs(diff.two_body_tensor) / (2 * np.pi)
    assert np.allclose(abs_two_body_diff, np.around(abs_two_body_diff))

def test_greedy_execution_strategy():
    qubits = cirq.LineQubit.range(2)
    swap_network = cirq.Circuit.from_ops(cca.acquaint(*qubits[:2]),
            device=cca.UnconstrainedAcquaintanceDevice)
    gates = {(0, 1): cirq.CNOT, (1, 0): cirq.CNOT}
    initial_mapping = {q: i for i, q in enumerate(qubits)}

    circuit = swap_network.copy()
    GreedyExecutionStrategy(gates, initial_mapping)(circuit)
    assert cca.uses_consistent_swap_gate(circuit, ofc.FSWAP)
    keep = lambda op: not isinstance(op.gate, cca.PermutationGate)
    circuit = cirq.Circuit.from_ops(cirq.decompose(circuit, keep=keep))
    expected_diagram = """
0: ───@───×ᶠ───@───×ᶠ───
      │   │    │   │
1: ───X───×ᶠ───X───×ᶠ───
""".strip()
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

    gates = {(2,): cirq.X}
    circuit = swap_network.copy()
    GreedyExecutionStrategy(gates, initial_mapping)(circuit)
    cirq.DropEmptyMoments()(circuit)
    assert not circuit
