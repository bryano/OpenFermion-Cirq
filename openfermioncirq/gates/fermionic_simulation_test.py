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
from typing import cast, Tuple

import numpy as np
import pytest
import scipy.linalg as la
import sympy

import cirq
import cirq.contrib.acquaintance as cca
import openfermioncirq as ofc
from openfermioncirq.gates.fermionic_simulation import (
        state_swap_eigen_component)


def test_state_swap_eigen_component_args():
    with pytest.raises(TypeError):
        state_swap_eigen_component(0, '12', 1)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', '01', 1)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', '10', 0)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', '100', 1)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', 'ab', 1)


@pytest.mark.parametrize('index_pair,n_qubits', [
    ((0, 1), 2),
    ((0, 3), 2),
    ])
def test_state_swap_eigen_component(index_pair, n_qubits):
    state_pair = tuple(format(i, '0' + str(n_qubits) + 'b') for i in index_pair)
    i, j = index_pair
    dim = 2 ** n_qubits
    for sign in (-1, 1):
        actual_component = state_swap_eigen_component(
                state_pair[0], state_pair[1], sign)
        expected_component = np.zeros((dim, dim))
        expected_component[i, i] = expected_component[j, j] = 0.5
        expected_component[i, j] = expected_component[j, i] = sign * 0.5
        assert np.allclose(actual_component, expected_component)

def random_real(size=None, mag=20):
    return np.random.uniform(-mag, mag, size)

def random_complex(size=None, mag=20):
    return random_real(size, mag) + 1j * random_real(size, mag)

def random_fermionic_simulation_gate(order):
    exponent = random_real()
    if order == 2:
        weights = (random_complex(), random_real())
        return ofc.QuadraticFermionicSimulationGate(weights, exponent=exponent)
    weights = random_complex(3)
    if order == 3:
        return ofc.CubicFermionicSimulationGate(weights, exponent=exponent)
    if order == 4:
        return ofc.QuarticFermionicSimulationGate(weights, exponent=exponent)

def assert_symbolic_decomposition_consistent(gate):
    expected_unitary = cirq.unitary(gate)

    weights = tuple(sympy.Symbol(f'w{i}') for i in range(gate.num_weights))
    exponent = sympy.Symbol('t')
    symbolic_gate = type(gate)(weights, exponent=exponent)
    qubits = cirq.LineQubit.range(gate.num_qubits())
    circuit = cirq.Circuit.from_ops(symbolic_gate._decompose_(qubits))
    resolver = {'t': gate.exponent}
    for i, w in enumerate(gate.weights):
        resolver[f'w{i}'] = w
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)
    decomp_unitary = resolved_circuit.to_unitary_matrix(qubit_order=qubits)

    assert np.allclose(expected_unitary, decomp_unitary)

def assert_fswap_consistent(gate):
    n_qubits = gate.num_qubits()
    for i in range(n_qubits - 1):
        fswap = cirq.kron(np.eye(1 << i), cirq.unitary(ofc.FSWAP),
                np.eye(1 << (n_qubits - i - 2)))
        assert fswap.shape == (1 << n_qubits,) * 2
        generator = gate.generator
        fswapped_generator = np.linalg.multi_dot([fswap, generator, fswap])
        gate.fswap(i)
        assert np.allclose(gate.generator, fswapped_generator)


def assert_permute_consistent(gate):
    n_qubits = gate.num_qubits()
    qubits = cirq.LineQubit.range(n_qubits)
    for pos in itertools.permutations(range(n_qubits)):
        permuted_gate = gate.__copy__()
        gate.permute(pos)
        actual_unitary = cirq.unitary(permuted_gate)

        ops = [
            cca.LinearPermutationGate(n_qubits,
                dict(zip(range(n_qubits), pos)), ofc.FSWAP)(*qubits),
            gate(*qubits),
            cca.LinearPermutationGate(n_qubits,
                dict(zip(pos, range(n_qubits))), ofc.FSWAP)(*qubits)
            ]
        circuit = cirq.Circuit.from_ops(ops)
        expected_unitary = cirq.unitary(circuit)
        assert np.allclose(actual_unitary, expected_unitary)


random_quadratic_gates = [
        random_fermionic_simulation_gate(2) for _ in range(5)]
manual_quadratic_gates = [ofc.QuadraticFermionicSimulationGate(weights)
        for weights in
        [cast(Tuple[float, float], (1, 1)), (1, 0), (0, 1), (0, 0)]]
quadratic_gates = random_quadratic_gates + manual_quadratic_gates
cubic_gates = ([ofc.CubicFermionicSimulationGate()] +
    [random_fermionic_simulation_gate(3) for _ in range(5)])
quartic_gates = ([ofc.QuarticFermionicSimulationGate()] +
        [random_fermionic_simulation_gate(4) for _ in range(5)])
gates = quadratic_gates + cubic_gates + quartic_gates

@pytest.mark.parametrize('gate', gates)
def test_fermionic_simulation_gate(gate):
    ofc.testing.assert_implements_consistent_protocols(gate)

    generator = gate.generator
    expected_unitary = la.expm(-1j * gate.exponent * generator)
    actual_unitary = cirq.unitary(gate)
    assert np.allclose(expected_unitary, actual_unitary)

    assert_fswap_consistent(gate)
    assert_permute_consistent(gate)


@pytest.mark.parametrize('weights', np.random.rand(10, 3))
def test_weights_and_exponent(weights):
    exponents = np.linspace(-1, 1, 8)
    gates = tuple(
        ofc.QuarticFermionicSimulationGate(weights / exponent,
                                         exponent=exponent,
                                         absorb_exponent=True)
        for exponent in exponents)

    for g1, g2 in itertools.combinations(gates, 2):
        assert cirq.approx_eq(g1, g2, atol=1e-100)

    for i, (gate, exponent) in enumerate(zip(gates, exponents)):
        assert gate.exponent == 1
        new_exponent = exponents[-i]
        new_gate = gate._with_exponent(new_exponent)
        assert new_gate.exponent == new_exponent


@pytest.mark.parametrize('gate', random_quadratic_gates)
def test_quadratic_fermionic_simulation_gate_symbolic_decompose(gate):
    assert_symbolic_decomposition_consistent(gate)


def test_cubic_fermionic_simulation_gate_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        ofc.CubicFermionicSimulationGate() ** 0.5,
        ofc.CubicFermionicSimulationGate((1,) * 3, exponent=0.5),
        ofc.CubicFermionicSimulationGate((0.5,) * 3)
        )
    eq.add_equality_group(
        ofc.CubicFermionicSimulationGate((1j, 0, 0)),
        ofc.CubicFermionicSimulationGate(((1 + 2 * np.pi) * 1j, 0, 0))
        )
    eq.add_equality_group(
        ofc.CubicFermionicSimulationGate((sympy.Symbol('s'), 0, 0), exponent=2),
        ofc.CubicFermionicSimulationGate(
            (2 * sympy.Symbol('s'), 0, 0), exponent=1)
        )
    eq.add_equality_group(
        ofc.CubicFermionicSimulationGate((0, 0.7, 0), global_shift=2),
        ofc.CubicFermionicSimulationGate(
            (0, 0.35, 0), global_shift=1, exponent=2)
        )


@pytest.mark.parametrize('exponent,control',
    itertools.product(
        [0, 1, -1, 0.25, -0.5, 0.1],
        [0, 1, 2]))
def test_cubic_fermionic_simulation_gate_consistency_special(exponent, control):
    weights = tuple(np.eye(1, 3, control)[0] * 0.5 * np.pi)
    general_gate  = ofc.CubicFermionicSimulationGate(weights, exponent=exponent)
    general_unitary = cirq.unitary(general_gate)

    indices = np.dot(
            list(itertools.product((0, 1), repeat=3)),
            (2 ** np.roll(np.arange(3), -control))[::-1])
    special_gate = ofc.CXXYYPowGate(exponent=exponent)
    special_unitary = (
            cirq.unitary(special_gate)[indices[:, np.newaxis], indices])

    assert np.allclose(general_unitary, special_unitary)


quartic_fermionic_simulation_simulator_test_cases = [
        (ofc.QuarticFermionicSimulationGate((0, 0, 0)), 1.,
         np.ones(16) / 4.,
         np.ones(16) / 4.,
         5e-6),
        (ofc.QuarticFermionicSimulationGate((0.2, -0.1, 0.7)), 0.,
         np.array([1, -1, -1, -1, -1, -1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1]) / 4.,
         np.array([1, -1, -1, -1, -1, -1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1]) / 4.,
         5e-6),
        (ofc.QuarticFermionicSimulationGate((0.2, -0.1, 0.7)), 0.3,
         np.array([1, -1, -1, -1, -1, -1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1]) / 4.,
         np.array([1, -1, -1, -np.exp(0.21j),
                      -1, -np.exp(-0.03j),
                      np.exp(-0.06j), 1,
                      1, np.exp(-0.06j),
                      np.exp(-0.03j), 1,
                      np.exp(0.21j), 1, 1, 1]) / 4.,
         5e-6),
        (ofc.QuarticFermionicSimulationGate((1. / 3, 0, 0)), 1.,
         np.array([0, 0, 0, 0, 0, 0, 1., 0,
                      0, 1., 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
         np.array([0, 0, 0, 0, 0, 0, 1., 0,
                      0, 1., 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
         5e-6),
        (ofc.QuarticFermionicSimulationGate((0, np.pi / 3, 0)), 1.,
         np.array([1., 1., 0, 0, 0, 1., 0, 0,
                      0, 0., -1., 0, 0, 0, 0, 0]) / 2.,
         np.array([1., 1., 0, 0, 0, -np.exp(4j * np.pi / 3), 0, 0,
                      0, 0., -np.exp(1j * np.pi / 3), 0, 0, 0, 0, 0]
                     ) / 2.,
         5e-6),
        (ofc.QuarticFermionicSimulationGate((0, 0, -np.pi / 2)), 1.,
         np.array([0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1., 0, 0, 0]),
         np.array([0, 0, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0]),
         5e-6),
        (ofc.QuarticFermionicSimulationGate((0, 0, -0.25 * np.pi)), 1.,
         np.array([0, 0, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 1j, 0, 0, 0]) / np.sqrt(2),
         5e-6),
        (ofc.QuarticFermionicSimulationGate(
            (-np.pi / 4, np.pi /6, -np.pi / 2)), 1.,
         np.array([0, 0, 0, 0, 0, 0, 1, 0,
                      0, 0, 1, 0, 1, 0, 0, 0]) / np.sqrt(3),
         np.array([0, 0, 0, 1j, 0, -1j / 2., 1 / np.sqrt(2), 0,
                      0, 1j / np.sqrt(2), np.sqrt(3) / 2, 0, 0, 0, 0, 0]
                     ) / np.sqrt(3),
         5e-6),
        ]
@pytest.mark.parametrize(
    'gate, exponent, initial_state, correct_state, atol',
    quartic_fermionic_simulation_simulator_test_cases)
def test_quartic_fermionic_simulation_on_simulator(
        gate, exponent, initial_state, correct_state, atol):

    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit.from_ops(gate(a, b, c, d)**exponent)
    result = circuit.apply_unitary_effect_to_state(initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
        result, correct_state, atol=atol)


def test_quartic_fermionic_simulation_eq():
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(
            ofc.QuarticFermionicSimulationGate((1.2, 0.4, -0.4), exponent=0.5),
            ofc.QuarticFermionicSimulationGate((0.3, 0.1, -0.1), exponent=2),
            ofc.QuarticFermionicSimulationGate((-0.6, -0.2, 0.2), exponent=-1),
            ofc.QuarticFermionicSimulationGate((0.6, 0.2, 2 * np.pi - 0.2)),
            )

    eq.make_equality_group(
            lambda: ofc.QuarticFermionicSimulationGate(
                (0.1, -0.3, 0.0), exponent=0.0))
    eq.make_equality_group(
            lambda: ofc.QuarticFermionicSimulationGate(
                (1., -1., 0.5), exponent=0.75))


def test_quartic_fermionic_simulation_gate_text_diagram():
    gate = ofc.QuarticFermionicSimulationGate((1,1,1))
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.Circuit.from_ops(
            [gate(*qubits[:4]), gate(*qubits[-4:])])

    actual_text_diagram = circuit.to_text_diagram()
    expected_text_diagram = """
0: ───⇊⇈────────
      │
1: ───⇊⇈────────
      │
2: ───⇊⇈───⇊⇈───
      │    │
3: ───⇊⇈───⇊⇈───
           │
4: ────────⇊⇈───
           │
5: ────────⇊⇈───
    """.strip()
    assert actual_text_diagram == expected_text_diagram

    actual_text_diagram = circuit.to_text_diagram(use_unicode_characters=False)
    expected_text_diagram = """
0: ---a*a*aa------------
      |
1: ---a*a*aa------------
      |
2: ---a*a*aa---a*a*aa---
      |        |
3: ---a*a*aa---a*a*aa---
               |
4: ------------a*a*aa---
               |
5: ------------a*a*aa---
    """.strip()
    assert actual_text_diagram == expected_text_diagram
