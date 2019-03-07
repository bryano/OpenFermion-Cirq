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
from typing import Dict, Tuple

import cirq
import numpy as np
import openfermion

import pytest

import openfermioncirq as ofc
from openfermioncirq.trotter.algorithms.general_swap_network import (
        trotterize)

def normal_ordered_interaction_operator(
        operator: openfermion.InteractionOperator
        ) -> openfermion.InteractionOperator:
    constant = operator.constant
    one_body_tensor = operator.one_body_tensor.copy()
    two_body_tensor = np.zeros_like(operator.two_body_tensor)
    for indices in itertools.product(*(
        range(d) for d in two_body_tensor.shape)):
        normal_indices = tuple(sorted(indices[:2]) + sorted(indices[2:]))
        two_body_tensor[normal_indices] += (
                operator.two_body_tensor[indices])
    return openfermion.InteractionOperator(
            constant, one_body_tensor, two_body_tensor)


def untrotterize(n_modes: int, gates: Dict[Tuple[int, ...], cirq.Gate]):
    one_body_tensor = np.zeros((n_modes,) * 2)
    two_body_tensor = np.zeros((n_modes,) * 4)

    global_shift = 0

    for indices, gate in gates.items():
        if isinstance(gate, cirq.ZPowGate):
            coeff = -gate._exponent * np.pi
            global_shift += gate._exponent * gate._global_shift * np.pi
            one_body_tensor[indices * 2] += coeff
        elif isinstance(gate, ofc.CombinedSwapAndZ):
            weights = tuple(-w * gate._exponent for w in gate.weights)
            global_shift += gate._exponent * gate._global_shift
            one_body_tensor[indices] += weights[0]
            two_body_tensor[indices * 2] += weights[1]
        elif isinstance(gate, ofc.CombinedCXXYYPowGate):
            weights = tuple(
                    -0.5 * np.pi * w * gate._exponent for w in gate.weights)
            global_shift += gate._exponent * gate._global_shift
            p, q, r = indices
            two_body_tensor[p, q, p, r] += weights[0]
            two_body_tensor[p, q, q, r] += weights[1]
            two_body_tensor[p, r, q, r] += weights[2]
        elif isinstance(gate, ofc.CombinedDoubleExcitationGate):
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


def random_symmetric_matrix(n):
    m = np.random.standard_normal((n, n))
    return (m + m.T) / 2.


def random_interaction_operator(n_modes):
    constant = 0
    one_body_tensor = random_symmetric_matrix(n_modes)
    one_body_tensor = np.zeros((n_modes,) * 2)
    two_body_tensor = (
            random_symmetric_matrix(n_modes ** 2).reshape((n_modes,) * 4))
    two_body_tensor = np.zeros((n_modes,) * 4)
    for p, q in itertools.combinations(range(n_modes), 2):
        two_body_tensor[p, q, p, q] = 1
    return openfermion.InteractionOperator(
            constant, one_body_tensor, two_body_tensor)


@pytest.mark.parametrize('constant,potential',
    [(1, 1), (0, 1), (0, 0.3), (-0.5, 0.7)])
def test_untrotterize_linear(constant, potential):
    # exp(i pi e (v + s))
    exponent = -potential / np.pi
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
        [random_interaction_operator(2) for _ in range(10)])
def test_trotterize(hamiltonian):
    n_modes = len(hamiltonian.one_body_tensor)
    normal_ordered_hamiltonian = (
            normal_ordered_interaction_operator(hamiltonian))

    gates = trotterize(hamiltonian)

    other_hamiltonian = untrotterize(n_modes, gates)
    other_normal_ordered_hamiltonian = (
            normal_ordered_interaction_operator(other_hamiltonian))
    assert hamiltonian == other_hamiltonian
