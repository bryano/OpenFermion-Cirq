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

import numpy as np
import pytest
import scipy.linalg as la
import sympy

import cirq
from cirq.testing import EqualsTester

import openfermioncirq as ofc


def test_apply_unitary_effect():
    cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents(
        ofc.CXXYY,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1, sympy.Symbol('s')])

    cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents(
        ofc.CYXXY,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1, sympy.Symbol('s')])


def test_cxxyy_eq():
    eq = EqualsTester()

    eq.add_equality_group(ofc.CXXYY**-0.5,
                          ofc.CXXYYPowGate(exponent=3.5),
                          ofc.CXXYYPowGate(exponent=-0.5))

    eq.add_equality_group(ofc.CXXYYPowGate(exponent=1.5),
                          ofc.CXXYYPowGate(exponent=-2.5))

    eq.make_equality_group(lambda: ofc.CXXYYPowGate(exponent=0))
    eq.make_equality_group(lambda: ofc.CXXYYPowGate(exponent=0.5))


@pytest.mark.parametrize('exponent', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cxxyy_decompose(exponent):
    cirq.testing.assert_decompose_is_consistent_with_unitary(
            ofc.CXXYY**exponent)


def test_cxxyy_repr():
    assert repr(ofc.CXXYY) == 'CXXYY'
    assert repr(ofc.CXXYY**0.5) == 'CXXYY**0.5'


def test_cyxxy_eq():
    eq = EqualsTester()

    eq.add_equality_group(ofc.CYXXY**-0.5,
                          ofc.CYXXYPowGate(exponent=3.5),
                          ofc.CYXXYPowGate(exponent=-0.5))

    eq.add_equality_group(ofc.CYXXYPowGate(exponent=1.5),
                          ofc.CYXXYPowGate(exponent=-2.5))

    eq.make_equality_group(lambda: ofc.CYXXYPowGate(exponent=0))
    eq.make_equality_group(lambda: ofc.CYXXYPowGate(exponent=0.5))


@pytest.mark.parametrize('exponent', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cyxxy_decompose(exponent):
    cirq.testing.assert_decompose_is_consistent_with_unitary(
            ofc.CYXXY**exponent)


def test_cyxxy_repr():
    assert repr(ofc.CYXXYPowGate(exponent=1)) == 'CYXXY'
    assert repr(ofc.CYXXYPowGate(exponent=0.5)) == 'CYXXY**0.5'


@pytest.mark.parametrize(
        'gate, initial_state, correct_state', [
            (ofc.CXXYY,
                np.array([0, 0, 0, 0, 0, 1, 1, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 0, -1j, -1j, 0]) / np.sqrt(2)),
            (ofc.CXXYY**0.5,
                np.array([0, 0, 0, 0, 1, 1, 0, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 1 / np.sqrt(2), 0.5, -0.5j, 0])),
            (ofc.CXXYY**-0.5,
                np.array([0, 0, 0, 0, 1, 1, 0, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 1 / np.sqrt(2), 0.5, 0.5j, 0])),
            (ofc.CXXYY,
                np.array([1 / np.sqrt(2), 0, 0, 0, 0, 0.5, 0.5, 0]),
                np.array([1 / np.sqrt(2), 0, 0, 0, 0, -0.5j, -0.5j, 0])),
            (ofc.CXXYY,
                np.array([0, 1, 1, 0, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([0, 1, 1, 0, 0, 0, 0, 0]) / np.sqrt(2)),
            (ofc.CXXYY**0.5,
                np.array([1, 1, 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([1, 1, 0, 0, 0, 0, 0, 0]) / np.sqrt(2)),
            (ofc.CXXYY**-0.5,
                np.array([1, 0, 0, 1, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([1, 0, 0, 1, 0, 0, 0, 0]) / np.sqrt(2)),
            (ofc.CYXXY,
                np.array([0, 0, 0, 0, 0, 1, 1, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 0, 1, -1, 0]) / np.sqrt(2)),
            (ofc.CYXXY**0.5,
                np.array([0, 0, 0, 0, 0, 1, 1, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 0, 0, 1, 0])),
            (ofc.CYXXY**-0.5,
                np.array([0, 0, 0, 0, 0, 1, 1, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 0, 1, 0, 0])),
            (ofc.CYXXY**-0.5,
                np.array([1 / np.sqrt(2), 0, 0, 0, 0, 0.5, 0.5, 0]),
                np.array([1, 0, 0, 0, 0, 1, 0, 0]) / np.sqrt(2)),
            (ofc.CYXXY,
                np.array([0, 1, 1, 0, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([0, 1, 1, 0, 0, 0, 0, 0]) / np.sqrt(2)),
            (ofc.CYXXY**0.5,
                np.array([1, 1, 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([1, 1, 0, 0, 0, 0, 0, 0]) / np.sqrt(2)),
            (ofc.CYXXY**-0.5,
                np.array([1, 0, 0, 1, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([1, 0, 0, 1, 0, 0, 0, 0]) / np.sqrt(2))
])
def test_three_qubit_rotation_gates_on_simulator(gate: cirq.Gate,
                                                 initial_state: np.ndarray,
                                                 correct_state: np.ndarray):
    op = gate(*cirq.LineQubit.range(3))
    result = cirq.Circuit.from_ops(op).apply_unitary_effect_to_state(
        initial_state, dtype=np.complex128)
    cirq.testing.assert_allclose_up_to_global_phase(result,
                                                    correct_state,
                                                    atol=1e-8)


@pytest.mark.parametrize('rads', [
    2*np.pi, np.pi, 0.5*np.pi, 0.25*np.pi, 0.1*np.pi, 0.0, -0.5*np.pi])
def test_crxxyy_unitary(rads):
    np.testing.assert_allclose(
            cirq.unitary(ofc.CRxxyy(rads)),
            cirq.unitary(cirq.ControlledGate(ofc.Rxxyy(rads))),
            atol=1e-8)


@pytest.mark.parametrize('rads', [
    2*np.pi, np.pi, 0.5*np.pi, 0.25*np.pi, 0.1*np.pi, 0.0, -0.5*np.pi])
def test_cryxxy_unitary(rads):
    np.testing.assert_allclose(
            cirq.unitary(ofc.CRyxxy(rads)),
            cirq.unitary(cirq.ControlledGate(ofc.Ryxxy(rads))),
            atol=1e-8)


def test_three_qubit_gate_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    circuit = cirq.Circuit.from_ops(
        ofc.CXXYY(a, b, c),
        ofc.CYXXY(a, b, c))
    cirq.testing.assert_has_diagram(circuit, """
a: ───@──────@──────
      │      │
b: ───XXYY───YXXY───
      │      │
c: ───XXYY───#2─────
""")

    circuit = cirq.Circuit.from_ops(
        ofc.CXXYY(a, b, c)**-0.5,
        ofc.CYXXY(a, b, c)**-0.5)
    cirq.testing.assert_has_diagram(circuit, """
a: ───@───────────@─────────
      │           │
b: ───XXYY────────YXXY──────
      │           │
c: ───XXYY^-0.5───#2^-0.5───
""")


def test_combined_cxxyy():
    ofc.testing.assert_eigengate_implements_consistent_protocols(
        ofc.CombinedCXXYYPowGate)


def test_combined_cxxyy_equality():
    eq = EqualsTester()
    eq.add_equality_group(
        ofc.CombinedCXXYYPowGate() ** 0.5,
        ofc.CombinedCXXYYPowGate((1,) * 3, exponent=0.5),
        ofc.CombinedCXXYYPowGate((0.5,) * 3)
        )
    eq.add_equality_group(
        ofc.CombinedCXXYYPowGate((1j, 0, 0)),
        ofc.CombinedCXXYYPowGate((5j, 0, 0))
        )
    eq.add_equality_group(
        ofc.CombinedCXXYYPowGate((sympy.Symbol('s'), 0, 0), exponent=2),
        ofc.CombinedCXXYYPowGate((2 * sympy.Symbol('s'), 0, 0), exponent=1)
        )


@pytest.mark.parametrize('exponent,control',
    itertools.product(
        [0, 1, -1, 0.25, -0.5, 0.1],
        [0, 1, 2]))
def test_combined_cxxyy_consistency_special(exponent, control):
    weights = tuple(np.eye(1, 3, control)[0])
    general_gate  = ofc.CombinedCXXYYPowGate(weights, exponent=exponent)
    general_unitary = cirq.unitary(general_gate)

    indices = np.dot(
            list(itertools.product((0, 1), repeat=3)),
            (2 ** np.roll(np.arange(3), -control))[::-1])
    special_gate = ofc.CXXYYPowGate(exponent=exponent)
    special_unitary = (
            cirq.unitary(special_gate)[indices[:, np.newaxis], indices])

    assert np.allclose(general_unitary, special_unitary)


@pytest.mark.parametrize('weights,exponent', [
    (np.random.uniform(-5, 5, 3) + 1j * np.random.uniform(-5, 5, 3),
        np.random.uniform(-5, 5)) for _ in range(5)
])
def test_combined_cxxyy_consistency_docstring(weights, exponent):
    generator = np.zeros((8, 8), dtype=np.complex128)
    # w0 |110><101| + h.c.
    generator[6, 5] = weights[0]
    generator[5, 6] = weights[0].conjugate()
    # w1 |110><011| + h.c.
    generator[6, 3] = weights[1]
    generator[3, 6] = weights[1].conjugate()
    # w2 |101><011| + h.c.
    generator[5, 3] = weights[2]
    generator[3, 5] = weights[2].conjugate()
    expected_unitary = la.expm(-0.5j * exponent * np.pi * generator)

    gate  = ofc.CombinedCXXYYPowGate(weights, exponent=exponent)
    actual_unitary = cirq.unitary(gate)

    assert np.allclose(expected_unitary, actual_unitary)
