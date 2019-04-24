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

"""Common gates that target three qubits."""

from typing import Optional, Tuple

import numpy as np

import cirq
from cirq._compat import proper_repr

from openfermioncirq.gates import common_gates
from openfermioncirq.gates.four_qubit_gates import _canonicalize_weight


def rot111(rads: float):
    """Phases the |111> state of three qubits by e^{i rads}."""
    return cirq.CCZ**(rads / np.pi)


class CXXYYPowGate(cirq.EigenGate,
                         cirq.ThreeQubitGate):
    """Controlled XX + YY interaction."""

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs
                        ) -> Optional[np.ndarray]:
        return cirq.apply_unitary(
            cirq.ControlledGate(common_gates.XXYY**self.exponent),
            args,
            default=None)

    def _eigen_components(self):
        minus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]),
            np.array([[0.5, 0.5],
                         [0.5, 0.5]]),
            np.diag([0]))
        plus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]),
            np.array([[0.5, -0.5],
                         [-0.5, 0.5]]),
            np.diag([0]))

        return [(0, np.diag([1, 1, 1, 1, 1, 0, 0, 1])),
                (-0.5, minus_half_component),
                (0.5, plus_half_component)]

    def _decompose_(self, qubits):
        control, a, b = qubits
        yield cirq.CNOT(a, b)
        yield cirq.H(a)
        yield cirq.CCZ(control, a, b)**self.exponent
        # Note: Clifford optimization would merge this CZ into the CCZ decomp.
        yield cirq.CZ(control, b)**(-self.exponent / 2)
        yield cirq.H(a)
        yield cirq.CNOT(a, b)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=('@', 'XXYY', 'XXYY'),
            exponent=self._diagram_exponent(args))

    def __repr__(self):
        if self.exponent == 1:
            return 'CXXYY'
        return 'CXXYY**{!r}'.format(self.exponent)


class CYXXYPowGate(cirq.EigenGate,
                         cirq.ThreeQubitGate):
    """Controlled YX - XY interaction."""

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs
                        ) -> Optional[np.ndarray]:
        return cirq.apply_unitary(
            cirq.ControlledGate(common_gates.YXXY**self.exponent),
            args,
            default=None)

    def _eigen_components(self):
        minus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]),
            np.array([[0.5, -0.5j],
                         [0.5j, 0.5]]),
            np.diag([0]))
        plus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]),
            np.array([[0.5, 0.5j],
                         [-0.5j, 0.5]]),
            np.diag([0]))

        return [(0, np.diag([1, 1, 1, 1, 1, 0, 0, 1])),
                (-0.5, minus_half_component),
                (0.5, plus_half_component)]

    def _decompose_(self, qubits):
        control, a, b = qubits
        yield cirq.CNOT(a, b)
        yield cirq.X(a)**0.5
        yield cirq.CCZ(control, a, b)**self.exponent
        # Note: Clifford optimization would merge this CZ into the CCZ decomp.
        yield cirq.CZ(control, b)**(-self.exponent / 2)
        yield cirq.X(a)**-0.5
        yield cirq.CNOT(a, b)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=('@', 'YXXY', '#2'),
            exponent=self._diagram_exponent(args))

    def __repr__(self):
        if self.exponent == 1:
            return 'CYXXY'
        return 'CYXXY**{!r}'.format(self.exponent)


def CRxxyy(rads: float) -> CXXYYPowGate:
    """Controlled version of ofc.Rxxyy"""
    return CXXYYPowGate(exponent=2 * rads / np.pi)


def CRyxxy(rads: float) -> CYXXYPowGate:
    """Controlled version of ofc.Ryxxy"""
    return CYXXYPowGate(exponent=2 * rads / np.pi)


CXXYY = CXXYYPowGate()
CYXXY = CYXXYPowGate()


@cirq.value_equality(approximate=True)
class CubicFermionicSimulationGate(
        cirq.EigenGate,
        cirq.ThreeQubitGate):
    """w0 * |110><101| + w1 * |110><011| + w2 * |101><011| + hc interaction.

    With weights (w0, w1, w2) and exponent t, this gate's matrix is defined as
    follows:
        exp(-i π 0.5 t (w0 |110><101| + h.c.) +
                        w1 |110><011| + h.c.) +
                        w2 |101><011| + h.c.)))

    Args:
        weights: The weights of the terms in the Hamiltonian.
    """

    def __init__(self,
                 weights: Tuple[complex, complex, complex]=(1., 1., 1.),
                 **kwargs) -> None:

        assert len(weights) == 3
        self.weights = weights

        super().__init__(**kwargs)

    def _eigen_components(self):
        components = [(0, np.diag([1, 1, 1, 0, 1, 0, 0, 1]))]
        nontrivial_part = np.zeros((3, 3), dtype=np.complex128)
        for ij, w in zip([(1, 2), (0, 2), (0, 1)], self.weights):
            nontrivial_part[ij] = w
            nontrivial_part[ij[::-1]] = w.conjugate()
        assert(np.allclose(nontrivial_part, nontrivial_part.conj().T))
        eig_vals, eig_vecs = np.linalg.eigh(nontrivial_part)
        for eig_val, eig_vec in zip(eig_vals, eig_vecs.T):
            exp_factor = -0.5 * eig_val
            proj = np.zeros((8, 8), dtype=np.complex128)
            nontrivial_indices = np.array([3, 5, 6], dtype=np.intp)
            proj[nontrivial_indices[:, np.newaxis], nontrivial_indices] = (
                    np.outer(eig_vec.conjugate(), eig_vec))
            components.append((exp_factor, proj))
        return components

    def _value_equality_values_(self):
        return tuple(_canonicalize_weight(w * self.exponent)
                for w in list(self.weights) + [self._global_shift])

    def _is_parameterized_(self) -> bool:
        return any(cirq.is_parameterized(v)
                for V in self._value_equality_values_()
                for v in V)

    def __repr__(self):
        return (
            'ofc.CubicFermionicSimulationGate(' +
            '({})'.format(' ,'.join(proper_repr(w) for w in self.weights)) +
            ('' if self.exponent == 1 else
             (', exponent=' + proper_repr(self.exponent))) +
            ('' if self._global_shift == 0 else
             (', global_shift=' + proper_repr(self._global_shift))) +
            ')')
