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

from typing import Optional, Tuple, Union

import cirq
import numpy as np
import scipy.linalg as la
import sympy

from openfermioncirq.gates.common_gates import XXYYPowGate


def _arg(x):
    if x == 0:
        return 0
    if cirq.is_parameterized(x):
        return sympy.arg(x)
    return np.angle(x)


def _canonicalize_weight(w):
    if w == 0:
        return (0, 0)
    if cirq.is_parameterized(w):
        return (cirq.PeriodicValue(abs(w), 4), sympy.arg(w))
    return (np.round((w % 4) if (w == np.real(w)) else
        (abs(w) % 4) * w / abs(w), 8), 0)


def state_swap_eigen_component(x: str, y: str, sign: int = 1, angle: float=0):
    """The +/- eigen-component of the operation that swaps states x and y.

    For example, state_swap_eigen_component('01', '10', ±1) with angle θ returns
        ┌                               ┐
        │0, 0,           0,            0│
        │0, 0.5,         ±0.5 e^{-iθ}, 0│
        │0, ±0.5 e^{iθ}, 0.5,          0│
        │0, 0,           0,            0│
        └                               ┘

    Args:
        x: The first state to swap, as a bitstring.
        y: The second state to swap, as a bitstring. Must have high index than
            x.
        sign: The sign of the off-diagonal elements (indicated by +/-1).
        angle: The phase of the complex off-diagonal elements. Defaults to 0.

    Returns: The eigen-component.

    Raises:
        ValueError:
            * x and y have different lengths
            * x or y contains a character other than '0' and '1'
            * x and y are the same
            * sign is not -1 or 1
        TypeError: x or y is not a string
    """
    if not (isinstance(x, str) and isinstance(y, str)):
        raise TypeError('not (isinstance(x, str) and isinstance(y, str))')
    if len(x) != len(y):
        raise ValueError('len(x) != len(y)')
    if set(x).union(y).difference('01'):
        raise ValueError('Arguments must be 0-1 strings.')
    if x == y:
        raise ValueError('x == y')
    if sign not in (-1, 1):
        raise ValueError('sign not in (-1, 1)')

    dim = 2 ** len(x)
    i, j = int(x, 2), int(y, 2)

    component = np.zeros((dim, dim), dtype=np.complex128)
    component[i, i] = component[j, j] = 0.5
    component[j, i]= sign * 0.5 * 1j**(angle * 2 / np.pi)
    component[i, j]= sign * 0.5 * 1j**(-angle * 2 / np.pi)
    return component


class QuadraticFermionicSimulationGate(
        cirq.EigenGate,
        cirq.InterchangeableQubitsGate,
        cirq.TwoQubitGate):
    """(w0 |10><01| + h.c.) + w1 * |11><11| interaction.

    With weights (w0, w1) and exponent t, this gate's matrix is defined as
    follows:
        exp(-i t ((w0 |10><01| + h.c.) + w1 |11><11|))
    """

    def __init__(self,
                 weights: Tuple[float, float]=(1, 1),
                 **kwargs) -> None:
        self.weights = weights

        super().__init__(**kwargs)

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        r = 2 * abs(self.weights[0]) / np.pi
        theta = _arg(self.weights[0]) / np.pi
        yield cirq.Z(qubits[0]) ** -theta
        yield XXYYPowGate(exponent=r * self.exponent)(*qubits)
        yield cirq.Z(qubits[0]) ** theta
        yield cirq.CZPowGate(
                exponent=-self.weights[1] * self.exponent / np.pi)(*qubits)

    def _eigen_components(self):
        components = [
            (0, np.diag([1, 0, 0, 0])),
            (-self.weights[1] / np.pi, np.diag([0, 0, 0, 1]))
            ]
        r = abs(self.weights[0]) / np.pi
        theta = 2 * _arg(self.weights[0]) / np.pi
        for s in (-1, 1):
            components.append((-s * r,
                np.array([
                    [0, 0, 0, 0],
                    [0, 1, s * 1j**(-theta), 0],
                    [0, s * 1j**(theta), 1, 0],
                    [0, 0, 0, 0]
                ]) / 2))
        return components

    def __repr__(self):
        exponent_str = ('' if self.exponent == 1 else
                ', exponent=' + cirq._compat.proper_repr(self.exponent))
        return ('ofc.QuadraticFermionicSimulationGate(({}){})'.format(
                ', '.join(cirq._compat.proper_repr(v) for v in self.weights),
                exponent_str))


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
            '({})'.format(' ,'.join(
                cirq._compat.proper_repr(w) for w in self.weights)) +
            ('' if self.exponent == 1 else
             (', exponent=' + cirq._compat.proper_repr(self.exponent))) +
            ('' if self._global_shift == 0 else
             (', global_shift=' + cirq._compat.proper_repr(self._global_shift))) +
            ')')


@cirq.value_equality(approximate=True)
class QuarticFermionicSimulationGate(cirq.EigenGate):
    """Rotates Hamming-weight 2 states into their bitwise complements.

    For weights (t0, t1, t2), is equivalent to
        exp(0.5 π i (t0 |1001><0110| + h.c.) +
                      t1 |1010><0101| + h.c.) +
                      t2 |1100><0011| + h.c.)))
    """

    def __init__(self,
                 weights: Tuple[complex, complex, complex]=(1, 1, 1),
                 absorb_exponent: bool=True,
                 *,  # Forces keyword args.
                 exponent: Optional[Union[sympy.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None
                 ) -> None:
        """Initialize the gate.

        At most one of exponent, rads, degs, or duration may be specified.
        If more are specified, the result is considered ambiguous and an
        error is thrown. If no argument is given, the default value of one
        half-turn is used.

        Args:
            weights: The weights of the terms in the Hamiltonian.
            absorb_exponent: Whether to absorb the given exponent into the
                weights. If true, the exponent of the returned gate is 1.
            exponent: The exponent angle, in half-turns.
            rads: The exponent angle, in radians.
            degs: The exponent angle, in degrees.
            duration: The exponent as a duration of time.
        """

        assert len(weights) == 3
        self.weights = weights

        if len([1 for e in [exponent, rads, degs, duration]
                if e is not None]) > 1:
            raise ValueError('Redundant exponent specification. '
                             'Use ONE of exponent, rads, degs, or duration.')

        if duration is not None:
            exponent = 2 * duration / np.pi
        else:
            exponent = cirq.chosen_angle_to_half_turns(
                half_turns=exponent,
                rads=rads,
                degs=degs)

        super().__init__(exponent=exponent)

        if absorb_exponent:
            self.absorb_exponent_into_weights()

    def num_qubits(self):
        return 4

    def _eigen_components(self):
        # projector onto subspace spanned by basis states with
        # Hamming weight != 2
        zero_component = np.diag([int(bin(i).count('1') != 2)
                                  for i in range(16)])

        state_pairs = (('0110', '1001'),
                       ('0101', '1010'),
                       ('0011', '1100'))

        plus_minus_components = tuple(
            (abs(weight) * sign / 2,
             state_swap_eigen_component(
                 state_pair[0], state_pair[1], sign, np.angle(weight)))
            for weight, state_pair in zip(self.weights, state_pairs)
            for sign in (-1, 1))

        return ((0, zero_component),) + plus_minus_components

    def _with_exponent(self,
                       exponent: Union[sympy.Symbol, float]
                       ) -> 'QuarticFermionicSimulationGate':
        gate = QuarticFermionicSimulationGate(self.weights)
        gate._exponent = exponent
        return gate

    def _decompose_(self, qubits):
        if self._is_parameterized_():
            return NotImplemented

        individual_rotations = [
            la.expm(-0.25j * self.exponent * np.pi * np.array([
                [np.real(w), 1j * s * np.imag(w)],
                [-1j * s * np.imag(w), -np.real(w)]]))
            for s, w in zip([1, -1, -1], self.weights)]

        combined_rotations = {}
        combined_rotations[0] = la.sqrtm(np.linalg.multi_dot([
                la.inv(individual_rotations[1]),
                individual_rotations[0],
                individual_rotations[2]]))
        combined_rotations[1] = la.inv(combined_rotations[0])
        combined_rotations[2] = np.linalg.multi_dot([
                la.inv(individual_rotations[0]),
                individual_rotations[1],
                combined_rotations[0]])
        combined_rotations[3] = individual_rotations[0]

        controlled_rotations = {i: cirq.ControlledGate(
            cirq.SingleQubitMatrixGate(combined_rotations[i]))
            for i in range(4)}

        a, b, c, d = qubits

        basis_change = list(cirq.flatten_op_tree([
            cirq.CNOT(b, a),
            cirq.CNOT(c, b),
            cirq.CNOT(d, c),
            cirq.CNOT(c, b),
            cirq.CNOT(b, a),
            cirq.CNOT(a, b),
            cirq.CNOT(b, c),
            cirq.CNOT(a, b),
            [cirq.X(c), cirq.X(d)],
            [cirq.CNOT(c, d), cirq.CNOT(d, c)],
            [cirq.X(c), cirq.X(d)],
            ]))

        controlled_rotations = list(cirq.flatten_op_tree([
            controlled_rotations[0](b, c),
            cirq.CNOT(a, b),
            controlled_rotations[1](b, c),
            cirq.CNOT(b, a),
            cirq.CNOT(a, b),
            controlled_rotations[2](b, c),
            cirq.CNOT(a, b),
            controlled_rotations[3](b, c)
            ]))

        controlled_swaps = [
            [cirq.CNOT(c, d), cirq.H(c)],
            cirq.CNOT(d, c),
            controlled_rotations,
            cirq.CNOT(d, c),
            [cirq.inverse(op) for op in reversed(controlled_rotations)],
            [cirq.H(c), cirq.CNOT(c, d)],
            ]

        yield basis_change
        yield controlled_swaps
        yield basis_change[::-1]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters:
            wire_symbols = ('⇊⇈',) * 4
        else:
            wire_symbols = ('a*a*aa',) * 4
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols,
                                       exponent=self._diagram_exponent(args))

    def absorb_exponent_into_weights(self):
        new_weights = []
        for weight in self.weights:
            if not weight:
                new_weights.append(weight)
                continue
            old_abs = abs(weight)
            new_abs = (old_abs * self._exponent) % 4
            new_weights.append(weight * new_abs / old_abs)
        self.weights = tuple(new_weights)
        self._exponent = 1

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs
                        ) -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return NotImplemented

        am, bm, cm = (la.expm(0.5j * self.exponent * np.pi *
                      np.array([[0, w], [w.conjugate(), 0]]))
                      for w in self.weights)

        a1 = args.subspace_index(0b1001)
        b1 = args.subspace_index(0b0101)
        c1 = args.subspace_index(0b0011)

        a2 = args.subspace_index(0b0110)
        b2 = args.subspace_index(0b1010)
        c2 = args.subspace_index(0b1100)

        cirq.apply_matrix_to_slices(args.target_tensor,
                                    am,
                                    slices=[a1, a2],
                                    out=args.available_buffer)
        cirq.apply_matrix_to_slices(args.available_buffer,
                                    bm,
                                    slices=[b1, b2],
                                    out=args.target_tensor)
        return cirq.apply_matrix_to_slices(args.target_tensor,
                                           cm,
                                           slices=[c1, c2],
                                           out=args.available_buffer)

    def _value_equality_values_(self):
        return tuple(_canonicalize_weight(w * self.exponent)
                for w in list(self.weights) + [self._global_shift])

    def _is_parameterized_(self) -> bool:
        return any(cirq.is_parameterized(v)
                for V in self._value_equality_values_()
                for v in V)

    def __repr__(self):
        return (
            'ofc.QuarticFermionicSimulationGate(({}), '
            'absorb_exponent=False, '
            'exponent={})'.format(
                ', '.join(cirq._compat.proper_repr(v) for v in self.weights),
                cirq._compat.proper_repr(self.exponent)))
