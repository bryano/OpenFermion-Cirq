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

"""Gates that target four qubits."""

from typing import Optional, Union

import numpy

import cirq


class LocalPQRSGate(cirq.EigenGate,
                    cirq.CompositeGate,
                    cirq.TextDiagrammable):
    """Evolve under |0011><1100| + h.c. for some time."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None) -> None:
        """Initialize the gate.

        At most one of half_turns, rads, degs, or duration may be specified.
        If more are specified, the result is considered ambiguous and an
        error is thrown. If no argument is given, the default value of one
        half-turn is used.

        Args:
            half_turns: The exponent angle, in half-turns.
            rads: The exponent angle, in radians.
            degs: The exponent angle, in degrees.
            duration: The exponent as a duration of time.
        """

        if len([1 for e in [half_turns, rads, degs, duration]
                if e is not None]) > 1:
            raise ValueError('Redundant exponent specification. '
                             'Use ONE of half_turns, rads, degs, or duration.')

        if duration is not None:
            exponent = 2 * duration / numpy.pi
        else:
            exponent = cirq.value.chosen_angle_to_half_turns(
                half_turns=half_turns,
                rads=rads,
                degs=degs)

        super().__init__(exponent=exponent)

    @property
    def half_turns(self) -> Union[cirq.Symbol, float]:
        return self._exponent

    def _eigen_components(self):
        return [
            (-1, numpy.array(
                [[0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0] for i in range(3)] +
                [[0, 0, 0, 0.5, 0, 0, 0, 0,
                  0, 0, 0, 0, 0.5, 0, 0, 0]] +
                [[0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0] for i in range(8)] +
                [[0, 0, 0, 0.5, 0, 0, 0, 0,
                  0, 0, 0, 0, 0.5, 0, 0, 0]] +
                [[0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0] for i in range(3)])),
            (1, numpy.array(
                [[0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0] for i in range(3)] +
                [[0, 0, 0, 0.5, 0, 0, 0, 0,
                  0, 0, 0, 0, -0.5, 0, 0, 0]] +
                [[0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0] for i in range(8)] +
                [[0, 0, 0, -0.5, 0, 0, 0, 0,
                  0, 0, 0, 0, 0.5, 0, 0, 0]] +
                [[0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0] for i in range(3)])),
            (0, numpy.diag([1, 1, 1, 0, 1, 1, 1, 1,
                            1, 1, 1, 1, 0, 1, 1, 1]))
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 2

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]
                       ) -> 'LocalPQRSGate':
        return LocalPQRSGate(half_turns=exponent)

    def default_decompose(self, qubits):
        p, q, r, s = qubits

        yield cirq.CNOT(r, s)
        yield cirq.CNOT(q, p)
        yield cirq.CNOT(q, r)

        yield cirq.X(q) ** self.half_turns
        yield cirq.Z(q) ** 0.125
        yield cirq.CNOT(r, q)
        yield cirq.Z(q) ** -0.125

        yield cirq.CNOT(s, r)
        yield cirq.CNOT(r, q)
        yield cirq.CNOT(s, r)

        yield cirq.Z(q) ** 0.125
        yield cirq.CNOT(r, q)
        yield cirq.Z(q) ** -0.125
        yield cirq.X(p)
        yield cirq.CNOT(p, q)
        yield cirq.X(p)
        yield cirq.Z(q) ** 0.125
        yield cirq.CNOT(r, q)
        yield cirq.Z(q) ** -0.125

        yield cirq.CNOT(s, r)
        yield cirq.CNOT(r, q)
        yield cirq.CNOT(s, r)

        yield cirq.Z(q) ** 0.125
        yield cirq.CNOT(r, q)
        yield cirq.Z(q) ** -0.125
        yield cirq.X(q) ** -self.half_turns
        yield cirq.Z(q) ** 0.125
        yield cirq.CNOT(r, q)
        yield cirq.Z(q) ** -0.125

        yield cirq.CNOT(s, r)
        yield cirq.CNOT(r, q)
        yield cirq.CNOT(s, r)

        yield cirq.Z(q) ** 0.125
        yield cirq.CNOT(r, q)
        yield cirq.Z(q) ** -0.125
        yield cirq.X(p)
        yield cirq.CNOT(p, q)
        yield cirq.X(p)
        yield cirq.Z(q) ** 0.125
        yield cirq.CNOT(r, q)
        yield cirq.Z(q) ** -0.125

        yield cirq.CNOT(s, r)
        yield cirq.CNOT(r, q)
        yield cirq.CNOT(s, r)

        yield cirq.Z(q) ** 0.125
        yield cirq.CNOT(r, q)
        yield cirq.Z(q) ** -0.125

        yield cirq.CNOT(q, p)
        yield cirq.CNOT(q, r)
        yield cirq.CNOT(r, s)

    def text_diagram_info(self, args: cirq.TextDiagramInfoArgs
                          ) -> cirq.TextDiagramInfo:
        return cirq.TextDiagramInfo(
            wire_symbols=('P', 'Q', 'R', 'S'),
            exponent=self.half_turns)

    def __repr__(self):
        if self.half_turns == 1:
            return 'PQRS'
        return 'PQRS**{!r}'.format(self.half_turns)


PQRS = LocalPQRSGate()
