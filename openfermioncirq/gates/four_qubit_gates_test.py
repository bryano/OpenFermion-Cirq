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

import numpy
import pytest

import cirq
from cirq.testing import EqualsTester

from openfermioncirq.gates import LocalPQRSGate, PQRS


def test_pqrs_repr():
    assert repr(LocalPQRSGate(half_turns=1)) == 'PQRS'
    assert repr(LocalPQRSGate(half_turns=0.5)) == 'PQRS**0.5'


def test_pqrs_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = LocalPQRSGate(half_turns=1.0, duration=numpy.pi/2)


def test_pqrs_eq():
    eq = EqualsTester()

    eq.add_equality_group(LocalPQRSGate(half_turns=1.5),
                          LocalPQRSGate(half_turns=-0.5),
                          LocalPQRSGate(rads=-0.5 * numpy.pi),
                          LocalPQRSGate(degs=-90),
                          LocalPQRSGate(duration=-0.5 * numpy.pi / 2))

    eq.add_equality_group(LocalPQRSGate(half_turns=0.5),
                          LocalPQRSGate(half_turns=-1.5),
                          LocalPQRSGate(rads=0.5 * numpy.pi),
                          LocalPQRSGate(degs=90),
                          LocalPQRSGate(duration=-1.5 * numpy.pi / 2))

    eq.make_equality_group(lambda: LocalPQRSGate(half_turns=0.0))
    eq.make_equality_group(lambda: LocalPQRSGate(half_turns=0.75))


@pytest.mark.parametrize('half_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_pqrs_decompose(half_turns):
    gate = PQRS ** half_turns
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)

    cirq.testing.assert_allclose_up_to_global_phase(
        matrix, gate.matrix(), atol=1e-7)


@pytest.mark.skip(reason="skip parametrized tests for now")
#@pytest.mark.parametrize(
    #'gate, half_turns, initial_state, correct_state, atol', [
        #(PQRS, 1.0,
            #numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
            #numpy.array([0, 0, 0, 0, 0, -1j, -1j, 0]) / numpy.sqrt(2),
            #5e-6),
        #(PQRS, 0.5,
            #numpy.array([0, 0, 0, 0, 1, 1, 0, 0]) / numpy.sqrt(2),
            #numpy.array([0, 0, 0, 0, 1 / numpy.sqrt(2), 0.5, -0.5j, 0]),
            #5e-6),
        #(PQRS, -0.5,
            #numpy.array([0, 0, 0, 0, 1, 1, 0, 0]) / numpy.sqrt(2),
            #numpy.array([0, 0, 0, 0, 1 / numpy.sqrt(2), 0.5, 0.5j, 0]),
            #5e-6),
        #(PQRS, 1.0,
            #numpy.array([1 / numpy.sqrt(2), 0, 0, 0, 0, 0.5, 0.5, 0]),
            #numpy.array([1 / numpy.sqrt(2), 0, 0, 0, 0, -0.5j, -0.5j, 0]),
            #5e-6),
        #(PQRS, 1.0,
            #numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
            #numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
            #5e-6),
        #(PQRS, 0.5,
            #numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
            #numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
            #5e-6),
        #(PQRS, -0.5,
            #numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
            #numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
            #5e-6)
    #])
def test_four_qubit_rotation_gates_on_simulator(
        gate, half_turns, initial_state, correct_state, atol):

    simulator = cirq.google.XmonSimulator()
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit.from_ops(gate(a, b, c, d)**half_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
        result.final_state, correct_state, atol=atol)


def test_pqrs_gate_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    d = cirq.NamedQubit('d')

    circuit = cirq.Circuit.from_ops(
        PQRS(a, b, c, d))
    assert circuit.to_text_diagram().strip() == """
a: ───P───
      │
b: ───Q───
      │
c: ───R───
      │
d: ───S───
""".strip()

    circuit = cirq.Circuit.from_ops(
        PQRS(a, b, c, d)**-0.5)
    assert circuit.to_text_diagram().strip() == """
a: ───P────────
      │
b: ───Q────────
      │
c: ───R────────
      │
d: ───S^-0.5───
""".strip()

    circuit = cirq.Circuit.from_ops(
        PQRS(a, c, b, d)**0.2)
    assert circuit.to_text_diagram().strip() == """
a: ───P───────
      │
b: ───R───────
      │
c: ───Q───────
      │
d: ───S^0.2───
""".strip()

    circuit = cirq.Circuit.from_ops(
        PQRS(d, b, a, c)**0.7)
    assert circuit.to_text_diagram().strip() == """
a: ───R───────
      │
b: ───Q───────
      │
c: ───S───────
      │
d: ───P^0.7───
""".strip()
