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

import collections
import random

import cirq
import cirq.contrib.acquaintance as cca
import numpy as np
import openfermion

import pytest

import openfermioncirq as ofc
from openfermioncirq.primitives.general_swap_network import (
    trotter_unitary)
from openfermioncirq.variational.letter_with_subscripts import LetterWithSubscripts
from openfermioncirq.variational.ansatzes.coupled_cluster import (
    CoupledClusterOperator, GeneralizedCoupledClusterOperator,
    PairedCoupledClusterOperator,
    UnitaryCoupledClusterAnsatz)

def test_cc_operator():
    pass
#   operator = CoupledClusterOperator(7, 3)
#   operator.operator()

def test_generalized_cc_operator():
    pass
#   operator = GeneralizedCoupledClusterOperator(5)
#   operator.operator()


@pytest.mark.parametrize('n_spatial_modes',
    range(2, 7))
def test_paired_cc_operator(n_spatial_modes):
    operator = PairedCoupledClusterOperator(n_spatial_modes)
    params = list(operator.params())
    assert len(set(params)) == len(params)
    assert len(params) == n_spatial_modes * (n_spatial_modes - 1)
    T = operator.operator()
    H = T - openfermion.hermitian_conjugated(T)
    resolver = {p: random.uniform(-5, 5) for p in operator.params()}
    H = cirq.resolve_parameters(H, resolver)
    exponent = 1j * H
    assert openfermion.is_hermitian(exponent)


def test_paired_ucc():
    n_spatial_modes = 2
    qubits = cirq.LineQubit.range(2 * n_spatial_modes)
    qubit_pairs = [qubits[2 * i: 2 * (i + 1)] for i in range(n_spatial_modes)]
    swap_network, qubit_order = cca.quartic_paired_acquaintance_strategy(
            qubit_pairs, ofc.FSWAP)
    initial_mapping = {p: l.x for p, l in zip(qubits, qubit_order)}

    cluster_operator = PairedCoupledClusterOperator(n_spatial_modes)
    ansatz = UnitaryCoupledClusterAnsatz(
            cluster_operator, swap_network, initial_mapping)
    resolver = {p: 1j * random.uniform(-5, 5) for p in ansatz.params()}
    resolver = {p: 0 for p in ansatz.params()}
    resolver[LetterWithSubscripts('t', 0, 1)] = 1j * np.pi

    circuit = cirq.resolve_parameters(ansatz._circuit, resolver)
    actual_unitary = circuit.to_unitary_matrix(qubit_order=qubits)

    acquaintance_dag = cca.get_acquaintance_dag(swap_network, initial_mapping)
    operator = cluster_operator.operator()
    resolved_operator = cirq.resolve_parameters(operator, resolver)
    hamiltonian = -1j * (resolved_operator - openfermion.hermitian_conjugated(resolved_operator))
    expected_unitary = trotter_unitary(acquaintance_dag, hamiltonian)

    assert np.allclose(actual_unitary, expected_unitary)
