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
import random

import cirq
import cirq.contrib.acquaintance as cca
import numpy as np
import openfermion

import pytest

from openfermioncirq.primitives.general_swap_network import (
    trotter_unitary)
from openfermioncirq.variational.ansatzes.coupled_cluster import (
    CoupledClusterOperator,
    PairedCoupledClusterOperator,
    UnitaryCoupledClusterAnsatz)

from openfermioncirq.optimization import COBYLA, OptimizationParams
from openfermioncirq import VariationalStudy
from openfermioncirq import HamiltonianObjective

def test_cc_operator():
    pass


@pytest.mark.parametrize('n_spatial_modes',
    range(2, 7))
def test_paired_cc_operator(n_spatial_modes):
    for kwargs in (dict(zip(['include_real_part', 'include_imag_part'], flags))
            for flags in itertools.product((True, False), repeat=2)):
        operator = PairedCoupledClusterOperator(n_spatial_modes, **kwargs)
        params = list(operator.params())
        assert len(set(params)) == len(params)
        assert len(params) == (
                n_spatial_modes * (n_spatial_modes - 1) * sum(kwargs.values()))
        T = operator.operator()
        H = T - openfermion.hermitian_conjugated(T)
        resolver = {p: random.uniform(-5, 5) for p in operator.params()}
        H = cirq.resolve_parameters(H, resolver)
        exponent = 1j * H
        assert (not exponent) or openfermion.is_hermitian(exponent)


def print_hermitian_matrices(*matrices):
    matrices = tuple(matrices)
    assert len(set(m.shape for m in matrices)) == 1
    N = len(matrices[0])
    n = int(np.log2(N))
    assert N == 1 << n
    for i, j in itertools.combinations_with_replacement(range(N), 2):
        vs = tuple(matrix[i, j] for matrix in matrices)
        if any(vs):
            line = '{0:0{n}b} {1:0{n}b} '.format(i, j, n=n)
            line += ' '.join('{}'.format(v) for v in vs)
            print(line)


@pytest.mark.parametrize('cluster_operator,parameters,n_repetitions',
    [(cluster_operator, np.random.uniform(-5, 5, shape), n_repetitions)
        for n_spatial_modes in [2, 3, 4]
        for cluster_operator in [PairedCoupledClusterOperator(n_spatial_modes,
            include_real_part=False)]
#       for n_repetitions in (None, 1, 3)
        for n_repetitions in (3,)
        for shape in [
            (n_repetitions or 1, len(list(cluster_operator.params())))]
        for _ in range(3)
        ])
def test_paired_ucc(cluster_operator, parameters, n_repetitions):
    ansatz = UnitaryCoupledClusterAnsatz(
            cluster_operator, n_repetitions=n_repetitions)
    resolver = dict(zip(ansatz.params(), parameters.flatten()))

    swap_network = cluster_operator.swap_network()

    circuit = cirq.resolve_parameters(ansatz._circuit, resolver)
    actual_unitary = circuit.to_unitary_matrix(
            qubit_order=swap_network.qubit_order)

    acquaintance_dag = cca.get_acquaintance_dag(
            swap_network.circuit, swap_network.initial_mapping)
    partial_unitaries = []
    for repetition in range(len(parameters)):
        subparameters = parameters[repetition]
        subresolver = dict(zip(cluster_operator.params(), subparameters))
        operator = cluster_operator.operator()
        resolved_operator = cirq.resolve_parameters(operator, subresolver)
        hamiltonian = -1j * (resolved_operator -
                openfermion.hermitian_conjugated(resolved_operator))
        partial_unitary = trotter_unitary(acquaintance_dag, hamiltonian)
        partial_unitaries.append(partial_unitary)
    if len(partial_unitaries) > 1:
        expected_unitary = np.linalg.multi_dot(partial_unitaries[::-1])
    else:
        expected_unitary = partial_unitaries[0]

    assert np.allclose(actual_unitary, expected_unitary)


def test_integration_paired_ucc():
    diatomic_bond_length = .7414
    geometry = [('H', (0., 0., 0.)), 
                ('H', (0., 0., diatomic_bond_length))]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    description = format(diatomic_bond_length)

    molecule = openfermion.MolecularData(
        geometry,
        basis,
        multiplicity,
        description=description)
    molecule.load()

    hamiltonian = molecule.get_molecular_hamiltonian()

    n_spatial_modes = 2
    cluster_operator = PairedCoupledClusterOperator(
            n_spatial_modes, include_real_part=False)
    ansatz = UnitaryCoupledClusterAnsatz(cluster_operator, n_repetitions=3)

    objective = HamiltonianObjective(hamiltonian)

    q0, q1, _, _ = ansatz.qubits
    preparation_circuit = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.X(q1))
    study = VariationalStudy(
        name='my_hydrogen_study',
        ansatz=ansatz,
        objective=objective,
        preparation_circuit=preparation_circuit)

    initial_guess = [0.01 for _ in ansatz.params()]
    optimization_params = OptimizationParams(
        algorithm=COBYLA,
        initial_guess=initial_guess)
    result = study.optimize(optimization_params)

    # Pretty loose testing conditions here but it is a start.
    assert result.optimal_value <= 0.9 * molecule.hf_energy
    assert result.optimal_value >= molecule.fci_energy
