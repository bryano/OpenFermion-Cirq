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

import abc
import collections
import itertools
from typing import Dict, Optional, Tuple, Union

import cirq
import cirq.contrib.acquaintance as cca
import numpy
import openfermion
import sympy

from openfermioncirq.gates import FSWAP
from openfermioncirq.primitives.general_swap_network import (
        FermionicSwapNetwork, trotterize, GreedyExecutionStrategy)
from openfermioncirq.variational.ansatz import VariationalAnsatz
from openfermioncirq.variational.letter_with_subscripts import (
        LetterWithSubscripts)


OneBodyIndex = Tuple[int, int]
TwoBodyIndex = Tuple[int, int, int, int]
Coefficient = Union[int, float, LetterWithSubscripts]


class SymbolicTensor(collections.defaultdict):
    def __init__(self, shape, *args, **kwargs):
        self.shape = shape
        super().__init__(*args, **kwargs)

    def _resolve_parameters_(self, resolver):
        tensor = numpy.zeros(self.shape, dtype = numpy.complex128)
        for key, val in self.items():
            tensor[key] = cirq.resolve_parameters(val, resolver)
        return tensor

    def _is_parameterized_(self):
        return any(cirq.is_parameterized(val) for val in self.values())

    def __bool__(self):
        return any(self.values())

    def __len__(self):
        return self.shape[0]

    def copy(self):
        return SymbolicTensor(tuple(self.shape),
                self.default_factory, self.items())

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __add__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError('self.shape != other.shape')
        keys = set(self.keys) | set(other.keys)
        items = ((key, self[key] + other[key]) for key in  keys)
        return type(self)(self.shape, self.default_factory, items)

    def __sub__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError('self.shape != other.shape')
        keys = set(self.keys()) | set(other.keys())
        items = ((key, self[key] - other[key]) for key in  keys)
        return type(self)(self.shape, self.default_factory, items)

    def __imul__(self, other):
        if not isinstance(other, (int, float, complex, sympy.Basic)):
            return NotImplemented
        for key in self:
            self[key] *= other
        return self


class SymbolicInteractionOperator(openfermion.InteractionOperator):
    def __init__(self,
            constant: Coefficient,
            one_body_tensor: SymbolicTensor,
            two_body_tensor: SymbolicTensor):
        self.n_qubits = len(one_body_tensor)
        self.n_body_tensors = {
                (): constant,
                (1, 0): one_body_tensor,
                (1, 1, 0, 0): two_body_tensor}

    def _is_parameterized_(self):
        return any(cirq.is_parameterized(tensor) for tensor in
                self.n_body_tensors.values())

    def _resolve_parameters_(self, resolver):
        constant = cirq.resolve_parameters(self.constant, resolver)
        one_body_tensor = cirq.resolve_parameters(
                self.one_body_tensor, resolver)
        two_body_tensor = cirq.resolve_parameters(
                self.two_body_tensor, resolver)
        return openfermion.InteractionOperator(
                constant, one_body_tensor, two_body_tensor)

    def __bool__(self):
        return any(bool(tensor) for tensor in self.n_body_tensors.values())



class CoupledClusterOperatorType(metaclass=abc.ABCMeta):
    """A Coupled Cluster operator.
    """

    @abc.abstractproperty
    def n_spatial_modes(self):
        pass

    @property
    def n_spin_modes(self):
        return 2 * self.n_spatial_modes

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def swap_network(self) -> FermionicSwapNetwork:
        pass

    @abc.abstractmethod
    def operator(self) -> SymbolicInteractionOperator:
        pass


updown_indices = (openfermion.up_index, openfermion.down_index)


class CoupledClusterOperator(CoupledClusterOperatorType):
    """A Coupled Cluster operator with singles, doubles, etc.
    """

    def __init__(self,
            n_spatial_modes: int,
            n_occupied_spatial_modes: int,
            order: int = 2) -> None:
        self._n_spatial_modes = n_spatial_modes
        self.n_occupied_spatial_modes = n_occupied_spatial_modes
        self.order = order

    @property
    def n_spatial_modes(self):
        return self._n_spatial_modes

    @property
    def occupied_spatial_modes(self):
        return range(self.n_occupied_spatial_modes)

    @property
    def virtual_spatial_modes(self):
        return range(self.n_occupied_spatial_modes, self.n_spatial_modes)

    def params(self):
        for k in range(1, self.order + 1):
            occupied_index_sets = (I for I in
                    itertools.combinations(self.occupied_orbitals, k)
                    if max(collections.Counter(I).values()) <= 2)
            virtual_index_sets = (I for I in
                    itertools.combinations(self.virtual_orbitals, k)
                    if max(collections.Counter(I).values()) <= 2)
            for I, J in itertools.product(
                    virtual_index_sets, occupied_index_sets):
                yield LetterWithSubscripts('t', *(I + J))

    def operator(self):
        raise NotImplementedError()


class PairedCoupledClusterOperator(CoupledClusterOperatorType):
    def __init__(self,
            n_spatial_modes: int,
            include_real_part: bool = True,
            include_imag_part: bool = True
            ) -> None:
        self._n_spatial_modes = n_spatial_modes
        complex_parts = {} # type: Dict[str, complex]
        if include_real_part:
            complex_parts['re'] = 1
        if include_imag_part:
            complex_parts['im'] = 1j
        self.complex_parts = complex_parts

    @property
    def n_spatial_modes(self):
        return self._n_spatial_modes

    def params(self):
        for i, j in itertools.combinations(range(self.n_spatial_modes), 2):
            for part in self.complex_parts:
                yield LetterWithSubscripts('t', part, i, j)
                yield LetterWithSubscripts('t', part, i, i, j, j)

    def operator(self):
        constant = 0
        one_body_tensor = SymbolicTensor((self.n_spin_modes,) * 2, int)
        two_body_tensor = SymbolicTensor((self.n_spin_modes,) * 4, int)

        for i, j in itertools.combinations(range(self.n_spatial_modes), 2):
            param = sum(v * LetterWithSubscripts('t', part, i, j)
                    for part, v in self.complex_parts.items())
            for spin in updown_indices:
                one_body_tensor[spin(i), spin(j)] += param
            param = sum(v * LetterWithSubscripts('t', part, i, i, j, j)
                    for part, v in self.complex_parts.items())
            indices = tuple(spin(k) for k in (i, j) for spin in updown_indices)
            two_body_tensor[indices] += param

        return SymbolicInteractionOperator(
               constant, one_body_tensor, two_body_tensor)

    def swap_network(self) -> FermionicSwapNetwork:
        n_qubits = 2 * self.n_spatial_modes

        qubits = cirq.LineQubit.range(n_qubits)
        qubit_pairs = [qubits[2 * i: 2 * (i + 1)]
                for i in range(self.n_spatial_modes)]
        circuit, qubit_order = cca.quartic_paired_acquaintance_strategy(
                qubit_pairs, FSWAP)
        qubit_mapping = dict(zip(qubit_order, qubits))
        circuit = circuit.with_device(circuit.device, qubit_mapping.get)
        initial_permutation = {l.x: i for i, l in enumerate(qubit_order)}
        initial_permutation_gate = cca.LinearPermutationGate(
                n_qubits, initial_permutation, swap_gate = FSWAP)
        circuit.insert(0, initial_permutation_gate(*qubits))
        initial_mapping = {q: i for i, q in enumerate(qubits)}

        cca.return_to_initial_mapping(circuit, FSWAP)

        return FermionicSwapNetwork(circuit, initial_mapping, qubits)


class UnitaryCoupledClusterAnsatz(VariationalAnsatz):
    """A Unitary Coupled Cluster ansatz.

    TODO
    """

    def __init__(self,
            cluster_operator: CoupledClusterOperator,
            execution_strategy: cca.executor.ExecutionStrategy =
                GreedyExecutionStrategy,
            n_repetitions: Optional[int] = None,
            **kwargs) -> None:

        self.cluster_operator = cluster_operator
        operator = self.cluster_operator.operator()
        exponent = -1j * (
                operator - openfermion.hermitian_conjugated(operator))
        gates = trotterize(exponent)
        swap_network = cluster_operator.swap_network()
        execution_strategy(gates, swap_network.initial_mapping)(
                swap_network.circuit)

        self.n_repetitions = n_repetitions
        if n_repetitions is None:
            circuit = swap_network.circuit
        else:
            circuit = cirq.Circuit()
            for repetition in range(n_repetitions):
                resolver = {}
                for param in cluster_operator.params():
                    assert isinstance(param, LetterWithSubscripts)
                    subscripts = param.subscripts + (repetition,)
                    new_param = LetterWithSubscripts(param.letter, *subscripts)
                    resolver[param] = new_param
                subcircuit = cirq.resolve_parameters(
                        swap_network.circuit, resolver)
                circuit += subcircuit
        self._circuit = circuit

        super().__init__(**kwargs)

    def params(self):
        if self.n_repetitions is None:
            for param in self.cluster_operator.params():
                yield param
        else:
            for repetition in range(self.n_repetitions):
                for param in self.cluster_operator.params():
                    assert isinstance(param, LetterWithSubscripts)
                    subscripts = param.subscripts + (repetition,)
                    yield LetterWithSubscripts(param.letter, *subscripts)

    def default_initial_params(self):
        return numpy.zeros(len(list(self.params())))

    def operations(self, qubits):
        func = lambda q: qubits[q.x]
        for op in self._circuit.all_operations():
            yield op.transform_qubits(func)

    def _generate_qubits(self):
        return cirq.LineQubit.range(self.cluster_operator.n_spin_modes)
