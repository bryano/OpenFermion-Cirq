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
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import cirq
import cirq.contrib.acquaintance as cca
import openfermion

from openfermioncirq.primitives.general_swap_network import trotterize
from openfermioncirq.variational.ansatz import VariationalAnsatz
from openfermioncirq.variational.letter_with_subscripts import (
        LetterWithSubscripts)

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import DefaultDict

OneBodyIndex = Tuple[int, int]
TwoBodyIndex = Tuple[int, int, int, int]
Coefficient = Union[int, LetterWithSubscripts]

SymbolicInteractionOperator = collections.namedtuple(
        'SymbolicInteractionOperator',
        ['n_qubits', 'constant', 'one_body_tensor', 'two_body_tensor'])


class CoupledClusterOperatorType(metaclass=abc.ABCMeta):
    """A Coupled Cluster operator.

    TODO
    """

    @abc.abstractproperty
    def n_spin_modes(self):
        pass

    @abc.abstractmethod
    def indices_iter(self, order: Optional[int] = None):
        pass

    @abc.abstractmethod
    def order_iter(self):
        pass

    def params(self):
        spatial_index_sets = set(tuple(i // 2 for i in indices)
                for indices in self.indices_iter())
        for indices in spatial_index_sets:
            yield LetterWithSubscripts('t', *indices)

    def operator(self) -> SymbolicInteractionOperator:
        if max(self.order_iter()) > 2:
            raise NotImplementedError()
        constant = 0
        one_body_tensor = collections.defaultdict(int
                ) # type: DefaultDict[OneBodyIndex, Coefficient]
        for indices in self.indices_iter(1):
            one_body_tensor[indices] += LetterWithSubscripts('t', *indices)
        two_body_tensor = collections.defaultdict(int
                ) # type: DefaultDict[TwoBodyIndex, Coefficient]
        for indices in self.indices_iter(2):
            two_body_tensor[indices] += LetterWithSubscripts('t', *indices)
        return SymbolicInteractionOperator(
                self.n_spin_modes, constant, one_body_tensor, two_body_tensor)

updown_indices = (openfermion.up_index, openfermion.down_index)


class CoupledClusterOperator(CoupledClusterOperatorType):
    """A Coupled Cluster operator with singles, doubles, etc.

    TODO
    """

    def __init__(self,
            n_spatial_modes: int,
            n_occupied_spatial_modes: int,
            order: int = 2) -> None:
        self.n_spatial_modes = n_spatial_modes
        self.n_occupied_spatial_modes = n_occupied_spatial_modes
        self.order = order

    @property
    def spatial_modes(self):
        return tuple(range(self.n_spatial_modes))

    @property
    def occupied_spatial_modes(self):
        return tuple(range(self.n_occupied_spatial_modes))

    @property
    def virtual_spatial_modes(self):
        return tuple(range(self.n_occupied_spatial_modes, self.n_spatial_modes))

    @property
    def spin_modes(self):
        return tuple(spin(i) for i, spin in
                itertools.product(self.spatial_modes, updown_indices))

    @property
    def occupied_spin_modes(self):
        return tuple(spin(i) for i, spin in
                itertools.product(self.occupied_spatial_modes, updown_indices))

    @property
    def virtual_spin_modes(self):
        return tuple(spin(i) for i, spin in
                itertools.product(self.virtual_spatial_modes, updown_indices))

    def order_iter(self):
        return range(1, self.order + 1)

    def indices_iter(self, order):
        if order is None:
            for order in self.order_iter():
                yield self.indices_iter(order)
        occupied_and_virtual_spin_modes = (
                itertools.combinations(spin_modes, order) for spin_modes
                in (self.occupied_spin_modes, self.virtual_spin_modes))
        for I, J in itertools.product(*occupied_and_virtual_spin_modes):
            if sum(i % 2 for i in I) == sum(j % 2 for j in J):
                yield I + J

    @property
    def n_spin_modes(self):
        return 2 * self.n_spatial_modes


class GeneralizedCoupledClusterOperator(CoupledClusterOperatorType):
    def __init__(self,
            n_spatial_modes: int,
            order: int = 2) -> None:
        self.n_spatial_modes = n_spatial_modes
        self.order = order

    def order_iter(self):
        return range(1, self.order + 1)

    @property
    def spatial_modes(self):
        return tuple(range(self.n_spatial_modes))

    @property
    def spin_modes(self):
        return tuple(spin(i) for i, spin in
                itertools.product(self.spatial_modes, updown_indices))

    def indices_iter(self, order):
        if order is None:
            for order in self.order_iter():
                yield self.indices_iter(order)
        spin_modes = itertools.combinations(self.spin_modes, order)
        for I, J in itertools.combinations(spin_modes, 2):
            if sum(i % 2 for i in I) == sum(j % 2 for j in J):
                yield I + J

    @property
    def n_spin_modes(self):
        return 2 * self.n_spatial_modes


class PairedCoupledClusterOperator(CoupledClusterOperatorType):
    def __init__(self, n_spatial_modes: int) -> None:
        self.n_spatial_modes = n_spatial_modes

    def order_iter(self):
        return (1, 2)

    @property
    def spatial_modes(self):
        return range(self.n_spatial_modes)

    def indices_iter(self, order):
        if order is None:
            for order in self.order_iter():
                yield self.indices_iter(order)
        if order == 1:
            for i in self.spatial_modes:
                for spin in updown_indices:
                    yield (spin(i),) * 2
        elif order == 2:
            for ij in itertools.combinations(self.spatial_modes, 2):
                yield tuple(spin(i) for i in ij for spin in updown_indices)

    @property
    def n_spin_modes(self):
        return 2 * self.n_spatial_modes


class UnitaryCoupledClusterAnsatz(VariationalAnsatz):
    """A Unitary Coupled Cluster ansatz.

    TODO
    """

    def __init__(self,
            cluster_operator: CoupledClusterOperator,
            swap_network: cirq.Circuit,
            initial_mapping: Dict[cirq.LineQubit, int],
            execution_strategy: cca.executor.ExecutionStrategy =
                cca.GreedyExecutionStrategy,
            **kwargs) -> None:

        self.cluster_operator = cluster_operator
        gates = trotterize(self.cluster_operator.operator())
        circuit = swap_network.copy()
        execution_strategy(gates, initial_mapping)(circuit)
        self._circuit = circuit

        super().__init__(**kwargs)

    def params(self):
        return self.cluster_operator.params()

    def operations(self, qubits):
        func = lambda q: qubits[q.x]
        for op in self._circuit.all_operations():
            yield op.transform_qubits(func)

    def _generate_qubits(self):
        return cirq.LineQubit.range(self.cluster_operator.n_spin_modes)
