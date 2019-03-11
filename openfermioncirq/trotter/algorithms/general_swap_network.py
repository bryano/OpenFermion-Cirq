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

"""A Trotter algorithm using the "fermionic simulation gate"."""

import enum
from typing import Optional, Sequence, Tuple, Union

import cirq
import cirq.contrib.acquaintance as cca
from openfermion import InteractionOperator


from openfermioncirq.trotter.trotter_algorithm import (
        Hamiltonian,
        TrotterStep,
        TrotterAlgorithm)


@enum.unique
class SwapNetworkTrotterStrategy(enum.Enum):
    FIRST = 1
    RANDOM = 2
    ALL = 3

    def __repr__(self):
        return '.'.join(
                ('openfermioncirq', 'trotter', 'algorithms',
                 'general_swap_network', 'SwapNetworkTrotterStrategy',
                 self.name))



class GeneralSwapNetworkTrotterAlgorithm(TrotterAlgorithm):
    """A Trotter algorithm using the "fermionic simulation gate".

    This algorithm simulates an InteractionOperator. It uses layers of
    fermionic swap networks to simulate the one- and two-body
    interactions.
    """

    def __init__(self,
            initial_order: Sequence[cirq.QubitId],
            swap_network: cirq.Circuit,
            strategy: Union[str, SwapNetworkTrotterStrategy] =
                SwapNetworkTrotterStrategy.FIRST
            ) -> None:
        if not cca.is_acquaintance_strategy(swap_network):
            raise TypeError('not is_acquaintance_strategy(swap_network)')
        self.swap_network = swap_network

        if not (set(initial_order) < set(swap_network.all_qubits())):
            raise ValueError(
                    'not all qubits in initial order are in the swap network.')
        self.initial_order = tuple(initial_order)

        self.strategy = (
                strategy if isinstance(strategy, SwapNetworkTrotterStrategy)
                else SwapNetworkTrotterStrategy[strategy])
        if self.strategy != SwapNetworkTrotterStrategy.FIRST:
            raise NotImplementedError()

    supported_types = {InteractionOperator}

    def asymmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        return AsymmetricGeneralSwapNetworkTrotterStep(hamiltonian,
                self.initial_order, self.swap_network, self.strategy)


class GeneralSwapNetworkTrotterStep(TrotterStep):

    def __init__(self,
            hamiltonian: InteractionOperator,
            initial_order: Tuple[cirq.QubitId, ...],
            swap_network: cirq.Circuit,
            strategy: SwapNetworkTrotterStrategy
            ) -> None:

        self.hamiltonian = hamiltonian
        self.initial_order = initial_order
        self.swap_network = swap_network
        self.strategy = strategy


class AsymmetricGeneralSwapNetworkTrotterStep(GeneralSwapNetworkTrotterStep):

    def trotter_step(
            self,
            qubits: Sequence[cirq.QubitId],
            time: float,
            control_qubit: Optional[cirq.QubitId]=None
            ) -> cirq.OP_TREE:
        raise NotImplementedError()
