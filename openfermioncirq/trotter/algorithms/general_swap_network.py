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
import itertools
from typing import cast, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import cirq
import cirq.contrib.acquaintance as cca
import numpy as np
from openfermion import InteractionOperator

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Dict


from openfermioncirq.trotter.trotter_algorithm import (
        Hamiltonian,
        TrotterStep,
        TrotterAlgorithm)
from openfermioncirq.gates import (
        CombinedSwapAndZ, CombinedCXXYYPowGate, CombinedDoubleExcitationGate)


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


def trotterize(hamiltonian: InteractionOperator):
    """

    Returns gates such that $e^{i H} ~ \prod_a e^{i H_a}$.

    """
    n_qubits = hamiltonian.n_qubits
    one_body_tensor = hamiltonian.one_body_tensor
    two_body_tensor = hamiltonian.two_body_tensor
    assert np.allclose(one_body_tensor, np.conj(one_body_tensor))
    assert np.allclose(one_body_tensor, one_body_tensor.T)
    assert np.allclose(two_body_tensor.reshape((n_qubits ** 2,) * 2),
            two_body_tensor.reshape((n_qubits ** 2,) * 2).T)

    gates = {} # type: Dict[Tuple[int, ...], cirq.Gate]
    for p in range(n_qubits):
        coeff = one_body_tensor[p, p]
        if coeff:
            gates[(p,)] = cirq.Z**(-coeff / np.pi)
    for p, q in itertools.combinations(range(n_qubits), 2):
        tunneling_coeff = one_body_tensor[p, q]
        interaction_coeff = (
                two_body_tensor[p, q, p, q] +
                two_body_tensor[p, q, q, p] +
                two_body_tensor[q, p, q, p])
        weights = (-tunneling_coeff, -interaction_coeff
                ) # type: Tuple[float, ...]
        if any(weights):
            gates[(p, q)] = CombinedSwapAndZ(
                    cast(Tuple[float, float], weights))
    for i, j, k in itertools.combinations(range(n_qubits), 3):
        weights = tuple(
            two_body_tensor[p, r, q, r] +
            two_body_tensor[r, p, q, r] +
            two_body_tensor[p, r, r, q] +
            two_body_tensor[r, p, r, q]
            for p, q, r in [(i, j, k), (k, i, j), (j, k, i)])
        if any(weights):
            gates[(i, j, k)] = CombinedCXXYYPowGate(
                    cast(Tuple[float, float, float], weights))
    for i, j, k, l  in itertools.combinations(range(n_qubits), 4):
        weights = tuple(
            two_body_tensor[p, q, r, s] +
            two_body_tensor[p, q, s, r] +
            two_body_tensor[q, p, r, s] +
            two_body_tensor[q, p, s, r]
            for p, q, r, s in [(i, j, k, l), (i, k, j, l), (i, l, j, k)])
        if any(weights):
            gates[(i, j, k, l)] = CombinedDoubleExcitationGate(
                    cast(Tuple[float, float, float], weights))
    return gates


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
