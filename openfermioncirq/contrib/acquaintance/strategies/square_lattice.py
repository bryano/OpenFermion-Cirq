# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides acquaintance strategies for circuits based on square lattices."""

from typing import Sequence, Tuple

from cirq import circuits, ops

from cirq.contrib.acquaintance.devices import UnconstrainedAcquaintanceDevice
from cirq.contrib.acquaintance.gates import SwapNetworkGate
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.bipartite import (
        BipartiteSwapNetworkGate, BipartiteGraphType)

def square_lattice_acquaintance_strategy(
        shape: Tuple[int, int],
        qubit_order: Sequence[ops.QubitId],
        qubits_per_site: int=1,
        swap_gate: ops.Gate=ops.SWAP,
        subgraph: BipartiteGraphType=BipartiteGraphType.COMPLETE,
        ) -> circuits.Circuit:
    """
    Returns an acquaintance strategy ... TODO

    Args:
        shape: The dimensions of the square lattice.
        qubit_order: The qubits on which the strategy should be defined.
        qubits_per_site: TODO
        swap_gate: TODO
        BipartiteGraphType: TODO

    Returns:
        TODO
    """
    width, height = shape
    n_sites = width * height
    n_qubits = n_sites * qubits_per_site
    if n_qubits != len(qubit_order):
        raise ValueError('width * height * qubits_per_site != len(qubits)')

    shift_gate = CircularShiftGate(qubits_per_site, swap_gate)
    bipartite_acquaintance_gate = BipartiteSwapNetworkGate(
            subgraph=subgraph, part_size=qubits_per_site, swap_gate=swap_gate)

    strategy = circuits.Circuit(device=UnconstrainedAcquaintanceDevice)
    for layer in range(1-width, width + (width % 2)):
        operations = []
        for x in range(abs(layer), n_sites - abs(layer) - (layer <= 0), 2):
            if width % 2:
                if width + layer + ((x // width) % 2) - 1 <= x % width < -layer:
                    continue
                if width - layer <= x % width < layer - ((x // width) % 2):
                    continue
            else:
                if any(width + sgn * layer <= x % width < -sgn * layer
                       for sgn in (-1, 1)):
                    continue
            qubits = qubit_order[qubits_per_site * x: qubits_per_site * (x + 2)]
            if (x + 1) % width:
                operations.append(shift_gate(*qubits))
            else:
                operations.append(bipartite_acquaintance_gate(*qubits))
        strategy._moments.append(circuits.Moment(operations))

    return strategy
