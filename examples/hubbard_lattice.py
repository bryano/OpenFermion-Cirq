import itertools
import math
from typing import Optional

import cirq
import cirq.contrib.acquaintance as cca
import openfermion
import openfermioncirq as ofc


FSWAP = cca.SwapPermutationGate(swap_gate=ofc.FSWAP)
device = cca.UnconstrainedAcquaintanceDevice


def get_spin_orbitals(lattice):
    spin_orbitals = [lattice.to_spin_orbital_index(lattice.to_site_index((x, y)), 0, s)
            for y in range(lattice.y_dimension)
            for x in range(lattice.x_dimension)[::(-1)**y]
            for s in (0, 1)[::(-1)**(x + y)]]
    for i in range(len(spin_orbitals) - 2):
        j1, _, s1 = lattice.from_spin_orbital_index(spin_orbitals[i])
        j2, _, s2 = lattice.from_spin_orbital_index(spin_orbitals[i + 1])
        if i % 2:
            assert lattice.manhattan_distance(j1, j2, True) == 1
            assert s1 == s2
        else:
            assert j1 == j2
            assert s1 != s2
    return spin_orbitals


def get_edge_sets(lattice):
    N = lattice.n_spin_orbitals
    width = lattice.x_dimension
    edge_sets = {}
    edge_sets['even'] = [(i, i + 1) for i in range(0, N - 1, 2)]
    edge_sets['odd'] = [(i, i + 1) for i in range(1, N - 1, 2)]
    edge_sets['interface'] = [(i - 1, i) for i in range(2 * width, N, 2 * width)]
    assert len(edge_sets['even']) == len(edge_sets['odd']) + 1 == N // 2
    assert len(edge_sets['interface']) == lattice.y_dimension - 1
    return edge_sets


def get_tunneling_pairs(lattice, edge_type='neighbor'):
    tunneling_pairs = set(frozenset(lattice.to_spin_orbital_index(i, 0, s) for i in edge)
            for edge in lattice.site_pairs_iter(edge_type, False)
            for s in lattice.spin_indices)
    for i1, i2 in tunneling_pairs:
        j1, _, s1 = lattice.from_spin_orbital_index(i1)
        j2, _, s2 = lattice.from_spin_orbital_index(i2)
        assert s1 == s2
        assert lattice.manhattan_distance(j1, j2, True) == 1
    return tunneling_pairs


def test_swap_network(
        width: int,
        height: Optional[int] = None,
        fixed: bool = False) -> int:
    if height is None:
        height = width

    N = 2 * width * height
    lattice = openfermion.HubbardSquareLattice(width, height, periodic=False)
    assert lattice.n_spin_orbitals == N

    qubits = cirq.LineQubit.range(N)
    spin_orbitals = get_spin_orbitals(lattice)
    assert len(spin_orbitals) == N

    canonical_mapping = dict(zip(qubits, spin_orbitals))
    alt_spin_orbitals = [spin_orbitals[i + di]
            for i in range(0, N, 2) for di in (1, 0)]
    alt_canonical_mapping = dict(zip(qubits, alt_spin_orbitals))

    edge_sets = get_edge_sets(lattice)

    U_L = cirq.Moment(FSWAP(qubits[i], qubits[j]) for i, j in edge_sets['even'])
    U_R = cirq.Moment(FSWAP(qubits[i], qubits[j]) for i, j in edge_sets['odd'])

    acquaintance_layers = {
        edges_type: cirq.Moment(cca.acquaint(qubits[i], qubits[j]) for i, j in edges)
        for edges_type, edges in edge_sets.items()}

    tunneling_pairs = get_tunneling_pairs(lattice, 'neighbor')
    assert len(tunneling_pairs) == 2 * (height * (width - 1) + width * (height - 1))


    # horizontal

    horizontal_tunneling_pairs = get_tunneling_pairs(lattice, 'horizontal_neighbor')
    horizontal_network = [
            acquaintance_layers['odd'], U_L, acquaintance_layers['odd']]
    mapping = dict(canonical_mapping)
    circuit = cirq.Circuit.from_ops(horizontal_network, device=device)
    opps = cca.get_logical_acquaintance_opportunities(circuit, mapping)
    assert horizontal_tunneling_pairs <= opps
    cca.update_mapping(mapping, circuit.all_operations())
    assert mapping == alt_canonical_mapping


    # vertical

    if fixed:
        n_reps = width
    else:
        n_reps = math.ceil((N / 8) ** 0.5) - 1
    first_vertical_network = (
            [U_L, U_R, acquaintance_layers['interface']] * n_reps +
            [U_R, U_L] * n_reps)

    mapping = dict(alt_canonical_mapping)
    circuit = cirq.Circuit.from_ops(first_vertical_network, device=device)
    opps = cca.get_logical_acquaintance_opportunities(circuit, mapping)
    cca.update_mapping(mapping, circuit.all_operations())
    assert mapping == alt_canonical_mapping

    second_vertical_network = (
            [U_L] +
            [acquaintance_layers['interface'], U_L, U_R] * n_reps +
            [acquaintance_layers['interface']])
    circuit = cirq.Circuit.from_ops(second_vertical_network, device=device)
    opps = cca.get_logical_acquaintance_opportunities(circuit, mapping)

    vertical_network = first_vertical_network + second_vertical_network
    circuit = cirq.Circuit.from_ops(vertical_network, device=device)
    opps = cca.get_logical_acquaintance_opportunities(circuit, mapping)


    initial_mapping = dict(canonical_mapping)
    moments = horizontal_network + first_vertical_network + second_vertical_network
    circuit = cirq.Circuit.from_ops(moments, device=cca.UnconstrainedAcquaintanceDevice)
    acquaintance_opps = cca.get_logical_acquaintance_opportunities(circuit, initial_mapping)

    diff = tunneling_pairs - acquaintance_opps
    return diff


if __name__ == '__main__':
    print('shape: # missing')
    for shape in itertools.product(range(3, 7), repeat=2):
        ns_missing = [len(test_swap_network(*shape, fixed)) for fixed in (False, True)]
        assert not ns_missing[1]
        print(f'{shape}: {ns_missing[0]}')
