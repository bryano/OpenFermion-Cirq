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

import numpy as np

import openfermion
from openfermion.utils._testing_utils import random_interaction_operator

def random_symmetric_matrix(n: int):
    m = np.random.standard_normal((n, n))
    return (m + m.T) / 2.

def random_interaction_operator_term(
        order: int,
        real: bool = True
        ) -> openfermion.InteractionOperator:

    n_orbitals = order

    operator = random_interaction_operator(n_orbitals, real=real)
    operator.constant = 0

    for indices in itertools.product(range(n_orbitals), repeat=2):
        if len(set(indices)) != order:
            operator.one_body_tensor[indices] = 0

    for indices in itertools.product(range(n_orbitals), repeat=4):
        if len(set(indices)) != order:
            operator.two_body_tensor[indices] = 0

    return operator