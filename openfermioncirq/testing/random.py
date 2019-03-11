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


def random_symmetric_matrix(n):
    m = np.random.standard_normal((n, n))
    return (m + m.T) / 2.


def random_interaction_operator(n_modes):
    constant = 0
    one_body_tensor = random_symmetric_matrix(n_modes)
    one_body_tensor = np.zeros((n_modes,) * 2)
    two_body_tensor = (
            random_symmetric_matrix(n_modes ** 2).reshape((n_modes,) * 4))
    two_body_tensor = np.zeros((n_modes,) * 4)
    for p, q in itertools.combinations(range(n_modes), 2):
        two_body_tensor[p, q, p, q] = 1
    return openfermion.InteractionOperator(
            constant, one_body_tensor, two_body_tensor)

