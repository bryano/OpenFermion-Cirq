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


def normal_ordered_interaction_operator(
        operator: openfermion.InteractionOperator
        ) -> openfermion.InteractionOperator:
    constant = operator.constant
    one_body_tensor = operator.one_body_tensor.copy()
    two_body_tensor = np.zeros_like(operator.two_body_tensor)
    for indices in itertools.product(*(
        range(d) for d in two_body_tensor.shape)):
        normal_indices = tuple(sorted(indices[:2]) + sorted(indices[2:]))
        two_body_tensor[normal_indices] += (
                operator.two_body_tensor[indices])
    return openfermion.InteractionOperator(
            constant, one_body_tensor, two_body_tensor)
