import numpy as np
import math
import pytest

# Import functions
from direct_stiffness_method.direct_stiffness_method import Structure, BoundaryConditions, Solver, PostProcessing, BucklingAnalysis, PlotResults

# We test against problems that I solved by hand!

# Define test data
nodes = {
    0: [0, 0.0, 0.0, 10.0],
    1: [1, 15.0, 0.0, 10.0],
    2: [2, 15.0, 0.0, 0.0]
}

elements = [
    [0, 1],
    [1, 2]
]

element_properties = {
    0: {"b": 0.5, "h": 1.0, "E": 1000, "nu": 0.3},
    1: {"b": 1.0, "h": 0.5, "E": 1000, "nu": 0.3}
}

# Expected results
expected_results = {
    0: {"A": 0.5, "Iy": 0.010416667, "Iz": 0.041666667, "J": 0.052083333, "l": 15},
    1: {"A": 0.5, "Iy": 0.041666667, "Iz": 0.010416667, "J": 0.052083333, "l": 10}
}

# Initialize the structure instance
structure = Structure(nodes, elements, element_properties)

@pytest.mark.parametrize("elem_id", [0, 1])
def test_compute_section_properties(elem_id):
    A, Iy, Iz, J = structure.compute_section_properties(elem_id)
    
    assert np.isclose(A, expected_results[elem_id]["A"], atol=1e-6), f"Failed for element {elem_id}, A"
    assert np.isclose(Iy, expected_results[elem_id]["Iy"], atol=1e-6), f"Failed for element {elem_id}, Iy"
    assert np.isclose(Iz, expected_results[elem_id]["Iz"], atol=1e-6), f"Failed for element {elem_id}, Iz"
    assert np.isclose(J, expected_results[elem_id]["J"], atol=1e-6), f"Failed for element {elem_id}, J"

@pytest.mark.parametrize("elem_id, nodes_pair", [(0, (0, 1)), (1, (1, 2))])
def test_element_length(elem_id, nodes_pair):
    length = structure.element_length(*nodes_pair)
    
    assert np.isclose(length, expected_results[elem_id]["l"], atol=1e-6), f"Failed for element {elem_id}, length"

