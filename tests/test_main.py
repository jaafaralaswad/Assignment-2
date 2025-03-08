import numpy as np
import math
import pytest

# Import functions
from direct_stiffness_method.direct_stiffness_method import Structure, StiffnessMatrices, BoundaryConditions, Solver, BucklingAnalysis, PlotResults

# Testing Class Structure for rectangular and circular cross-sections

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

# Define test data for circular sections
nodes_circular = {
    0: [0, 0.0, 0.0, 0.0],
    1: [1, -5.0, 1.0, 10.0],
    2: [2, -1.0, 5.0, 13.0],
    3: [3, -3.0, 7.0, 11.0],
    4: [4, 6.0, 9.0, 5.0]
}

elements_circular = [
    [0, 1],
    [1, 2],
    [2, 3],
    [2, 4]
]

element_properties_circular = {
    0: {"r": 1.0, "E": 500, "nu": 0.3},
    1: {"r": 1.0, "E": 500, "nu": 0.3},
    2: {"r": 1.0, "E": 500, "nu": 0.3},
    3: {"r": 1.0, "E": 500, "nu": 0.3},
}

# Expected results for circular sections
expected_circular_results = {
    0: {"A": 3.141592654, "Iy": 0.785398163, "Iz": 0.785398163, "J": 1.570796327},
    1: {"A": 3.141592654, "Iy": 0.785398163, "Iz": 0.785398163, "J": 1.570796327},
    2: {"A": 3.141592654, "Iy": 0.785398163, "Iz": 0.785398163, "J": 1.570796327},
    3: {"A": 3.141592654, "Iy": 0.785398163, "Iz": 0.785398163, "J": 1.570796327},
}

# Initialize the structure instance
structure_circular = Structure(nodes_circular, elements_circular, element_properties_circular)

@pytest.mark.parametrize("elem_id", [0, 1, 2, 3])
def test_compute_section_properties_circular(elem_id):
    A, Iy, Iz, J = structure_circular.compute_section_properties(elem_id)
    
    assert np.isclose(A, expected_circular_results[elem_id]["A"], atol=1e-6), f"Failed for element {elem_id}, A"
    assert np.isclose(Iy, expected_circular_results[elem_id]["Iy"], atol=1e-6), f"Failed for element {elem_id}, Iy"
    assert np.isclose(Iz, expected_circular_results[elem_id]["Iz"], atol=1e-6), f"Failed for element {elem_id}, Iz"
    assert np.isclose(J, expected_circular_results[elem_id]["J"], atol=1e-6), f"Failed for element {elem_id}, J"

# Define minimal test data
nodes_test = {
    0: [0, 0.0, 0.0, 0.0]
}

elements_test = [
    [0, 0]
]

@pytest.mark.parametrize("invalid_elem_id, invalid_properties", [
    (0, {"E": 500, "nu": 0.3})  # Missing both r and (b, h)
])
def test_invalid_section_properties_else_case(invalid_elem_id, invalid_properties):
    """Test that compute_section_properties raises ValueError when neither (b, h) nor r is defined."""
    
    # Create a temporary structure with invalid properties
    invalid_element_properties = {invalid_elem_id: invalid_properties}
    structure_invalid = Structure(nodes_test, elements_test, invalid_element_properties)
    
    with pytest.raises(ValueError, match="Invalid element properties. Define either \\(b, h\\) or \\(r\\)."):
        structure_invalid.compute_section_properties(invalid_elem_id)

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

# Expected stiffness matrices extracted from images
expected_stiffness_0 = np.array([
    [33.333, 0, 0, 0, 0, 0, -33.333, 0, 0, 0, 0, 0],
    [0, 0.14815, 0, 0, 0, 1.1111, 0, -0.14815, 0, 0, 0, 1.1111],
    [0, 0, 0.037037, 0, -0.2778, 0, 0, 0, -0.037037, 0, -0.2778, 0],
    [0, 0, 0, 1.3355, 0, 0, 0, 0, 0, -1.3355, 0, 0],
    [0, 0, -0.27778, 0, 2.7778, 0, 0, 0, 0.27778, 0, 1.3889, 0],
    [0, 1.1111, 0, 0, 0, 11.111, 0, -1.1111, 0, 0, 0, 5.5556],
    [-33.333, 0, 0, 0, 0, 0, 33.333, 0, 0, 0, 0, 0],
    [0, -0.1481, 0, 0, 0, -1.1111, 0, 0.14815, 0, 0, 0, -1.1111],
    [0, 0, -0.037037, 0, 0.2778, 0, 0, 0, 0.037037, 0, 0.2778, 0],
    [0, 0, 0, -1.3355, 0, 0, 0, 0, 0, 1.3355, 0, 0],
    [0, 0, -0.27778, 0, 1.3889, 0, 0, 0, 0.27778, 0, 2.7778, 0],
    [0, 1.1111, 0, 0, 0, 5.5556, 0, -1.1111, 0, 0, 0, 11.111]
])

expected_stiffness_1 = np.array([
    [50, 0, 0, 0, 0, 0, -50, 0, 0, 0, 0, 0],
    [0, 0.125, 0, 0, 0, 0.625, 0, -0.125, 0, 0, 0, 0.625],
    [0, 0, 0.5, 0, -2.5, 0, 0, 0, -0.5, 0, -2.5, 0],
    [0, 0, 0, 2.0032, 0, 0, 0, 0, 0, -2.0032, 0, 0],
    [0, 0, -2.5, 0, 16.667, 0, 0, 0, 2.5, 0, 8.3333, 0],
    [0, 0.625, 0, 0, 0, 4.1667, 0, -0.625, 0, 0, 0, 2.0833],
    [-50, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0],
    [0, -0.125, 0, 0, 0, -0.625, 0, 0.125, 0, 0, 0, -0.625],
    [0, 0, -0.5, 0, 2.5, 0, 0, 0, 0.5, 0, 2.5, 0],
    [0, 0, 0, -2.0032, 0, 0, 0, 0, 0, 2.0032, 0, 0],
    [0, 0, -2.5, 0, 8.3333, 0, 0, 0, 2.5, 0, 16.667, 0],
    [0, 0.625, 0, 0, 0, 2.0833, 0, -0.625, 0, 0, 0, 4.1667]
])

# Initialize structure
structure = Structure(nodes, elements, element_properties)

@pytest.mark.parametrize("elem_id, expected_matrix", [
    (0, expected_stiffness_0),
    (1, expected_stiffness_1)
])
def test_local_elastic_stiffness_matrix_3D_beam(elem_id, expected_matrix):
    """Test that local_elastic_stiffness_matrix_3D_beam computes correct stiffness matrices."""
    
    # Compute element length
    n1, n2 = elements[elem_id]
    L = structure.element_length(n1, n2)

    # Compute the stiffness matrix
    computed_matrix = structure.local_elastic_stiffness_matrix_3D_beam(elem_id, L)

    # Compare with expected matrix
    assert np.allclose(computed_matrix, expected_matrix, atol=1e-3), f"Mismatch in stiffness matrix for element {elem_id}"
