import numpy as np
import math
import pytest

# Import functions
from direct_stiffness_method.direct_stiffness_method import Structure, BoundaryConditions, Solver, PostProcessing, BucklingAnalysis, PlotResults

# I test against problems that I solved "by hand" using spreadsheets and MATLAB!

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

    def test_display_summary():
        """Test that display_summary produces the expected output."""

        # Define test input
        nodes_test = {
            0: [0, 0.0, 0.0, 10.0],
            1: [1, 15.0, 0.0, 10.0],
            2: [2, 15.0, 0.0, 0.0]
        }

        elements_test = [
            [0, 1],
            [1, 2]
        ]

        element_properties_test = {
            0: {"b": 0.5, "h": 1.0, "E": 1000, "nu": 0.3},
            1: {"b": 1.0, "h": 0.5, "E": 1000, "nu": 0.3}
        }

        # Expected output
        expected_output = """--- Structure Summary ---
    Number of Elements: 2
    Elasticity Modulus (E):
    Element 0: 1000
    Element 1: 1000

    Poisson's Ratio (nu):
    Element 0: 0.3
    Element 1: 0.3

    --- Element Properties ---
    Element 1:
    Length: 15.0000
    Area (A): 0.5000
    Moment of Inertia Iy: 0.0104
    Moment of Inertia Iz: 0.0417
    Polar Moment of Inertia J: 0.0521
    Node 1: (0.0, 0.0, 10.0), Node 2: (15.0, 0.0, 10.0)

    Element 2:
    Length: 10.0000
    Area (A): 0.5000
    Moment of Inertia Iy: 0.0417
    Moment of Inertia Iz: 0.0104
    Polar Moment of Inertia J: 0.0521
    Node 1: (15.0, 0.0, 10.0), Node 2: (15.0, 0.0, 0.0)

    --- Connectivity Matrix ---
    Element 1: [0 1]
    Element 2: [1 2]

    Global Node Numbering:
    Global Node 0: Coordinates (0.0, 0.0, 10.0)
    Global Node 1: Coordinates (15.0, 0.0, 10.0)
    Global Node 2: Coordinates (15.0, 0.0, 0.0)

    * * * * * * * * * *
    """

        # Capture output
        structure = Structure(nodes_test, elements_test, element_properties_test)
        captured_output = io.StringIO()
        sys.stdout = captured_output
        structure.display_summary()
        sys.stdout = sys.__stdout__  # Reset standard output

        # Compare captured output with expected output
        assert captured_output.getvalue().strip() == expected_output.strip(), "display_summary output mismatch."





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
expected_stiffness_matrices = {
    0: np.array([
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
    ]),
    1: np.array([
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
}

# Initialize structure
structure = Structure(nodes, elements, element_properties)

def test_compute_local_stiffness_matrices():
    """Test that compute_local_stiffness_matrices computes correct local stiffness matrices for all elements."""

    # Compute the local stiffness matrices
    computed_matrices = structure.compute_local_stiffness_matrices()

    # Check that all elements' matrices match expected values
    for elem_id, expected_matrix in expected_stiffness_matrices.items():
        computed_matrix = computed_matrices[elem_id]
        assert np.allclose(computed_matrix, expected_matrix, atol=1e-3), f"Mismatch in local stiffness matrix for element {elem_id}"

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

# Expected global stiffness matrices extracted from the images
expected_global_stiffness_matrices = {
    0: np.array([
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
    ]),
    1: np.array([
        [0.125, 0, 0, 0, -0.625, 0, -0.125, 0, 0, 0, -0.625, 0],
        [0, 0.5, 0, 2.5, 0, 0, 0, -0.5, 0, 2.5, 0, 0],
        [0, 0, 50, 0, 0, 0, 0, 0, -50, 0, 0, 0],
        [0, 2.5, 0, 16.667, 0, 0, 0, -2.5, 0, 8.3333, 0, 0],
        [-0.625, 0, 0, 0, 4.1667, 0, 0.625, 0, 0, 0, 2.0833, 0],
        [0, 0, 0, 0, 0, 2.0032, 0, 0, 0, 0, 0, -2.0032],
        [-0.125, 0, 0, 0, 0.625, 0, 0.125, 0, 0, 0, 0.625, 0],
        [0, -0.5, 0, -2.5, 0, 0, 0, 0.5, 0, -2.5, 0, 0],
        [0, 0, -50, 0, 0, 0, 0, 0, 50, 0, 0, 0],
        [0, 2.5, 0, 8.3333, 0, 0, 0, -2.5, 0, 16.667, 0, 0],
        [-0.625, 0, 0, 0, 2.0833, 0, 0.625, 0, 0, 0, 4.1667, 0],
        [0, 0, 0, 0, 0, -2.0032, 0, 0, 0, 0, 0, 2.0032]
    ])
}

# Initialize structure
structure = Structure(nodes, elements, element_properties)

def test_compute_global_stiffness_matrices():
    """Test that compute_global_stiffness_matrices computes correct global stiffness matrices."""
    
    # Compute the global stiffness matrices
    computed_matrices = structure.compute_global_stiffness_matrices()

    # Check that all elements' matrices match expected values
    for elem_id, expected_matrix in expected_global_stiffness_matrices.items():
        computed_matrix = computed_matrices[elem_id]
        assert np.allclose(computed_matrix, expected_matrix, atol=1e-3), f"Mismatch in global stiffness matrix for element {elem_id}"


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

# Hardcoded expected global stiffness matrix
expected_K_global = np.array([
    [33.333, 0, 0, 0, 0, 0, -33.333, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0],
    [0, 0.1481, 0, 0, 0, 1.1111, 0, -0.1481, 0, 0, 0, 1.1111, 0,  0,  0,  0,  0,  0],
    [0, 0, 0.037, 0, -0.2778, 0, 0, 0, -0.037, 0, -0.2778, 0, 0,  0,  0,  0,  0,  0],
    [0, 0, 0, 1.3355, 0, 0, 0, 0, 0, -1.3355, 0, 0, 0,  0,  0,  0,  0,  0],
    [0, 0, -0.2778, 0, 2.7778, 0, 0, 0, 0.2778, 0, 1.3889, 0, 0,  0,  0,  0,  0,  0],
    [0, 1.1111, 0, 0, 0, 11.111, 0, -1.1111, 0, 0, 0, 5.5556, 0,  0,  0,  0,  0,  0],
    [-33.333, 0, 0, 0, 0, 0, 33.458, 0, 0, 0, -0.625, 0, -0.125, 0,  0,  0, -0.625, 0],
    [0, -0.1481, 0, 0, 0, -1.1111, 0, 0.6481, 0, 2.5, 0, -1.1111, 0, -0.5,  0,  2.5,  0,  0],
    [0, 0, -0.037, 0, 0.2778, 0, 0, 0, 50.037, 0, 0.2778, 0, 0,  0, -50,  0,  0,  0],
    [0, 0, 0, -1.3355, 0, 0, 0, 2.5, 0, 18.002, 0, 0, 0, -2.5,  0,  8.3333,  0,  0],
    [0, 0, -0.2778, 0, 1.3889, 0, -0.625, 0, 0.2778, 0, 6.9444, 0, 0.625, 0,  0,  0,  2.0833,  0],
    [0, 1.1111, 0, 0, 0, 5.5556, 0, -1.1111, 0, 0, 0, 13.1143, 0, 0,  0,  0,  0, -2.0032],
    [0, 0, 0, 0, 0, 0, -0.125, 0, 0, 0, 0.625, 0, 0.125, 0,  0,  0,  0.625,  0],
    [0, 0, 0, 0, 0, 0, 0, -0.5, 0, -2.5, 0, 0, 0, 0.5,  0, -2.5,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0, -50, 0, 0, 0, 0, 0,  50,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 2.5, 0, 8.3333, 0, 0, 0, -2.5,  0,  16.667,  0,  0],
    [0, 0, 0, 0, 0, 0, -0.625, 0, 0, 0, 2.0833, 0, 0.625, 0,  0,  0,  4.1667,  0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.0032, 0, 0,  0,  0,  0,  2.0032]
])

# Initialize structure
structure = Structure(nodes, elements, element_properties)

def test_assemble_global_stiffness_matrix():
    """Test that assemble_global_stiffness_matrix computes the correct global stiffness matrix."""
    
    # Compute the assembled global stiffness matrix
    computed_K_global = structure.assemble_global_stiffness_matrix()

    # Compare with the expected matrix
    assert np.allclose(computed_K_global, expected_K_global, atol=1e-3), "Mismatch in assembled global stiffness matrix"

# Define test data
loads = {
    0: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    1: [1, 0.1, 0.05, -0.07, 0.05, -0.1, 0.25],
    2: [2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

supports = {
    0: [0, 1, 1, 1, 1, 1, 1],
    1: [1, 0, 0, 0, 0, 0, 0],
    2: [2, 1, 1, 1, 0, 0, 0]
}

# Expected global load vector
expected_F_global = np.array([
    [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
    [0.1], [0.05], [-0.07], [0.05], [-0.1], [0.25],
    [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
])

# Expected boundary conditions summary output
expected_boundary_conditions = {
    0: [1, 1, 1, 1, 1, 1],
    1: [0, 0, 0, 0, 0, 0],
    2: [1, 1, 1, 0, 0, 0]
}

# Expected print output for global load vector
expected_load_vector_output = (
    "\n--- External Load Vector ---\n"
    "[[ 0.  ]\n [ 0.  ]\n [ 0.  ]\n [ 0.  ]\n [ 0.  ]\n [ 0.  ]\n"
    " [ 0.1 ]\n [ 0.05]\n [-0.07]\n [ 0.05]\n [-0.1 ]\n [ 0.25]\n"
    " [ 0.  ]\n [ 0.  ]\n [ 0.  ]\n [ 0.  ]\n [ 0.  ]\n [ 0.  ]]\n"
)

# Initialize boundary conditions
bc = BoundaryConditions(loads, supports)

def test_compute_global_load_vector():
    """Test that compute_global_load_vector constructs the correct global load vector."""
    computed_F_global = bc.compute_global_load_vector()
    assert np.allclose(computed_F_global, expected_F_global, atol=1e-3), "Mismatch in global load vector"

def test_print_global_load_vector(capfd):
    """Test that print_global_load_vector prints the correct global load vector."""
    bc.print_global_load_vector()
    captured = capfd.readouterr().out
    assert expected_load_vector_output.strip() in captured.strip(), "Mismatch in printed global load vector"

def test_summarize_boundary_conditions(capfd):
    """Test that summarize_boundary_conditions prints the correct constraints."""
    bc.summarize_boundary_conditions()
    captured = capfd.readouterr().out

    # Check if the expected output appears in the captured output
    for node, constraints in expected_boundary_conditions.items():
        expected_line = f"Node {node}: Constraints {constraints}"
        assert expected_line in captured, f"Mismatch in boundary conditions output for Node {node}"

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

loads = {
    0: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    1: [1, 0.1, 0.05, -0.07, 0.05, -0.1, 0.25],
    2: [2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

supports = {
    0: [0, 1, 1, 1, 1, 1, 1],  # Node 0: All DOFs constrained
    1: [1, 0, 0, 0, 0, 0, 0],  # Node 1: All DOFs free
    2: [2, 1, 1, 1, 0, 0, 0]   # Node 2: First 3 DOFs constrained
}

# Expected constrained DOFs (Node 0: [0-5], Node 2: [12-14])
expected_constrained_dofs = [0, 1, 2, 3, 4, 5, 12, 13, 14]

# Expected displacement results
expected_U_global = np.array([
    [0.00000000e+00], [0.00000000e+00], [0.00000000e+00], 
    [0.00000000e+00], [0.00000000e+00], [0.00000000e+00],
    [2.84049953e-03], [1.43541318e+00], [-1.30609178e-03],
    [-1.26072079e-01], [-1.67293339e-02], [1.66041318e-01],
    [0.00000000e+00], [0.00000000e+00], [0.00000000e+00],
    [-1.52275937e-01], [8.79074190e-03], [1.66041318e-01]
])

# Expected reaction results
expected_R_global = np.array([
    [-0.09468332], [-0.02816345], [0.00469541],
    [0.16836549], [-0.02359799], [-0.67245177],
    [0.00000000], [0.00000000], [0.00000000],
    [0.00000000], [0.00000000], [0.00000000],
    [-0.00531668], [-0.02183655], [0.06530459],
    [0.00000000], [0.00000000], [0.00000000]
])

# Initialize structure, boundary conditions, and solver
structure = Structure(nodes, elements, element_properties)
boundary_conditions = BoundaryConditions(loads, supports)
solver = Solver(structure, boundary_conditions)

def test_get_constrained_dofs():
    """Test that get_constrained_dofs correctly identifies constrained degrees of freedom."""
    computed_constrained_dofs = solver.get_constrained_dofs()
    assert set(computed_constrained_dofs) == set(expected_constrained_dofs), "Mismatch in constrained DOFs"

def test_solve():
    """Test that the solve function computes the correct displacement vector."""
    computed_U_global = solver.solve()
    assert np.allclose(computed_U_global, expected_U_global, atol=1e-3), "Mismatch in computed displacements"

def test_compute_reactions():
    """Test that the compute_reactions function computes the correct reaction forces."""
    computed_U_global = solver.solve()
    computed_R_global = solver.compute_reactions(computed_U_global)
    assert np.allclose(computed_R_global, expected_R_global, atol=1e-3), "Mismatch in computed reactions"









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

# Expected global displacement results from previous solver tests
expected_U_global = np.array([
    [0.00000000e+00], [0.00000000e+00], [0.00000000e+00], 
    [0.00000000e+00], [0.00000000e+00], [0.00000000e+00],
    [2.84049953e-03], [1.43541318e+00], [-1.30609178e-03],
    [-1.26072079e-01], [-1.67293339e-02], [1.66041318e-01],
    [0.00000000e+00], [0.00000000e+00], [0.00000000e+00],
    [-1.52275937e-01], [8.79074190e-03], [1.66041318e-01]
])

# Expected internal forces for each element
expected_internal_forces = {
    0: np.array([-0.09468332, -0.02816345,  0.00469541,  0.16836549, -0.02359799, -0.67245177,
                  0.09468332,  0.02816345, -0.00469541, -0.16836549, -0.04683318,  0.25]),
    1: np.array([6.53045890e-02, -5.31668247e-03,  2.18365490e-02, -1.02529531e-17,
                -2.18365490e-01, -5.31668247e-02, -6.53045890e-02,  5.31668247e-03,
                -2.18365490e-02,  1.02529531e-17,  2.86968227e-16,  1.15548791e-18])
}

# Initialize structure
structure = Structure(nodes, elements, element_properties)

# Initialize PostProcessing
post_processor = PostProcessing(structure, expected_U_global)

def test_compute_internal_forces():
    """Test that compute_internal_forces computes the correct internal forces."""
    post_processor.compute_internal_forces()
    computed_internal_forces = post_processor.get_internal_forces()

    for elem_id, expected_forces in expected_internal_forces.items():
        assert np.allclose(computed_internal_forces[elem_id], expected_forces, atol=1e-3), \
            f"Mismatch in internal forces for element {elem_id}"

def test_get_internal_forces():
    """Test that get_internal_forces returns the expected forces after computation."""
    post_processor.compute_internal_forces()
    computed_internal_forces = post_processor.get_internal_forces()

    assert isinstance(computed_internal_forces, dict), "Internal forces should be returned as a dictionary"
    assert len(computed_internal_forces) == len(expected_internal_forces), "Mismatch in number of computed elements"

# def test_print_internal_forces(capfd):
#     """Test that print_internal_forces prints the correct internal forces."""
#     post_processor.compute_internal_forces()
#     post_processor.print_internal_forces()
    
#     captured = capfd.readouterr().out

#     # Check expected output in the printed text
#     assert "--- Internal Forces in Local Coordinates ---" in captured
#     for elem_id, forces in expected_internal_forces.items():
#         assert f"Element {elem_id + 1}:" in captured, f"Element {elem_id + 1} missing in print output"

#     # Check force values are printed
#     for value in expected_internal_forces[0]:
#         assert f"{value:.6f}" in captured, f"Value {value:.6f} missing from printed internal forces"