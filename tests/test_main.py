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


def test_compute_global_stiffness_matrix():
    # Define nodes
    nodes = {
        0: [0, 0.0, 0.0, 0.0],
        1: [1, 3.0, 9.333333333, 7.333333333],
        2: [2, 6.0, 18.66666667, 14.66666667],
    }

    # Define elements
    elements = [
        [0, 1],
        [1, 2],
    ]

    # Define element properties
    element_properties = {
        0: {"r": 1.0, "E": 10000, "nu": 0.3},
        1: {"r": 1.0, "E": 10000, "nu": 0.3},
    }

    # Initialize structure
    structure = Structure(nodes, elements, element_properties)

    # Compute global stiffness matrix
    K_global = structure.compute_global_stiffness_matrix()

    # Expected result
    expected_K_global = np.array([
        [ 2.02352366e+02,  4.69756988e+02,  3.69094776e+02,  5.70847168e-15,
   1.88316510e+02, -2.39675559e+02, -2.02352366e+02, -4.69756988e+02,
  -3.69094776e+02,  5.70847168e-15,  1.88316510e+02, -2.39675559e+02,
   0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
   0.00000000e+00,  0.00000000e+00,],

[ 4.69756988e+02,  1.51282523e+03,  1.14829486e+03, -1.88316510e+02,  
  5.44929300e-15,  7.70385724e+01, -4.69756988e+02, -1.51282523e+03,  
 -1.14829486e+03, -1.88316510e+02,  5.44929300e-15,  7.70385724e+01,  
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00,],

[ 3.69094776e+02,  1.14829486e+03,  9.53590723e+02,  2.39675559e+02,  
 -7.70385724e+01,  0.00000000e+00, -3.69094776e+02, -1.14829486e+03,  
 -9.53590723e+02,  2.39675559e+02, -7.70385724e+01,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00,],

[-5.70847168e-15, -1.88316510e+02,  2.39675559e+02,  2.44160330e+03,  
 -3.87168210e+02, -3.04203593e+02,  5.70847168e-15,  1.88316510e+02,  
 -2.39675559e+02,  1.17635632e+03, -3.31858466e+02, -2.60745937e+02,  
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00,],

[ 1.88316510e+02, -5.44929300e-15, -7.70385724e+01, -3.87168210e+02,  
  1.36152691e+03, -9.46411180e+02, -1.88316510e+02,  5.44929300e-15,  
  7.70385724e+01, -3.31858466e+02,  2.50576553e+02, -8.11209583e+02,  
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00,],

[-2.39675559e+02,  7.70385724e+01,  0.00000000e+00, -3.04203593e+02,  
 -9.46411180e+02,  1.82244144e+03,  2.39675559e+02, -7.70385724e+01,  
  0.00000000e+00, -2.60745937e+02, -8.11209583e+02,  6.45646155e+02,  
  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00,],

[-2.02352366e+02, -4.69756988e+02, -3.69094776e+02, -5.70847168e-15,  
 -1.88316510e+02,  2.39675559e+02,  4.04704731e+02,  9.39513975e+02,  
  7.38189552e+02, -7.12389973e-15, -1.48556751e-07,  2.17086210e-07,  
 -2.02352366e+02, -4.69756987e+02, -3.69094776e+02, -1.41542805e-15,  
  1.88316510e+02, -2.39675558e+02,],

[-4.69756988e+02, -1.51282523e+03, -1.14829486e+03,  1.88316510e+02,  
 -5.44929300e-15, -7.70385724e+01,  9.39513975e+02,  3.02565046e+03,  
  2.29658972e+03,  1.48556779e-07, -6.71270072e-15, -1.02794218e-07,  
 -4.69756987e+02, -1.51282523e+03, -1.14829486e+03, -1.88316510e+02,  
 -1.26340773e-15,  7.70385723e+01,],

[-3.69094776e+02, -1.14829486e+03, -9.53590723e+02, -2.39675559e+02,  
  7.70385724e+01,  0.00000000e+00,  7.38189552e+02,  2.29658972e+03,  
  1.90718145e+03, -2.17086153e-07,  1.02794218e-07,  0.00000000e+00,  
 -3.69094776e+02, -1.14829486e+03, -9.53590723e+02,  2.39675558e+02,  
 -7.70385723e+01,  0.00000000e+00,],

[-5.70847168e-15, -1.88316510e+02,  2.39675559e+02,  1.17635632e+03,  
 -3.31858466e+02, -2.60745937e+02,  7.12389973e-15,  1.48556751e-07,  
 -2.17086210e-07,  4.88320660e+03, -7.74336419e+02, -6.08407187e+02,  
 -1.41542805e-15,  1.88316510e+02, -2.39675558e+02,  1.17635632e+03,  
 -3.31858465e+02, -2.60745937e+02,],

[ 1.88316510e+02, -5.44929300e-15, -7.70385724e+01, -3.31858466e+02,  
  2.50576553e+02, -8.11209583e+02, -1.48556779e-07,  6.71270072e-15,  
  1.02794218e-07, -7.74336419e+02,  2.72305381e+03, -1.89282236e+03,  
 -1.88316510e+02, -1.26340773e-15,  7.70385723e+01, -3.31858465e+02,  
  2.50576553e+02, -8.11209582e+02,],

[-2.39675559e+02,  7.70385724e+01,  0.00000000e+00, -2.60745937e+02,  
 -8.11209583e+02,  6.45646155e+02,  2.17086153e-07, -1.02794218e-07,  
  0.00000000e+00, -6.08407187e+02, -1.89282236e+03,  3.64488288e+03,  
  2.39675558e+02, -7.70385723e+01,  0.00000000e+00, -2.60745937e+02,  
 -8.11209582e+02,  6.45646155e+02,],

[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00, -2.02352366e+02, -4.69756987e+02,  
 -3.69094776e+02,  1.41542805e-15, -1.88316510e+02,  2.39675558e+02,  
  2.02352366e+02,  4.69756987e+02,  3.69094776e+02,  1.41542805e-15,  
 -1.88316510e+02,  2.39675558e+02,],

[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00, -4.69756987e+02, -1.51282523e+03,  
 -1.14829486e+03,  1.88316510e+02,  1.26340773e-15, -7.70385723e+01,  
  4.69756987e+02,  1.51282523e+03,  1.14829486e+03,  1.88316510e+02,  
  1.26340773e-15, -7.70385723e+01,],

[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00, -3.69094776e+02, -1.14829486e+03,  
 -9.53590723e+02, -2.39675558e+02,  7.70385723e+01,  0.00000000e+00,  
  3.69094776e+02,  1.14829486e+03,  9.53590723e+02, -2.39675558e+02,  
  7.70385723e+01,  0.00000000e+00,],

[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00,  1.41542805e-15, -1.88316510e+02,  
  2.39675558e+02,  1.17635632e+03, -3.31858465e+02, -2.60745937e+02,  
 -1.41542805e-15,  1.88316510e+02, -2.39675558e+02,  2.44160330e+03,  
 -3.87168210e+02, -3.04203593e+02,],

[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00,  1.88316510e+02,  1.26340773e-15,  
 -7.70385723e+01, -3.31858465e+02,  2.50576553e+02, -8.11209582e+02,  
 -1.88316510e+02, -1.26340773e-15,  7.70385723e+01, -3.87168210e+02,  
  1.36152691e+03, -9.46411179e+02,],

[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  
  0.00000000e+00,  0.00000000e+00, -2.39675558e+02,  7.70385723e+01,  
  0.00000000e+00, -2.60745937e+02, -8.11209582e+02,  6.45646155e+02,  
  2.39675558e+02, -7.70385723e+01,  0.00000000e+00, -3.04203593e+02,  
 -9.46411179e+02,  1.82244144e+03,]

    ])

    # Compare matrices with tolerance to handle floating-point errors
    np.testing.assert_allclose(K_global, expected_K_global, rtol=1e-5, atol=1e-5)


def test_compute_global_load_vector():
    # Define test loads
    loads = {
        0: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1: [1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        2: [2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        3: [3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        4: [4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        5: [5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        6: [6, 0.05, -0.1, 0.23, 0.1, -0.025, -0.08]
    }

    # No supports needed for this test
    supports = {}

    # Initialize BoundaryConditions class
    bc = BoundaryConditions(loads, supports)

    # Compute global load vector
    F_global = bc.compute_global_load_vector()

    # Define expected output
    expected_F_global = np.zeros((bc.n_nodes * 6, 1))
    expected_F_global[36:42, 0] = [0.05, -0.1, 0.23, 0.1, -0.025, -0.08]

    # Assert that computed and expected values match
    np.testing.assert_array_almost_equal(F_global, expected_F_global)

def test_solver():
    # Define nodes
    nodes = {
        0: [0, 0.0, 0.0, 0.0],
        1: [1, 3.0, 9.333333333, 7.333333333],
        2: [2, 6.0, 18.66666667, 14.66666667],
    }

    # Define elements
    elements = [
        [0, 1],
        [1, 2],
    ]

    # Define element properties
    element_properties = {
        0: {"r": 1.0, "E": 10000, "nu": 0.3},
        1: {"r": 1.0, "E": 10000, "nu": 0.3},
    }

    # Initialize structure
    structure = Structure(nodes, elements, element_properties)

    # Define loads
    loads = {
        0: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1: [1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        2: [2, 0.05, -0.1, 0.23, 0.1, -0.025, -0.08]    
    }

    # Define supports
    supports = {
        0: [0, 1, 1, 1, 1, 1, 1],
        1: [1, 0, 0, 0, 0, 0, 0],
        2: [2, 0, 0, 0, 0, 0, 0],
    }

    # Initialize boundary conditions
    bc = BoundaryConditions(loads, supports)

    # Initialize solver
    solver = Solver(structure, bc)

    # Solve for displacements
    U_global = solver.solve()

    # Expected displacements
    expected_U_global = np.array([
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.00666117],
        [-0.03115959],
        [ 0.03698065],
        [ 0.0068851 ],
        [-0.00081014],
        [-0.00192924],
        [ 0.02165854],
        [-0.10034391],
        [ 0.11894612],
        [ 0.00928081],
        [-0.00111626],
        [-0.00266339],
    ])

    # Check displacements
    np.testing.assert_allclose(U_global, expected_U_global, rtol=1e-5, atol=1e-5)

    # Compute reaction forces
    R_global = solver.compute_reactions(U_global)

    # Expected reaction forces
    expected_R_global = np.array([
        [-0.05      ],
        [ 0.1       ],
        [-0.23      ],
        [-5.86      ],
        [ 0.67166667],
        [ 1.61333333],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ],
        [ 0.        ]
    ])

    # Check reaction forces
    np.testing.assert_allclose(R_global, expected_R_global, rtol=1e-5, atol=1e-5)















def test_compute_internal_forces():
    # Define nodes
    nodes = {
        0: [0, 0.0, 0.0, 0.0],
    1: [1, 3.0, 9.333333333, 7.333333333],
    2: [2, 6.0, 18.66666667, 14.66666667],
    3: [3, 9.0, 28.0, 22.0],
    4: [4, 12.0, 37.33333333, 29.33333333],
    5: [5, 15.0, 46.66666667, 36.66666667],
    6: [6, 18.0, 56.0, 44.0],
    }

    # Define elements
    elements = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6]
    ]

    # Define element properties
    element_properties = {
        0: {"r": 1.0, "E": 10000, "nu": 0.3},
        1: {"r": 1.0, "E": 10000, "nu": 0.3},
        2: {"r": 1.0, "E": 10000, "nu": 0.3},
        3: {"r": 1.0, "E": 10000, "nu": 0.3},
        4: {"r": 1.0, "E": 10000, "nu": 0.3},
        5: {"r": 1.0, "E": 10000, "nu": 0.3},
        6: {"r": 1.0, "E": 10000, "nu": 0.3},
    }

    # Initialize structure
    structure = Structure(nodes, elements, element_properties)

    # Define loads
    loads = {
        0: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1: [1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        2: [2, 0.0, 0.0, 0.0, 0.0, 0., 0.0],
        3: [3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        4: [4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        5: [5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        6: [6, -0.245039747, -0.762345878, -0.598986047, 0.0, 0.0, 0.0]    
    }

    # Define supports
    supports = {
        0: [0, 1, 1, 1, 1, 1, 1],
        1: [1, 0, 0, 0, 0, 0, 0],
        2: [2, 0, 0, 0, 0, 0, 0],
        3: [3, 0, 0, 0, 0, 0, 0],
        4: [4, 0, 0, 0, 0, 0, 0],
        5: [5, 0, 0, 0, 0, 0, 0],
        6: [6, 0, 0, 0, 0, 0, 0]    
    }

    # Initialize boundary conditions
    bc = BoundaryConditions(loads, supports)

    # Initialize solver and solve for displacements
    solver = Solver(structure, bc)
    U_global = solver.solve()

    # Perform buckling analysis
    buckling_analysis = BucklingAnalysis(structure, solver, U_global)
    internal_forces = buckling_analysis.internal_forces

    # Expected internal forces
    expected_internal_forces = {
        0: np.array([1.00000000e+00, -4.67682450e-10, -8.53716865e-11,  1.86581954e-16,
                      6.73220874e-09, -3.49667949e-08, -1.00000000e+00,  4.67682450e-10,
                      8.53716865e-11, -1.86581954e-16, -5.68701071e-09,  2.92409999e-08]),
        1: np.array([1.00000000e+00, -5.67661704e-10, -1.60683099e-10,  1.85863289e-16,
                      5.68701070e-09, -2.92409999e-08, -1.00000000e+00,  5.67661704e-10,
                      1.60683099e-10, -1.85863289e-16, -3.71978170e-09,  2.22911678e-08]),
        2: np.array([1.00000000e+00, -3.92698073e-10, -2.88882234e-11,  2.12524527e-16,
                      3.71978169e-09, -2.22911679e-08, -1.00000000e+00,  3.92698073e-10,
                      2.88882234e-11, -2.12524527e-16, -3.36610572e-09,  1.74834000e-08]),
        3: np.array([1.00000000e+00, -3.92698134e-10, -2.88880595e-11,  1.76681182e-16,
                      3.36610575e-09, -1.74834001e-08, -1.00000000e+00,  3.92698134e-10,
                      2.88880595e-11, -1.76681182e-16, -3.01243178e-09,  1.26756315e-08]),
        4: np.array([1.00000000e+00, -6.42646229e-10, -2.17167308e-10,  1.61366379e-16,
                      3.01243178e-09, -1.26756315e-08, -1.00000000e+00,  6.42646229e-10,
                      2.17167308e-10, -1.61366379e-16, -3.53671592e-10,  4.80777038e-09]),
        5: np.array([1.00000000e+00, -3.92698245e-10, -2.88878421e-11,  1.07796531e-16,
                      3.53671551e-10, -4.80777063e-09, -1.00000000e+00,  3.92698245e-10,
                      2.88878421e-11, -1.07796531e-16, -2.49202677e-16,  6.59562327e-16]),
    }

    # Check internal forces for each element
    for elem_id, expected_forces in expected_internal_forces.items():
        np.testing.assert_allclose(internal_forces[elem_id], expected_forces, rtol=1e-5, atol=1e-5)


