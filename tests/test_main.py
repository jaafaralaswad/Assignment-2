import numpy as np
import math
import pytest

# Import functions
from direct_stiffness_method.direct_stiffness_method import Structure, BoundaryConditions, Solver, PostProcessing, BucklingAnalysis, PlotResults

def test_rectangular_section():
    nodes = {}
    elements = {}
    element_properties = {
        1: {"E": 1000, "nu": 0.3, "b": 0.2, "h": 0.4}
    }
    
    structure = Structure(nodes, elements, element_properties)
    A, Iy, Iz, J = structure.compute_section_properties(1)
    
    assert A == pytest.approx(0.08)
    assert Iy == pytest.approx((0.4 * 0.2**3) / 12)
    assert Iz == pytest.approx((0.2 * 0.4**3) / 12)
    assert J == pytest.approx(Iy + Iz)

def test_circular_section():
    nodes = {}
    elements = {}
    element_properties = {
        2: {"E": 1000, "nu": 0.3, "r": 0.1}
    }
    
    structure = Structure(nodes, elements, element_properties)
    A, Iy, Iz, J = structure.compute_section_properties(2)
    
    assert A == pytest.approx(math.pi * 0.1**2)
    assert Iy == pytest.approx((math.pi * 0.1**4) / 4)
    assert Iz == pytest.approx((math.pi * 0.1**4) / 4)
    assert J == pytest.approx(Iy + Iz)

def test_invalid_element_properties():
    nodes = {}
    elements = {}
    element_properties = {
        3: {"E": 1000, "nu": 0.3}
    }
    
    structure = Structure(nodes, elements, element_properties)
    
    with pytest.raises(ValueError, match="Invalid element properties"):
        structure.compute_section_properties(3)

def test_element_length():
    nodes = {
        0: (0, 0.0, 0.0, 0.0),
        1: (1, 3.0, 4.0, 0.0),
        2: (2, 1.0, 1.0, 1.0),
        3: (3, 4.0, 5.0, 6.0)
    }
    elements = {}
    element_properties = {}

    structure = Structure(nodes, elements, element_properties)

    # Test 2D distance
    assert structure.element_length(0, 1) == pytest.approx(5.0)

    # Test 3D distance
    expected_length = np.sqrt((4 - 1) ** 2 + (5 - 1) ** 2 + (6 - 1) ** 2)
    assert structure.element_length(2, 3) == pytest.approx(expected_length)

    # Test zero length (same node)
    assert structure.element_length(0, 0) == pytest.approx(0.0)

def test_display_summary(capfd):
    # Define nodes and their coordinates
    nodes = {
        0: [0, 0.0, 0.0, 10.0],
        1: [1, 15.0, 0.0, 10.0],
        2: [2, 15.0, 0.0, 0.0]
    }

    # Define elements
    elements = [
        [0, 1],
        [1, 2]
    ]

    # Define element properties
    element_properties = {
        0: {"b": 0.5, "h": 1.0, "E": 1000, "nu": 0.3},
        1: {"b": 1.0, "h": 0.5, "E": 1000, "nu": 0.3}
    }

    # Create structure object and call display_summary()
    structure = Structure(nodes, elements, element_properties)
    structure.display_summary()

    # Capture printed output
    captured = capfd.readouterr()
    output = captured.out.strip()

    # Expected output
    expected_output = """\
--- Structure Summary ---
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

* * * * * * * * * *"""

    # Assert expected output
    assert output == expected_output, f"\nExpected:\n{expected_output}\n\nGot:\n{output}"

def test_local_elastic_stiffness_matrix_3D_beam():
    # Define nodes and their coordinates
    nodes = {
        0: [0, 0.0, 0.0, 10.0],
        1: [1, 15.0, 0.0, 10.0],
        2: [2, 15.0, 0.0, 0.0]
    }

    # Define elements
    elements = [
        [0, 1],
        [1, 2]
    ]

    # Define element properties
    element_properties = {
        0: {"b": 0.5, "h": 1.0, "E": 1000, "nu": 0.3},
        1: {"b": 1.0, "h": 0.5, "E": 1000, "nu": 0.3}
    }

    # Create structure object
    structure = Structure(nodes, elements, element_properties)

    # Compute the stiffness matrix for element 0
    L = 15.0  # Length of element 0
    k_e = structure.local_elastic_stiffness_matrix_3D_beam(0, L)

    # Expected values (rounded for comparison)
    E = element_properties[0]["E"]
    nu = element_properties[0]["nu"]
    A, Iy, Iz, J = structure.compute_section_properties(0)

    axial_stiffness = E * A / L
    torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
    bending_zz = E * 12.0 * Iz / L ** 3.0
    bending_yy = E * 12.0 * Iy / L ** 3.0
    bending_zz_coupling = E * 6.0 * Iz / L ** 2.0
    bending_yy_coupling = E * 6.0 * Iy / L ** 2.0
    bending_zz_self = E * 4.0 * Iz / L
    bending_yy_self = E * 4.0 * Iy / L
    bending_zz_half = E * 2.0 * Iz / L
    bending_yy_half = E * 2.0 * Iy / L

    # Check all components of stiffness matrix

    ## Axial stiffness
    assert k_e[0, 0] == pytest.approx(axial_stiffness)
    assert k_e[0, 6] == pytest.approx(-axial_stiffness)
    assert k_e[6, 0] == pytest.approx(-axial_stiffness)
    assert k_e[6, 6] == pytest.approx(axial_stiffness)

    ## Torsional stiffness
    assert k_e[3, 3] == pytest.approx(torsional_stiffness)
    assert k_e[3, 9] == pytest.approx(-torsional_stiffness)
    assert k_e[9, 3] == pytest.approx(-torsional_stiffness)
    assert k_e[9, 9] == pytest.approx(torsional_stiffness)

    ## Bending stiffness about local z-axis (strong axis)
    assert k_e[1, 1] == pytest.approx(bending_zz)
    assert k_e[1, 7] == pytest.approx(-bending_zz)
    assert k_e[7, 1] == pytest.approx(-bending_zz)
    assert k_e[7, 7] == pytest.approx(bending_zz)
    assert k_e[1, 5] == pytest.approx(bending_zz_coupling)
    assert k_e[5, 1] == pytest.approx(bending_zz_coupling)
    assert k_e[1, 11] == pytest.approx(bending_zz_coupling)
    assert k_e[11, 1] == pytest.approx(bending_zz_coupling)
    assert k_e[5, 7] == pytest.approx(-bending_zz_coupling)
    assert k_e[7, 5] == pytest.approx(-bending_zz_coupling)
    assert k_e[7, 11] == pytest.approx(-bending_zz_coupling)
    assert k_e[11, 7] == pytest.approx(-bending_zz_coupling)
    assert k_e[5, 5] == pytest.approx(bending_zz_self)
    assert k_e[11, 11] == pytest.approx(bending_zz_self)
    assert k_e[5, 11] == pytest.approx(bending_zz_half)
    assert k_e[11, 5] == pytest.approx(bending_zz_half)

    ## Bending stiffness about local y-axis (weak axis)
    assert k_e[2, 2] == pytest.approx(bending_yy)
    assert k_e[2, 8] == pytest.approx(-bending_yy)
    assert k_e[8, 2] == pytest.approx(-bending_yy)
    assert k_e[8, 8] == pytest.approx(bending_yy)
    assert k_e[2, 4] == pytest.approx(-bending_yy_coupling)
    assert k_e[4, 2] == pytest.approx(-bending_yy_coupling)
    assert k_e[2, 10] == pytest.approx(-bending_yy_coupling)
    assert k_e[10, 2] == pytest.approx(-bending_yy_coupling)
    assert k_e[4, 8] == pytest.approx(bending_yy_coupling)
    assert k_e[8, 4] == pytest.approx(bending_yy_coupling)
    assert k_e[8, 10] == pytest.approx(bending_yy_coupling)
    assert k_e[10, 8] == pytest.approx(bending_yy_coupling)
    assert k_e[4, 4] == pytest.approx(bending_yy_self)
    assert k_e[10, 10] == pytest.approx(bending_yy_self)
    assert k_e[4, 10] == pytest.approx(bending_yy_half)
    assert k_e[10, 4] == pytest.approx(bending_yy_half)

    ## Check symmetry of the matrix
    assert np.allclose(k_e, k_e.T)

def test_compute_local_stiffness_matrices():
    # Define nodes and their coordinates
    nodes = {
        0: [0, 0.0, 0.0, 10.0],
        1: [1, 15.0, 0.0, 10.0],
        2: [2, 15.0, 0.0, 0.0]
    }

    # Define elements
    elements = [
        [0, 1],
        [1, 2]
    ]

    # Define element properties
    element_properties = {
        0: {"b": 0.5, "h": 1.0, "E": 1000, "nu": 0.3},
        1: {"b": 1.0, "h": 0.5, "E": 1000, "nu": 0.3}
    }

    # Create structure object
    structure = Structure(nodes, elements, element_properties)

    # Compute local stiffness matrices
    stiffness_matrices = structure.compute_local_stiffness_matrices()

    # Ensure correct number of matrices
    assert len(stiffness_matrices) == len(elements), "Incorrect number of stiffness matrices"

    for elem_id, (n1, n2) in enumerate(elements):
        # Compute expected stiffness matrix
        L = structure.element_length(n1, n2)
        expected_k_e = structure.local_elastic_stiffness_matrix_3D_beam(elem_id, L)

        # Check if matrix exists in results
        assert elem_id in stiffness_matrices, f"Element {elem_id} missing from stiffness matrices"

        # Extract computed matrix
        computed_k_e = stiffness_matrices[elem_id]

        # Ensure shape is correct (12x12)
        assert computed_k_e.shape == (12, 12), f"Stiffness matrix for element {elem_id} has incorrect shape"

        # Check if the computed matrix matches the expected matrix
        assert np.allclose(computed_k_e, expected_k_e), f"Stiffness matrix for element {elem_id} does not match expected values"

        # Ensure symmetry of the stiffness matrix
        assert np.allclose(computed_k_e, computed_k_e.T), f"Stiffness matrix for element {elem_id} is not symmetric"

def test_compute_global_stiffness_matrices():
    # Define nodes and their coordinates
    nodes = {
        0: [0, 0.0, 0.0, 10.0],
        1: [1, 15.0, 0.0, 10.0],
        2: [2, 15.0, 0.0, 0.0]
    }

    # Define elements
    elements = [
        [0, 1],
        [1, 2]
    ]

    # Define element properties
    element_properties = {
        0: {"b": 0.5, "h": 1.0, "E": 1000, "nu": 0.3},
        1: {"b": 1.0, "h": 0.5, "E": 1000, "nu": 0.3}
    }

    # Create structure object
    structure = Structure(nodes, elements, element_properties)

    # Compute global stiffness matrices
    global_stiffness_matrices = structure.compute_global_stiffness_matrices()

    # Ensure correct number of global stiffness matrices
    assert len(global_stiffness_matrices) == len(elements), "Incorrect number of global stiffness matrices"

    for elem_id, (n1, n2) in enumerate(elements):
        # Compute expected local stiffness matrix
        L = structure.element_length(n1, n2)
        k_local = structure.local_elastic_stiffness_matrix_3D_beam(elem_id, L)

        # Compute expected transformation matrices
        x1, y1, z1 = nodes[n1][1:]
        x2, y2, z2 = nodes[n2][1:]

        gamma = rotation_matrix_3D(x1, y1, z1, x2, y2, z2)
        Gamma = transformation_matrix_3D(gamma)

        # Compute expected global stiffness matrix
        expected_k_global = Gamma.T @ k_local @ Gamma

        # Check if matrix exists in results
        assert elem_id in global_stiffness_matrices, f"Element {elem_id} missing from global stiffness matrices"

        # Extract computed global stiffness matrix
        computed_k_global = global_stiffness_matrices[elem_id]

        # Ensure shape is correct (12x12)
        assert computed_k_global.shape == (12, 12), f"Global stiffness matrix for element {elem_id} has incorrect shape"

        # Check if the computed matrix matches the expected matrix
        assert np.allclose(computed_k_global, expected_k_global), f"Global stiffness matrix for element {elem_id} does not match expected values"

        # Ensure symmetry of the global stiffness matrix
        assert np.allclose(computed_k_global, computed_k_global.T), f"Global stiffness matrix for element {elem_id} is not symmetric"


def rotation_matrix_3D(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, v_temp: np.ndarray = None):
    """
    3D rotation matrix
    source: Chapter 5.1 of McGuire's Matrix Structural Analysis 2nd Edition
    Given:
        x, y, z coordinates of the ends of two beams: x1, y1, z1, x2, y2, z2
        optional: reference vector v_temp to orthonormalize the local y and z axes.
            If v_temp is not provided, a default is chosen based on the beam orientation.
    Compute:
        A 3x3 rotation matrix where the rows represent the local x, y, and z axes in global coordinates.
    """
    L = np.sqrt((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0 + (z2 - z1) ** 2.0)
    lxp = (x2 - x1) / L
    mxp = (y2 - y1) / L
    nxp = (z2 - z1) / L
    local_x = np.asarray([lxp, mxp, nxp])

    # Choose a vector to orthonormalize the local y axis if one is not given
    if v_temp is None:
        # if the beam is oriented vertically, switch to the global y axis
        if np.isclose(lxp, 0.0) and np.isclose(mxp, 0.0):
            v_temp = np.array([0, 1.0, 0.0])
        else:
            # otherwise use the global z axis
            v_temp = np.array([0, 0, 1.0])
    else:
        check_unit_vector(v_temp)
        check_parallel(local_x, v_temp)
    
    # Compute the local y axis by taking the cross product of v_temp and local_x
    local_y = np.cross(v_temp, local_x)
    local_y = local_y / np.linalg.norm(local_y)

    # Compute the local z axis
    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)

    # Assemble the rotation matrix (gamma)
    gamma = np.vstack((local_x, local_y, local_z))
    
    return gamma

def transformation_matrix_3D(gamma: np.ndarray) -> np.ndarray:
    """
    3D transformation matrix
    source: Chapter 5.1 of McGuire's Matrix Structural Analysis 2nd Edition
    Given:
        gamma -- the 3x3 rotation matrix
    Compute:
        Gamma -- the 12x12 transformation matrix which maps the 6 DOFs at each node.
    """
    Gamma = np.zeros((12, 12))
    Gamma[0:3, 0:3] = gamma
    Gamma[3:6, 3:6] = gamma
    Gamma[6:9, 6:9] = gamma
    Gamma[9:12, 9:12] = gamma
    return Gamma

def test_assemble_global_stiffness_matrix():
    # Define nodes and their coordinates
    nodes = {
        0: [0, 0.0, 0.0, 10.0],
        1: [1, 15.0, 0.0, 10.0],
        2: [2, 15.0, 0.0, 0.0]
    }

    # Define elements
    elements = [
        [0, 1],
        [1, 2]
    ]

    # Define element properties
    element_properties = {
        0: {"b": 0.5, "h": 1.0, "E": 1000, "nu": 0.3},
        1: {"b": 1.0, "h": 0.5, "E": 1000, "nu": 0.3}
    }

    # Create structure object
    structure = Structure(nodes, elements, element_properties)

    # Compute assembled global stiffness matrix
    K_global = structure.assemble_global_stiffness_matrix()

    # Compute expected global stiffness matrix
    n_global_nodes = len(nodes)  
    total_dofs = n_global_nodes * 6
    expected_K_global = np.zeros((total_dofs, total_dofs))

    # Get the global stiffness matrices for each element
    global_stiffness_matrices = structure.compute_global_stiffness_matrices()

    for elem_idx, (node1, node2) in enumerate(elements):
        # Determine the corresponding global DOF indices
        dofs = np.concatenate((
            np.arange(node1 * 6, node1 * 6 + 6),
            np.arange(node2 * 6, node2 * 6 + 6)
        ))

        # Add the element's contribution into the expected global stiffness matrix
        k_global = global_stiffness_matrices[elem_idx]
        for i_local in range(12):
            global_i = dofs[i_local]
            for j_local in range(12):
                global_j = dofs[j_local]
                expected_K_global[global_i, global_j] += k_global[i_local, j_local]

    # Ensure the computed global stiffness matrix has the correct shape
    assert K_global.shape == (total_dofs, total_dofs), "Global stiffness matrix has incorrect shape"

    # Check if the computed matrix matches the expected matrix
    assert np.allclose(K_global, expected_K_global), "Assembled global stiffness matrix does not match expected values"

    # Ensure symmetry of the global stiffness matrix
    assert np.allclose(K_global, K_global.T), "Global stiffness matrix is not symmetric"

def test_print_global_stiffness_matrix(capfd):
    # Define nodes and their coordinates
    nodes = {
        0: [0, 0.0, 0.0, 10.0],
        1: [1, 15.0, 0.0, 10.0],
        2: [2, 15.0, 0.0, 0.0]
    }

    # Define elements
    elements = [
        [0, 1],
        [1, 2]
    ]

    # Define element properties
    element_properties = {
        0: {"b": 0.5, "h": 1.0, "E": 1000, "nu": 0.3},
        1: {"b": 1.0, "h": 0.5, "E": 1000, "nu": 0.3}
    }

    # Create structure object
    structure = Structure(nodes, elements, element_properties)

    # Capture printed output
    structure.print_global_stiffness_matrix()
    captured = capfd.readouterr()

    # Compute expected global stiffness matrix
    K_global = structure.assemble_global_stiffness_matrix()

    # Convert expected output to a string (similar to print output)
    expected_output = np.array2string(K_global)

    # Ensure printed output matches expected matrix
    assert captured.out.strip() == expected_output.strip(), "Printed global stiffness matrix does not match expected values"














def test_compute_global_load_vector():
    # Define loads (node: [node_id, Fx, Fy, Fz, Mx, My, Mz])
    loads = {
        0: [0, 10, 0, -5, 0, 0, 0],  # Node 0 has forces/moments
        1: [1, 0, 20, 0, 0, -10, 0], # Node 1 has forces/moments
        2: [2, 5, 5, 5, 5, 5, 5]     # Node 2 has all components
    }

    # Define supports (not used in load vector calculation)
    supports = {
        0: [0, 1, 1, 1, 0, 0, 0],  # Fixed in x, y, z
        1: [1, 0, 1, 0, 1, 1, 0],  # Some constraints
        2: [2, 0, 0, 0, 0, 0, 0]   # Free node
    }

    # Create boundary conditions object
    bc = BoundaryConditions(loads, supports)

    # Compute expected global load vector
    total_dofs = len(loads) * 6
    expected_F_global = np.zeros((total_dofs, 1))

    for node, values in loads.items():
        dof_index = node * 6
        expected_F_global[dof_index:dof_index + 6, 0] = values[1:]  # Exclude node number

    # Compute actual load vector
    computed_F_global = bc.compute_global_load_vector()

    # Ensure the computed load vector matches the expected values
    assert computed_F_global.shape == (total_dofs, 1), "Global load vector has incorrect shape"
    assert np.allclose(computed_F_global, expected_F_global), "Global load vector values are incorrect"

def test_print_global_load_vector(capfd):
    # Define loads and supports
    loads = {
        0: [0, 10, 0, -5, 0, 0, 0],
        1: [1, 0, 20, 0, 0, -10, 0],
        2: [2, 5, 5, 5, 5, 5, 5]
    }
    supports = {0: [0, 1, 1, 1, 0, 0, 0]}

    # Create boundary conditions object
    bc = BoundaryConditions(loads, supports)

    # Capture printed output
    bc.print_global_load_vector()
    captured = capfd.readouterr()

    # Compute expected load vector
    F_global = bc.compute_global_load_vector()
    expected_output = "\n--- External Load Vector ---\n" + np.array2string(F_global)

    # Ensure printed output matches expected values
    assert captured.out.strip() == expected_output.strip(), "Printed global load vector does not match expected output"

def test_summarize_boundary_conditions(capfd):
    # Define loads (not used here) and supports
    loads = {0: [0, 0, 0, 0, 0, 0, 0]}
    supports = {
        0: [0, 1, 1, 1, 0, 0, 0],  # Fixed in x, y, z
        1: [1, 0, 1, 0, 1, 1, 0],  # Some constraints
        2: [2, 0, 0, 0, 0, 0, 0]   # Free node
    }

    # Create boundary conditions object
    bc = BoundaryConditions(loads, supports)

    # Capture printed output
    bc.summarize_boundary_conditions()
    captured = capfd.readouterr()

    # Expected output
    expected_output = """\
--- Boundary Conditions ---
Node 0: Constraints [1, 1, 1, 0, 0, 0]
Node 1: Constraints [0, 1, 0, 1, 1, 0]
Node 2: Constraints [0, 0, 0, 0, 0, 0]"""

    # Ensure printed output matches expected summary
    assert captured.out.strip() == expected_output.strip(), "Printed boundary conditions summary does not match expected output"