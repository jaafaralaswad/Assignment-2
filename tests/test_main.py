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