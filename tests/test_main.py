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
    nodes = {
        1: (1, 0.0, 0.0, 0.0),
        2: (2, 3.0, 4.0, 0.0),
        3: (3, 1.0, 1.0, 1.0),
        4: (4, 4.0, 5.0, 6.0)
    }

    elements = [(1, 2), (3, 4)]  # Two elements

    element_properties = {
        1: {"E": 200e9, "nu": 0.3, "b": 0.2, "h": 0.4},  # Updated to match element IDs
        2: {"E": 150e9, "nu": 0.25, "r": 0.1}
    }

    structure = Structure(nodes, elements, element_properties)
    structure.display_summary()

    # Capture printed output
    captured = capfd.readouterr()
    output = captured.out

    # Verify the number of elements
    assert "Number of Elements: 2" in output

    # Check material properties
    assert "Element 1: 200000000000.0" in output  # E for element 1
    assert "Element 2: 150000000000.0" in output  # E for element 2
    assert "Element 1: 0.3" in output  # Poisson's ratio for element 1
    assert "Element 2: 0.25" in output  # Poisson's ratio for element 2

    # Verify element lengths
    expected_length_1 = 5.0  # From (0,0,0) to (3,4,0)
    expected_length_2 = np.sqrt((4 - 1) ** 2 + (5 - 1) ** 2 + (6 - 1) ** 2)

    assert f"Length: {expected_length_1:.4f}" in output
    assert f"Length: {expected_length_2:.4f}" in output

    # Check section properties (rectangular and circular)
    A_rect = 0.2 * 0.4
    Iy_rect = (0.4 * 0.2**3) / 12
    Iz_rect = (0.2 * 0.4**3) / 12
    J_rect = Iy_rect + Iz_rect

    A_circ = math.pi * 0.1**2
    Iy_circ = (math.pi * 0.1**4) / 4
    Iz_circ = (math.pi * 0.1**4) / 4
    J_circ = Iy_circ + Iz_circ

    assert f"Area (A): {A_rect:.4f}" in output
    assert f"Moment of Inertia Iy: {Iy_rect:.4f}" in output
    assert f"Moment of Inertia Iz: {Iz_rect:.4f}" in output
    assert f"Polar Moment of Inertia J: {J_rect:.4f}" in output

    assert f"Area (A): {A_circ:.4f}" in output
    assert f"Moment of Inertia Iy: {Iy_circ:.4f}" in output
    assert f"Moment of Inertia Iz: {Iz_circ:.4f}" in output
    assert f"Polar Moment of Inertia J: {J_circ:.4f}" in output

    # Check node numbering
    assert "Global Node 1: Coordinates (0.0, 0.0, 0.0)" in output
    assert "Global Node 2: Coordinates (3.0, 4.0, 0.0)" in output
    assert "Global Node 3: Coordinates (1.0, 1.0, 1.0)" in output
    assert "Global Node 4: Coordinates (4.0, 5.0, 6.0)" in output

    # Check connectivity matrix
    assert "Element 1: [1 2]" in output
    assert "Element 2: [3 4]" in output
