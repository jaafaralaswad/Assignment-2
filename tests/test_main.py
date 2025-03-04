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
