import numpy as np
import math
import pytest

# Import functions
from direct_stiffness_method.direct_stiffness_method import Structure, BoundaryConditions, Solver, PostProcessing, BucklingAnalysis, PlotResults

def test_rectangular_section():
    nodes = {}
    elements = {}
    element_properties = {
        1: {"E": 200e9, "nu": 0.3, "b": 0.2, "h": 0.4}
    }
    
    structure = Structure(nodes, elements, element_properties)
    A, Iy, Iz, J = structure.compute_section_properties(1)
    
    assert A == pytest.approx(0.08)  # 0.2 * 0.4
    assert Iy == pytest.approx((0.4 * 0.2**3) / 12)
    assert Iz == pytest.approx((0.2 * 0.4**3) / 12)
    assert J == pytest.approx(Iy + Iz)

def test_circular_section():
    nodes = {}
    elements = {}
    element_properties = {
        2: {"E": 200e9, "nu": 0.3, "r": 0.1}
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
        3: {"E": 200e9, "nu": 0.3}  # Missing b, h, or r
    }
    
    structure = Structure(nodes, elements, element_properties)
    
    with pytest.raises(ValueError, match="Invalid element properties"):
        structure.compute_section_properties(3)