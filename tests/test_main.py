import numpy as np
import pytest

# Import functions
from direct_stiffness_method.direct_stiffness_method import Structure, BoundaryConditions, Solver, PostProcessing, BucklingAnalysis, PlotResults

def test_compute_section_properties_rectangular_0():
    element_properties = {0: {"b": 0.5, "h": 1.0, "E": 1000, "nu": 0.3}}
    structure = Structure(element_properties)  # Create an instance
    A, Iy, Iz, J = structure.compute_section_properties(0)
    assert np.isclose(A, 0.5)
    assert np.isclose(Iy, 0.01041667)
    assert np.isclose(Iz, 0.04166667)
    assert np.isclose(J, 0.05208333)

def test_compute_section_properties_rectangular_1():
    element_properties = {1: {"b": 1.0, "h": 0.5, "E": 1000, "nu": 0.3}}
    structure = Structure(element_properties)
    A, Iy, Iz, J = structure.compute_section_properties(1)
    assert np.isclose(A, 0.5)
    assert np.isclose(Iy, 0.04166667)
    assert np.isclose(Iz, 0.01041667)
    assert np.isclose(J, 0.05208333)

def test_compute_section_properties_circular():
    element_properties = {0: {"r": 1.0, "E": 500, "nu": 0.3}}
    structure = Structure(element_properties)
    A, Iy, Iz, J = structure.compute_section_properties(0)
    assert np.isclose(A, 3.14159265)
    assert np.isclose(Iy, 0.78539816)
    assert np.isclose(Iz, 0.78539816)
    assert np.isclose(J, 1.57079633)
