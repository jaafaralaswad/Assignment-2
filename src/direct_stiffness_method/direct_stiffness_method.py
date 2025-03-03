import numpy as np
import math
import matplotlib as plt


class Structure:
    def __init__(self, nodes, elements, element_properties):
        self.nodes = nodes
        self.elements = elements
        self.element_properties = element_properties
        self.E = {elem_id: props["E"] for elem_id, props in element_properties.items()}
        self.nu = {elem_id: props["nu"] for elem_id, props in element_properties.items()}
    
    def compute_section_properties(self, elem_id):
        """Compute A, Iy, Iz, J for a given element based on its shape."""
        elem = self.element_properties[elem_id]

        if "b" in elem and "h" in elem:  # Rectangular section
            b = elem["b"]
            h = elem["h"]
            A = b * h
            Iy = (h * b**3) / 12
            Iz = (b * h**3) / 12
            J = Iy + Iz

        elif "r" in elem:  # Circular section
            r = elem["r"]
            A = math.pi * r**2
            Iy = Iz = (math.pi * r**4) / 4
            J = Iy + Iz

        else:
            raise ValueError("Invalid element properties. Define either (b, h) or (r).")

        return A, Iy, Iz, J
        
    def element_length(self, node1, node2):
        x1, y1, z1 = self.nodes[node1][1:]
        x2, y2, z2 = self.nodes[node2][1:]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    def display_summary(self):
        print("--- Structure Summary ---")
        print(f"Number of Elements: {len(self.elements)}")
        print("Elasticity Modulus (E):")
        for elem_id, E in self.E.items():
            print(f"  Element {elem_id}: {E}")
        
        print("\nPoisson's Ratio (nu):")
        for elem_id, nu in self.nu.items():
            print(f"  Element {elem_id}: {nu}")

        print("\n--- Element Properties ---")
        for i, (n1, n2) in enumerate(self.elements):
            length = self.element_length(n1, n2)
            A, Iy, Iz, J = self.compute_section_properties(i)

            print(f"Element {i + 1}:")
            print(f"  Length: {length:.4f}")
            print(f"  Area (A): {A:.4f}")
            print(f"  Moment of Inertia Iy: {Iy:.4f}")
            print(f"  Moment of Inertia Iz: {Iz:.4f}")
            print(f"  Polar Moment of Inertia J: {J:.4f}")
            print(f"  Node 1: {tuple(self.nodes[n1][1:])}, Node 2: {tuple(self.nodes[n2][1:])}\n")

        print("--- Connectivity Matrix ---")
        for i, (n1, n2) in enumerate(self.elements, 1):
            print(f"Element {i}: [{n1} {n2}]")
        
        print("\nGlobal Node Numbering:")
        for num, data in self.nodes.items():
            print(f"Global Node {num}: Coordinates {tuple(data[1:])}")

        print("\n* * * * * * * * * *")

    def local_elastic_stiffness_matrix_3D_beam(self, elem_id, L):
        """
        Compute the local element elastic stiffness matrix for a 3D beam.
        """
        k_e = np.zeros((12, 12))

        # Retrieve individual properties
        E = self.E[elem_id]
        nu = self.nu[elem_id]
        A, Iy, Iz, J = self.compute_section_properties(elem_id)

        # Axial terms - extension of local x axis
        axial_stiffness = E * A / L
        k_e[0, 0] = axial_stiffness
        k_e[0, 6] = -axial_stiffness
        k_e[6, 0] = -axial_stiffness
        k_e[6, 6] = axial_stiffness

        # Torsion terms - rotation about local x axis
        torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
        k_e[3, 3] = torsional_stiffness
        k_e[3, 9] = -torsional_stiffness
        k_e[9, 3] = -torsional_stiffness
        k_e[9, 9] = torsional_stiffness

        # Bending terms - bending about local z axis
        k_e[1, 1] = E * 12.0 * Iz / L ** 3.0
        k_e[1, 7] = E * -12.0 * Iz / L ** 3.0
        k_e[7, 1] = E * -12.0 * Iz / L ** 3.0
        k_e[7, 7] = E * 12.0 * Iz / L ** 3.0
        k_e[1, 5] = E * 6.0 * Iz / L ** 2.0
        k_e[5, 1] = E * 6.0 * Iz / L ** 2.0
        k_e[1, 11] = E * 6.0 * Iz / L ** 2.0
        k_e[11, 1] = E * 6.0 * Iz / L ** 2.0
        k_e[5, 7] = E * -6.0 * Iz / L ** 2.0
        k_e[7, 5] = E * -6.0 * Iz / L ** 2.0
        k_e[7, 11] = E * -6.0 * Iz / L ** 2.0
        k_e[11, 7] = E * -6.0 * Iz / L ** 2.0
        k_e[5, 5] = E * 4.0 * Iz / L
        k_e[11, 11] = E * 4.0 * Iz / L
        k_e[5, 11] = E * 2.0 * Iz / L
        k_e[11, 5] = E * 2.0 * Iz / L

        # Bending terms - bending about local y axis
        k_e[2, 2] = E * 12.0 * Iy / L ** 3.0
        k_e[2, 8] = E * -12.0 * Iy / L ** 3.0
        k_e[8, 2] = E * -12.0 * Iy / L ** 3.0
        k_e[8, 8] = E * 12.0 * Iy / L ** 3.0
        k_e[2, 4] = E * -6.0 * Iy / L ** 2.0
        k_e[4, 2] = E * -6.0 * Iy / L ** 2.0
        k_e[2, 10] = E * -6.0 * Iy / L ** 2.0
        k_e[10, 2] = E * -6.0 * Iy / L ** 2.0
        k_e[4, 8] = E * 6.0 * Iy / L ** 2.0
        k_e[8, 4] = E * 6.0 * Iy / L ** 2.0
        k_e[8, 10] = E * 6.0 * Iy / L ** 2.0
        k_e[10, 8] = E * 6.0 * Iy / L ** 2.0
        k_e[4, 4] = E * 4.0 * Iy / L
        k_e[10, 10] = E * 4.0 * Iy / L
        k_e[4, 10] = E * 2.0 * Iy / L
        k_e[10, 4] = E * 2.0 * Iy / L

        return k_e

    def compute_local_stiffness_matrices(self):
        """
        Computes and stores local stiffness matrices for all elements.
        """
        stiffness_matrices = {}
        for i, (n1, n2) in enumerate(self.elements):
            L = self.element_length(n1, n2)
            stiffness_matrices[i] = self.local_elastic_stiffness_matrix_3D_beam(i, L)
        return stiffness_matrices
    
    def compute_global_stiffness_matrices(self):
        """
        Computes the global stiffness matrices by mapping local stiffness matrices from local to global coordinates.
        The transformation is given by: k_global = Gamma^T * k_local * Gamma,
        where Gamma is the 12x12 transformation matrix derived from the 3x3 rotation matrix.
        """
        global_stiffness_matrices = {}
        local_stiffness_matrices = self.compute_local_stiffness_matrices()

        for i, (n1, n2) in enumerate(self.elements):
            # Obtain nodal coordinates
            x1, y1, z1 = self.nodes[n1][1:]
            x2, y2, z2 = self.nodes[n2][1:]

            # Compute the 3x3 rotation matrix
            gamma = rotation_matrix_3D(x1, y1, z1, x2, y2, z2)

            # Compute the 12x12 transformation matrix
            Gamma = transformation_matrix_3D(gamma)

            # Transform the local stiffness matrix to global coordinates
            k_local = local_stiffness_matrices[i]
            k_global = Gamma.T @ k_local @ Gamma
            global_stiffness_matrices[i] = k_global

        return global_stiffness_matrices


    def assemble_global_stiffness_matrix(self):
        """
        Assembles the overall global stiffness matrix from all element global stiffness matrices.
        """
        n_global_nodes = len(self.nodes)  # FIXED
        total_dofs = n_global_nodes * 6
        K_global_assembled = np.zeros((total_dofs, total_dofs))

        # Get the global stiffness matrices for each element
        global_stiffness_matrices = self.compute_global_stiffness_matrices()

        for elem_idx, (node1, node2) in enumerate(self.elements):  # FIXED
            # Determine the corresponding global DOF indices
            dofs = np.concatenate((
                np.arange(node1 * 6, node1 * 6 + 6),
                np.arange(node2 * 6, node2 * 6 + 6)
            ))
            # Add the element's contribution into the overall global stiffness matrix
            k_global = global_stiffness_matrices[elem_idx]
            for i_local in range(12):
                global_i = dofs[i_local]
                for j_local in range(12):
                    global_j = dofs[j_local]
                    K_global_assembled[global_i, global_j] += k_global[i_local, j_local]

        return K_global_assembled

    def print_global_stiffness_matrix(self):
        K_global_assembled = self.assemble_global_stiffness_matrix()
        print(K_global_assembled)


class BoundaryConditions:
    def __init__(self, loads, supports):
        """
        Initializes the boundary conditions with loads and supports.
        """
        self.loads = loads
        self.supports = supports
        self.n_nodes = len(loads)  # Assuming all nodes have loads/supports defined

    def compute_global_load_vector(self):
        """
        Constructs and returns the global load vector as a column vector.
        """
        total_dofs = self.n_nodes * 6
        F_global = np.zeros((total_dofs, 1))  # Column vector (total_dofs x 1)

        for node, values in self.loads.items():
            dof_index = node * 6  # Starting DOF index for the node
            F_global[dof_index:dof_index + 6, 0] = values[1:]  # Skip the node number
                
        return F_global
    
    def print_global_load_vector(self):
        """
        Prints the global load vector.
        """
        F_global = self.compute_global_load_vector()
        print("\n--- External Load Vector ---")
        print(F_global)

    def summarize_boundary_conditions(self):
        """
        Summarizes the boundary conditions by showing which DOFs are constrained (1) and free (0).
        """
        print("\n--- Boundary Conditions ---")
        for node, values in self.supports.items():
            print(f"Node {node}: Constraints {values[1:]}")


class Solver:
    def __init__(self, structure, boundary_conditions):
        """
        Initializes the solver with the structure and boundary conditions.
        """
        self.structure = structure
        self.boundary_conditions = boundary_conditions

    def get_constrained_dofs(self):
        """
        Extracts the constrained degrees of freedom (DOFs) from the support conditions.
        """
        constrained_dofs = []
        for node, constraints in self.boundary_conditions.supports.items():
            node_index = node * 6  # Each node has 6 DOFs
            for dof in range(6):
                if constraints[dof + 1] == 1:  # Skip the node index (first element)
                    constrained_dofs.append(node_index + dof)
        return constrained_dofs

    def solve(self):
        """
        Solves for the unknown displacements by reducing the global system and solving:
        U_reduced = K_reduced^-1 * F_reduced.
        """
        # Step 1: Get full system matrices
        K_global = self.structure.assemble_global_stiffness_matrix()
        F_global = self.boundary_conditions.compute_global_load_vector()

        # Step 2: Identify constrained DOFs
        constrained_dofs = self.get_constrained_dofs()
        all_dofs = np.arange(K_global.shape[0])
        free_dofs = np.setdiff1d(all_dofs, constrained_dofs)

        # Step 3: Extract the reduced system
        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
        F_reduced = F_global[free_dofs]

        # Step 4: Solve for unknown displacements
        U_reduced = np.linalg.solve(K_reduced, F_reduced)

        # Step 5: Reassemble full displacement vector
        U_global = np.zeros((K_global.shape[0], 1))
        U_global[free_dofs] = U_reduced

        print("\n--- Computed Displacements ---")
        print(U_global)

        return U_global

    def compute_reactions(self, U_global):
        """
        Computes reaction forces at constrained degrees of freedom.
        """
        K_global = self.structure.assemble_global_stiffness_matrix()
        F_global = self.boundary_conditions.compute_global_load_vector()
        constrained_dofs = self.get_constrained_dofs()

        # Compute reaction forces at constrained DOFs
        R_global = np.zeros((K_global.shape[0], 1))
        R_global[constrained_dofs] = K_global[np.ix_(constrained_dofs,)] @ U_global

        print("\n--- Computed Reactions ---")
        print(R_global)

        return R_global



class PostProcessing:
    def __init__(self, structure, U_global):
        self.structure = structure
        self.U_global = U_global
        self.internal_forces = {}

    def compute_internal_forces(self):
        """Compute internal forces for each element."""
        self.internal_forces = {} 

        for elem_id, (node_i, node_j) in enumerate(self.structure.elements):
            # Extract global displacements for the element
            Ue_global = np.hstack([
                self.U_global[node_i * 6:(node_i + 1) * 6].flatten(),
                self.U_global[node_j * 6:(node_j + 1) * 6].flatten()
            ])

            # Compute the transformation matrix
            x1, y1, z1 = self.structure.nodes[node_i][1:]
            x2, y2, z2 = self.structure.nodes[node_j][1:]
            gamma = rotation_matrix_3D(x1, y1, z1, x2, y2, z2)
            T = transformation_matrix_3D(gamma)

            # Transform to local coordinates
            Ue_local = T @ Ue_global

            # Retrieve local stiffness matrix
            L = self.structure.element_length(node_i, node_j)
            k_local = self.structure.local_elastic_stiffness_matrix_3D_beam(elem_id, L)

            # Compute internal forces in local coordinates
            self.internal_forces[elem_id] = k_local @ Ue_local

    def get_internal_forces(self):
        """Return the computed internal forces."""
        return self.internal_forces

    def print_internal_forces(self):
        """Print internal forces for each element."""
        print("\n--- Internal Forces in Local Coordinates ---")
        for elem_id, forces in self.internal_forces.items():
            print(f"Element {elem_id + 1}:")
            print(forces)








# Useful functions
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

def check_unit_vector(vec: np.ndarray):
    """
    """
    if np.isclose(np.linalg.norm(vec), 1.0):
        return
    else:
        raise ValueError("Expected a unit vector for reference vector.")


def check_parallel(vec_1: np.ndarray, vec_2: np.ndarray):
    """
    """
    if np.isclose(np.linalg.norm(np.cross(vec_1, vec_2)), 0.0):
        raise ValueError("Reference vector is parallel to beam axis.")
    else:
        return
