import numpy as np
import math

class Structure:
    def __init__(self, nodes, elements, E, nu, A, Iy, Iz, J):
        self.nodes = nodes
        self.elements = elements
        self.E = E
        self.nu = nu
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
    
    def element_length(self, node1, node2):
        x1, y1, z1 = self.nodes[node1][1:]
        x2, y2, z2 = self.nodes[node2][1:]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    def display_summary(self):
        print("--- Structure Summary ---")
        print(f"Number of Elements: {len(self.elements)}")
        print(f"Elasticity Modulus (E): {self.E}")
        print(f"Poisson's Ratio (nu): {self.nu}\n")
        
        for i, (n1, n2) in enumerate(self.elements, 1):
            length = self.element_length(n1, n2)
            print(f"Element {i}:")
            print("Element Properties:")
            print(f"Area: {self.A:.4f}, Length: {length:.4f}")
            print(f"Iy: {self.Iy:.4f}, Iz: {self.Iz:.4f}, J: {self.J:.4f}")
            print(f"Node 1: {tuple(self.nodes[n1][1:])}, Node 2: {tuple(self.nodes[n2][1:])}\n")
        
        print("--- Connectivity Matrix ---")
        for i, (n1, n2) in enumerate(self.elements, 1):
            print(f"Element {i}: [{n1} {n2}]")
        
        print("\nGlobal Node Numbering:")
        for num, data in self.nodes.items():
            print(f"Global Node {num}: Coordinates {tuple(data[1:])}")
        print("\n* * * * * * * * * *")

    def local_elastic_stiffness_matrix_3D_beam(self, L):
        """
        Compute the local element elastic stiffness matrix for a 3D beam.
        """
        k_e = np.zeros((12, 12))
        
        # Axial terms - extension of local x axis
        axial_stiffness = self.E * self.A / L
        k_e[0, 0] = axial_stiffness
        k_e[0, 6] = -axial_stiffness
        k_e[6, 0] = -axial_stiffness
        k_e[6, 6] = axial_stiffness
        # Torsion terms - rotation about local x axis
        torsional_stiffness = self.E * self.J / (2.0 * (1 + self.nu) * L)
        k_e[3, 3] = torsional_stiffness
        k_e[3, 9] = -torsional_stiffness
        k_e[9, 3] = -torsional_stiffness
        k_e[9, 9] = torsional_stiffness
        # Bending terms - bending about local z axis
        k_e[1, 1] = self.E * 12.0 * self.Iz / L ** 3.0
        k_e[1, 7] = self.E * -12.0 * self.Iz / L ** 3.0
        k_e[7, 1] = self.E * -12.0 * self.Iz / L ** 3.0
        k_e[7, 7] = self.E * 12.0 * self.Iz / L ** 3.0
        k_e[1, 5] = self.E * 6.0 * self.Iz / L ** 2.0
        k_e[5, 1] = self.E * 6.0 * self.Iz / L ** 2.0
        k_e[1, 11] = self.E * 6.0 * self.Iz / L ** 2.0
        k_e[11, 1] = self.E * 6.0 * self.Iz / L ** 2.0
        k_e[5, 7] = self.E * -6.0 * self.Iz / L ** 2.0
        k_e[7, 5] = self.E * -6.0 * self.Iz / L ** 2.0
        k_e[7, 11] = self.E * -6.0 * self.Iz / L ** 2.0
        k_e[11, 7] = self.E * -6.0 * self.Iz / L ** 2.0
        k_e[5, 5] = self.E * 4.0 * self.Iz / L
        k_e[11, 11] = self.E * 4.0 * self.Iz / L
        k_e[5, 11] = self.E * 2.0 * self.Iz / L
        k_e[11, 5] = self.E * 2.0 * self.Iz / L
        # Bending terms - bending about local y axis
        k_e[2, 2] = self.E * 12.0 * self.Iy / L ** 3.0
        k_e[2, 8] = self.E * -12.0 * self.Iy / L ** 3.0
        k_e[8, 2] = self.E * -12.0 * self.Iy / L ** 3.0
        k_e[8, 8] = self.E * 12.0 * self.Iy / L ** 3.0
        k_e[2, 4] = self.E * -6.0 * self.Iy / L ** 2.0
        k_e[4, 2] = self.E * -6.0 * self.Iy / L ** 2.0
        k_e[2, 10] = self.E * -6.0 * self.Iy / L ** 2.0
        k_e[10, 2] = self.E * -6.0 * self.Iy / L ** 2.0
        k_e[4, 8] = self.E * 6.0 * self.Iy / L ** 2.0
        k_e[8, 4] = self.E * 6.0 * self.Iy / L ** 2.0
        k_e[8, 10] = self.E * 6.0 * self.Iy / L ** 2.0
        k_e[10, 8] = self.E * 6.0 * self.Iy / L ** 2.0
        k_e[4, 4] = self.E * 4.0 * self.Iy / L
        k_e[10, 10] = self.E * 4.0 * self.Iy / L
        k_e[4, 10] = self.E * 2.0 * self.Iy / L
        k_e[10, 4] = self.E * 2.0 * self.Iy / L

        return k_e

    def compute_local_stiffness_matrices(self):
        """
        Computes and stores local stiffness matrices for all elements.
        """
        stiffness_matrices = {}
        for i, (n1, n2) in enumerate(self.elements):
            L = self.element_length(n1, n2)
            stiffness_matrices[i] = self.local_elastic_stiffness_matrix_3D_beam(L)
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
        
        print("\n--- External Load Vector ---")
        print(F_global)
        
        return F_global

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
        
        Parameters:
        - structure: An instance of the Structure class (provides global stiffness matrix).
        - boundary_conditions: An instance of the BoundaryConditions class (provides loads and supports).
        """
        self.structure = structure
        self.boundary_conditions = boundary_conditions

    def get_constrained_dofs(self):
        """
        Extracts the constrained degrees of freedom (DOFs) from the support conditions.
        
        Returns:
        - constrained_dofs (list): Indices of constrained degrees of freedom.
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
        K_global = self.structure.assemble_global_stiffness_matrix()  # Global stiffness matrix
        F_global = self.boundary_conditions.compute_global_load_vector()  # Global force vector (column)

        # Step 2: Identify constrained DOFs
        constrained_dofs = self.get_constrained_dofs()
        all_dofs = np.arange(K_global.shape[0])  # All DOFs
        free_dofs = np.setdiff1d(all_dofs, constrained_dofs)  # Free DOFs

        # Step 3: Extract the reduced system
        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]  # Reduced stiffness matrix
        F_reduced = F_global[free_dofs]  # Reduced force vector

        # Step 4: Solve for unknown displacements
        U_reduced = np.linalg.solve(K_reduced, F_reduced)

        # Step 5: Reassemble full displacement vector
        U_global = np.zeros((K_global.shape[0], 1))  # Full displacement vector (column)
        U_global[free_dofs] = U_reduced  # Insert solved values (constrained DOFs remain zero)

        # Output results
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
