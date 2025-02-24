import numpy as np
import math

class Structure:
    def __init__(self):
        self.nel = int(input("Enter Number of Elements: "))
        self.E = float(input("Enter Elasticity Modulus (E): "))
        self.nu = float(input("Enter Poisson's Ratio (nu): "))
        self.elements = [Element(i) for i in range(self.nel)]

        # Compute local stiffness matrices and store them
        self.local_stiffness_matrices = self.compute_local_stiffness_matrices()
        # Construct connectivity matrix and number nodes
        self.connectivity_matrix = self.construct_connectivity_matrix()
        
    def display_structure(self):
        print("\n--- Structure Summary ---")
        print(f"Number of Elements: {self.nel}")
        print(f"Elasticity Modulus (E): {self.E}")
        print(f"Poisson's Ratio (nu): {self.nu}\n")
        for i, element in enumerate(self.elements):
            print(f"Element {i + 1}:")
            print(element, "\n")
            
    def display_connectivity_matrix(self):
        print("\n--- Connectivity Matrix ---")
        for i, row in enumerate(self.connectivity_matrix):
            print(f"Element {i + 1}: {row}")
        print("\nGlobal Node Numbering:")
        for node, number in self.node_numbers.items():
            print(f"Global Node {number}: Coordinates {node}")

    def compute_local_stiffness_matrices(self):
        """
        Computes and stores local stiffness matrices for all elements.
        """
        stiffness_matrices = {}
        for i, element in enumerate(self.elements):
            stiffness_matrices[i] = self.local_elastic_stiffness_matrix_3D_beam(
                self.E, self.nu, element.A, element.L, element.Iy, element.Iz, element.J
            )
        return stiffness_matrices

    def local_elastic_stiffness_matrix_3D_beam(self, E, nu, A, L, Iy, Iz, J):
        """
        Compute the local element elastic stiffness matrix for a 3D beam.
        """
        k_e = np.zeros((12, 12))

        # Axial stiffness
        axial_stiffness = E * A / L
        k_e[0, 0] = axial_stiffness
        k_e[0, 6] = -axial_stiffness
        k_e[6, 0] = -axial_stiffness
        k_e[6, 6] = axial_stiffness

        # Torsional stiffness
        torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
        k_e[3, 3] = torsional_stiffness
        k_e[3, 9] = -torsional_stiffness
        k_e[9, 3] = -torsional_stiffness
        k_e[9, 9] = torsional_stiffness

        # Bending stiffness - about local z-axis
        k_e[1, 1] = E * 12.0 * Iz / L ** 3.0
        k_e[1, 7] = -E * 12.0 * Iz / L ** 3.0
        k_e[7, 1] = -E * 12.0 * Iz / L ** 3.0
        k_e[7, 7] = E * 12.0 * Iz / L ** 3.0
        k_e[1, 5] = E * 6.0 * Iz / L ** 2.0
        k_e[5, 1] = E * 6.0 * Iz / L ** 2.0
        k_e[1, 11] = E * 6.0 * Iz / L ** 2.0
        k_e[11, 1] = E * 6.0 * Iz / L ** 2.0

        # Bending stiffness - about local y-axis
        k_e[2, 2] = E * 12.0 * Iy / L ** 3.0
        k_e[2, 8] = -E * 12.0 * Iy / L ** 3.0
        k_e[8, 2] = -E * 12.0 * Iy / L ** 3.0
        k_e[8, 8] = E * 12.0 * Iy / L ** 3.0
        k_e[2, 4] = -E * 6.0 * Iy / L ** 2.0
        k_e[4, 2] = -E * 6.0 * Iy / L ** 2.0
        k_e[2, 10] = -E * 6.0 * Iy / L ** 2.0
        k_e[10, 2] = -E * 6.0 * Iy / L ** 2.0

        return k_e
    
    def construct_connectivity_matrix(self):
        """
        Constructs the connectivity matrix.
        
        This function numbers the nodes based on their coordinates and constructs a matrix
        with shape (nElements, 2) where each row contains the two global node numbers for the element.
        """
        node_numbers = {}  # Maps node coordinate (tuple) to a global node number
        global_node_counter = 0
        connectivity_matrix = np.zeros((self.nel, 2), dtype=int)

        for i, element in enumerate(self.elements):
            # Process node 1
            if element.node1 not in node_numbers:
                node_numbers[element.node1] = global_node_counter
                global_node_counter += 1
            connectivity_matrix[i, 0] = node_numbers[element.node1]
            
            # Process node 2
            if element.node2 not in node_numbers:
                node_numbers[element.node2] = global_node_counter
                global_node_counter += 1
            connectivity_matrix[i, 1] = node_numbers[element.node2]
        
        # Store the node numbering dictionary for future reference.
        self.node_numbers = node_numbers
        
        return connectivity_matrix

    def compute_global_stiffness_matrices(self):
        """
        Computes the global stiffness matrices by mapping local stiffness matrices from local to global coordinates.
        The transformation is given by: k_global = Gamma^T * k_local * Gamma,
        where Gamma is the 12x12 transformation matrix derived from the 3x3 rotation matrix.
        """
        global_stiffness_matrices = {}
        for i, element in enumerate(self.elements):
            # Obtain the rotation matrix for the element based on its nodal coordinates
            gamma = rotation_matrix_3D(
                element.node1[0], element.node1[1], element.node1[2],
                element.node2[0], element.node2[1], element.node2[2]
            )
            # Build the transformation matrix Gamma (12x12)
            Gamma = transformation_matrix_3D(gamma)
            # Transform the local stiffness matrix to global coordinates:
            # Note: each node has 6 DOFs; for example, global node 0 has DOFs 0 to 5.
            k_local = self.local_stiffness_matrices[i]
            k_global = Gamma.T @ k_local @ Gamma
            global_stiffness_matrices[i] = k_global
        return global_stiffness_matrices


    def assemble_global_stiffness_matrix(self):
        """
        Assembles the overall global stiffness matrix from all element global stiffness matrices.
        Each node has 6 DOFs (e.g., node 0 → DOFs 0-5, node 1 → DOFs 6-11, etc.).
        """
        n_global_nodes = len(self.node_numbers)
        total_dofs = n_global_nodes * 6
        K_global_assembled = np.zeros((total_dofs, total_dofs))
        
        # Get the global stiffness matrices for each element
        global_stiffness_matrices = self.compute_global_stiffness_matrices()
        
        for elem_idx, element in enumerate(self.elements):
            # Retrieve the two global node numbers for the element
            node1, node2 = self.connectivity_matrix[elem_idx]
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


class Element:
    def __init__(self, index):
        print(f"\n--- Enter Section Properties for Element {index + 1} ---")
        self.A = float(input("Enter Area (A): "))
        self.Iy = float(input("Enter 2nd Moment of Area about Local y-axis (Iy): "))
        self.Iz = float(input("Enter 2nd Moment of Area about Local z-axis (Iz): "))
        self.J = float(input("Enter Polar 2nd Moment of Area about Local x-axis (J): "))

        print(f"\n--- Enter coordinates for nodes of Element {index + 1} ---")
        self.node1 = tuple(map(float, input("Enter (x, y, z) Coordinates for Node 1 (comma-separated, no parentheses): ").split(',')))
        self.node2 = tuple(map(float, input("Enter (x, y, z) Coordinates for Node 2 (comma-separated, no parentheses): ").split(',')))

        # Calculate the element length based on node coordinates
        self.L = self.calculate_length()

    def calculate_length(self):
        x1, y1, z1 = self.node1
        x2, y2, z2 = self.node2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def __str__(self):
        return (f"Element Properties:\n"
                f"Area: {self.A}, Length: {self.L:.4f}\n"
                f"Iy: {self.Iy}, Iz: {self.Iz}, J: {self.J}\n"
                f"Node 1: {self.node1}, Node 2: {self.node2}")


class BoundaryConditions:
    def __init__(self):
        # Loads will map global node numbers to a list of load values [Fx, Fy, Fz, Mx, My, Mz]
        self.loads = {}
        # Supports will map global node numbers to a list of restricted degrees of freedom
        self.supports = {}

    def prescribe_loads(self):
        num_loaded = int(input("Enter the number of global nodes that are loaded: "))
        for i in range(num_loaded):
            node = int(input(f"Enter global node number for load {i+1}: "))
            load_input = input("Enter loads (Fx, Fy, Fz, Mx, My, Mz) as comma-separated values: ")
            loads = list(map(float, load_input.split(',')))
            if len(loads) != 6:
                print("Error: Please enter exactly 6 load values.")
            else:
                self.loads[node] = loads

    def prescribe_supports(self):
        num_supported = int(input("Enter the number of global nodes that are supported: "))
        for i in range(num_supported):
            node = int(input(f"Enter global node number for support {i+1}: "))
            dofs_input = input("Enter restricted degrees of freedom (e.g., ux, uy, uz, theta_x, theta_y, theta_z) as comma-separated values: ")
            dofs = [dof.strip() for dof in dofs_input.split(',')]
            self.supports[node] = dofs

    def display_boundary_conditions(self):
        print("\n--- Prescribed Loads ---")
        for node, loads in self.loads.items():
            print(f"Global Node {node}: Loads {loads}")
        print("\n--- Prescribed Supports ---")
        for node, dofs in self.supports.items():
            print(f"Global Node {node}: Restricted DOFs {dofs}")

    def create_global_force_vector(self, structure_obj=None) -> np.ndarray:
        """
        Creates a global force vector (column vector) with 6*n_global_nodes rows.
        """
        if structure_obj is not None and hasattr(structure_obj, "node_numbers"):
            n_global_nodes = len(structure_obj.node_numbers)
        else:
            n_global_nodes = int(input("Enter the total number of global nodes: "))
            
        F_global = np.zeros((n_global_nodes * 6, 1))  # Column vector
        for node, loads in self.loads.items():
            start_idx = node * 6
            F_global[start_idx:start_idx + 6, 0] = np.array(loads)  # Remove reshape
        return F_global




class Solver:
    def __init__(self, structure, boundary_conditions):
        self.structure = structure
        self.boundary_conditions = boundary_conditions
        self.K_global = structure.assemble_global_stiffness_matrix()
        self.F_global = boundary_conditions.create_global_force_vector(structure)
        self.free_dofs, self.constrained_dofs = self.get_free_and_constrained_dofs()
        self.K_reduced, self.F_reduced = self.apply_geometric_boundary_conditions()

    def get_free_and_constrained_dofs(self):
        """
        Identifies constrained (0) and free (nonzero) DOFs.
        """
        total_dofs = self.F_global.shape[0]
        constrained_dofs = set()

        for node, dofs in self.boundary_conditions.supports.items():
            node_dof_start = node * 6  # Each node has 6 DOFs
            for i, dof_value in enumerate(dofs):
                if str(dof_value).strip() == "0":  # Fix: Convert input to string for proper comparison
                    constrained_dofs.add(node_dof_start + i)

        free_dofs = [i for i in range(total_dofs) if i not in constrained_dofs]

        return free_dofs, list(constrained_dofs)

    def apply_geometric_boundary_conditions(self):
        """
        Removes constrained DOFs (rows & columns from stiffness, rows from force vector).
        """
        if not self.free_dofs:
            raise ValueError("No free DOFs available. The structure may be overconstrained.")

        # Remove rows & columns corresponding to constrained DOFs
        K_reduced = self.K_global[np.ix_(self.free_dofs, self.free_dofs)]
        F_reduced = self.F_global[self.free_dofs].reshape(-1, 1)  # Ensure correct shape

        # Debugging output
        print("\n--- Checking Reduced Stiffness Matrix (K_reduced) ---")
        print(f"Shape: {K_reduced.shape} (should match number of free DOFs)")
        print(f"Determinant: {np.linalg.det(K_reduced):.4e}")

        return K_reduced, F_reduced

    def solve_displacements(self):
        """
        Solves for unknown displacements by inverting the reduced global stiffness matrix.
        """
        if np.linalg.matrix_rank(self.K_reduced) < self.K_reduced.shape[0]:
            raise np.linalg.LinAlgError("Reduced stiffness matrix is singular. Check constraints.")

        U_reduced = np.linalg.solve(self.K_reduced, self.F_reduced)
        return U_reduced

    def compute_reactions(self, U_reduced):
        """
        Computes reaction forces at constrained DOFs.
        """
        U_full = np.zeros((len(self.F_global), 1))
        U_full[self.free_dofs] = U_reduced.reshape(-1, 1)
        F_computed = self.K_global @ U_full
        return U_full, F_computed

    def display_results(self, U_full, F_computed):
        """
        Displays computed displacements and reaction forces per global node.
        """
        print("\n--- Displacements per Global Node ---")
        for node, number in self.structure.node_numbers.items():
            start_idx = number * 6
            print(f"Global Node {number}:")
            print(f"  ux = {U_full[start_idx, 0]:.6e}, uy = {U_full[start_idx + 1, 0]:.6e}, uz = {U_full[start_idx + 2, 0]:.6e}")
            print(f"  theta_x = {U_full[start_idx + 3, 0]:.6e}, theta_y = {U_full[start_idx + 4, 0]:.6e}, theta_z = {U_full[start_idx + 5, 0]:.6e}")

        print("\n--- Reaction Forces per Global Node ---")
        for node, number in self.structure.node_numbers.items():
            start_idx = number * 6
            print(f"Global Node {number}:")
            print(f"  Fx = {F_computed[start_idx, 0]:.6e}, Fy = {F_computed[start_idx + 1, 0]:.6e}, Fz = {F_computed[start_idx + 2, 0]:.6e}")
            print(f"  Mx = {F_computed[start_idx + 3, 0]:.6e}, My = {F_computed[start_idx + 4, 0]:.6e}, Mz = {F_computed[start_idx + 5, 0]:.6e}")















def check_unit_vector(vec: np.ndarray):
    """
    Checks if the provided vector is a unit vector.
    """
    if np.isclose(np.linalg.norm(vec), 1.0):
        return
    else:
        raise ValueError("Expected a unit vector for reference vector.")


def check_parallel(vec_1: np.ndarray, vec_2: np.ndarray):
    """
    Checks if two vectors are parallel.
    """
    if np.isclose(np.linalg.norm(np.cross(vec_1, vec_2)), 0.0):
        raise ValueError("Reference vector is parallel to beam axis.")
    else:
        return


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
