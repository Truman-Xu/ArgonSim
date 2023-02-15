# Brooks Lab at the University of Michigan, 2023

"""Argon MD simulation engine based on Verlet algorithm implemented with cupy"""
from itertools import combinations, permutations
import cupy as cp
from cupy import linalg as LA
from cupyx.scipy import ndimage as NI

class ArgonSim:
    """Simple molecular dynamics simulation engine for Argon systems. This is a
    reimplementation of the Verlet paper on Phys Review, 1976 except that this
    simulator does not use the reduced unit as Verlet did."""
    n = 6.0221415e23 # /mol
    m_argon = 39.948/n # gram
    m_argon_kg = m_argon*1e-3 # mass of one Argon atom in kg
    kb = 1.3806503e-23 # J/K
    # characteristic length for lennard jones
    sigma = 3.405e-10 # m
    def __init__(
        self,
        n_cells : int,
        temp : float = 120,
        dt : float = 1e-14, # seconds
        density : float = 1.374, # g/cm^3
        lennard_jones_cutoff : float = 2.5 # 2.5 sigma is set as lennard jones potential cutoff
    ):
        """
        Initialize the system by specifying the number of face-centered cubic 
        unit cells, temperature, and time step. The coordinates and the system 
        size for the Argon atoms will be calculated from its density in 130K, 
        and the atoms are arranged as FCC nodes. 
        The total number of atoms will be 4*N^3, where N is the number of 
        the unit cells per dimension.
            n_cells
            param : Number of unit cells per dimension. The entire system size 
                    will be N^3 unit cells
            type : int

            temp
            param : Temperature of the simulation, unit in Kelvin. Default to 
                    120K
            type : float

            dt
            param : Time step ∆t for the simulation, unit in second. Default 
                    value is set to 1 X 10^-14 s.
            type : float

            density
            param : Density of the Argon system, unit in g/cm^3. Default to 
                    1.374 g/cm^3 (@130K)
            type : float
        """
        if not isinstance(n_cells, int):
            raise TypeError(
                "Integer value is required to specify the number of unit cells"
            )
        self.dt = dt # second
        n_atoms = 4*n_cells**3
        L = (self.m_argon*n_atoms/density)**(1/3) # centimeter
        self.lennard_jones_cutoff = lennard_jones_cutoff * self.sigma
        self.box_len = L * 1e-2 # system dimension in meters
        self.lj_coef = 4*119.8*self.kb # Lennard Jones energy coeff
        self.del_pot_coef = 12*self.lj_coef # potential gradient coeff
        self.init_coords = self._init_coords(self.box_len, n_cells) # meters
        self._idx_array = cp.arange(self.init_coords.shape[0])
        self.velo = self._init_velocities(temp, self.init_coords.shape)
        # combination of indices (i, j) for atom i and atom j
        self.id_pairs = cp.array(
            list(combinations(range(len(self.init_coords)),2))
        )
        self.coords = cp.empty_like(self.init_coords, dtype=cp.float64)
        self.r_ij = None
        self.cutoff_ids = None
        self.pair_potentials = None
        self.last_coords = None
        self.accl = None
        self._init_step()
        
    def _init_coords(self, box_len, n_cells) -> cp.ndarray:
        """
        Initialize coordinates in face centered cubic unit cells
        """
        cell_len = box_len/n_cells
        x_corner = cp.linspace(0, box_len-cell_len, n_cells) # corner atom spacings
        x_face = cp.linspace(cell_len/2, box_len-cell_len/2, n_cells) # face atom spacings

        grid_c = cp.meshgrid(x_corner, x_corner, x_corner, indexing='ij')
        grid_f = cp.meshgrid(x_face, x_corner, x_face, indexing='ij')
        # get all 3 face nodes by combinations of grid_f
        grid_3f = cp.array(tuple(permutations(grid_f))[:3]).reshape((3,3,-1))
        face_coords = cp.concatenate(grid_3f, axis = 1)
        corner_coords = cp.array(grid_c).reshape((3,-1))
        # N x 3 coords array
        coords = cp.concatenate([face_coords, corner_coords], axis = 1).T
        return coords
    
    @staticmethod
    def _init_velocities(temp, size) -> cp.ndarray:
        """Initialize velocities for each atom from a Maxwell-Boltzmann 
        distribution"""
        kbT_over_m = 1.3806503*temp/(39.948e-3*6.0221415**(-1)) # m^2/s^2
        velo_std = (kbT_over_m)**0.5 # m/s
        init_velo = cp.random.normal(0, velo_std, size)
        return init_velo
    
    @staticmethod
    def _apply_pbc_dist(dist_vecs, box_len) -> None:
        """Apply periodic boundary conditions on distance vectors. Shortest 
        distance vector will be determined from self.box_length and assigned 
        to the distance array directly"""
        mask = cp.abs(dist_vecs) > box_len/2
        dist_vecs[mask] -= cp.sign(dist_vecs[mask])*box_len

    @staticmethod
    def _apply_pbc_coord(coords, box_len) -> None:
        """Apply periodic boundary conditions on coordinates. Any out of bound 
        coord will be wrapped and place to the other side of the box. The new 
        coords will be assigned to the self.coords array directly"""
        mask1 = coords > box_len
        mask2 = coords < 0
        wrapped_coords1 = coords[mask1] % box_len
        wrapped_coords2 = box_len - ((-coords[mask2]) % box_len)
        coords[mask1] = wrapped_coords1
        coords[mask2] = wrapped_coords2

    @staticmethod
    def verlet(curr_x, last_x, accl, dt) -> cp.ndarray:
        """Verlet algorithm"""
        return 2*curr_x-last_x+accl*dt**2

    @staticmethod
    def lennard_jones(lj_coeff, sigma, r) -> cp.ndarray:
        ''' 
        Lennard-Jones potentials
        v_LJ(r) = 4*epsilon[(sigma/r)^12-(sigma/r)^6]
        units in J
        epsilon/k_b = 120K => epsilon: J (m^2 kg s^-2)
        '''
        return lj_coeff*((sigma/r)**12-(sigma/r)**6)

    @staticmethod
    def del_potential(dp_coef, r, dist_vecs):
        """Gradient operator on Lennard Jones potentials to find force vectors
        """
        return dp_coef*dist_vecs*r**(-2)*((3.405e-10/r)**12-0.5*(3.405e-10/r)**6)

    def _find_cutoff_ids(self) -> cp.ndarray:
        cutoff_mask = self.r_ij <= self.lennard_jones_cutoff
        return cp.nonzero(cutoff_mask)[0]

    def _compute_pair_potentials(self) -> cp.ndarray:
        """Compute the potentials from the cutoff list only. Return pairwise
        potentials of atoms"""
        pair_potentials = self.lennard_jones(
            self.lj_coef,
            self.sigma,
            self.r_ij[self.cutoff_ids]
        )
        return pair_potentials

    @staticmethod
    def _sum_pair_forces_1d(pair_forces_1d, pair_labels, idx_array) -> cp.ndarray:
        """
        Sum pairwise forces (i,j pairs) according the label ids. The directions
        of the pairwise forces are taken into account.
        """
        # force j to i
        f_in = NI.sum_labels(pair_forces_1d, pair_labels[:,0], idx_array)
        # force i to j
        f_out = NI.sum_labels(pair_forces_1d, pair_labels[:,1], idx_array)
        f_total = f_in - f_out
        return f_total

    @staticmethod
    def _sum_pair_forces(pair_forces, pair_labels, coords_shape) -> cp.ndarray:
        """
        Sum pairwise forces (i,j pairs) according the label ids. The directions
        of the pairwise forces are taken into account.
        """
        
        f_total = cp.zeros(coords_shape)
        for i in range(f_total.shape[0]):
            # force j to i
            f_total[i] += pair_forces[pair_labels[:,0] == i].sum(0)
            # force i to j
            f_total[i] -= pair_forces[pair_labels[:,1] == i].sum(0)
        return f_total

    def _get_atom_forces_ni_sum(self, pair_forces, pair_labels):
        # calculate pairwise force vectors (fx,fy,fz) from atom j to atom i
        fx, fy, fz = pair_forces.T
        # force vectors for each atom
        f_atoms = cp.empty(self.coords.shape)
        for i, f1d in enumerate((fx, fy, fz)):
            # sum all pairwise forces by each dimension on each atom
            f_atoms[:,i] = self._sum_pair_forces_1d(
                f1d, pair_labels, self._idx_array
            )
        return f_atoms

    def get_accelerations(self, coords) -> cp.ndarray:
        """
        Get accelerations from a set of coordinates. The pairwaise distance 
        vectors and distances (vector norm) will be determined first. The 
        atom pairs that are within distance cutoff will be selected, and the 
        Lennard Jones potentials, the force vectors, and the accelerations on 
        each atom will be calculated. Only acceleration array is returned

        Args
            coords
            param : a set of coordinates of the Argon atoms at a given time step
            type: cp.array
            
        Return
            acclerations
            param : acceleration vectors for each atom with shape (N, 3)
            type : cp.array
        """
        id_pairs = self.id_pairs
        # dist vector from atom j to atom i
        dist_vecs = coords[id_pairs[:,0]] - coords[id_pairs[:,1]]
        # apply periodic boundary conditions on distances
        self._apply_pbc_dist(dist_vecs, self.box_len)
        # update the pairwise euclidean distances from atom j to atom i
        self.r_ij = LA.norm(dist_vecs, axis = 1)
        # update the indices of r_ij that need are within the 2.5 sigma cutoff
        self.cutoff_ids = self._find_cutoff_ids()
        # indices pair that are within cutoff
        effective_id_pairs = self.id_pairs[self.cutoff_ids]
        # pair forces from del operator
        pair_forces = self.del_potential(
            self.del_pot_coef,
            self.r_ij.reshape(-1,1)[self.cutoff_ids],
            dist_vecs[self.cutoff_ids]
        )
        f_atoms = self._get_atom_forces_ni_sum(pair_forces, effective_id_pairs)
        # acceleration
        accl = f_atoms/(self.m_argon_kg)
        return accl

    def _init_step(self) -> None:
        """The first two steps calculated using the taylor series expansion
        for t=0 and t+∆t"""
        self.accl = self.get_accelerations(self.init_coords)
        self.pair_potentials = self._compute_pair_potentials()
        self.coords = self.init_coords+self.velo*self.dt+self.accl*self.dt**2/2
        self._apply_pbc_coord(self.coords, self.box_len)
        self.last_coords = self.init_coords

    def step(self) -> None:
        """
        Advance the simulation for one time step ∆t. New velocities, 
        accelerations, coordinates, and last coordinates will be updated."""
        self.accl = self.get_accelerations(self.coords)
        self.pair_potentials = self._compute_pair_potentials()
        new_coords = self.verlet(
            self.coords, self.last_coords, self.accl, self.dt
        )
        self._apply_pbc_coord(new_coords, self.box_len)
        dx_2dt = new_coords - self.last_coords
        self._apply_pbc_dist(dx_2dt, self.box_len)
        self.velo = dx_2dt/(2*self.dt)
        self.last_coords = self.coords
        self.coords = new_coords

    def step_velocity_verlet(self) -> None:
        self.coords = self.coords+self.velo*self.dt+self.accl*self.dt**2/2
        self._apply_pbc_coord(self.coords, self.box_len)
        self.velo = self.velo+self.accl*self.dt/2
        self.accl = self.get_accelerations(self.coords)
        self.pair_potentials = self._compute_pair_potentials()
        self.velo = self.velo+self.accl*self.dt/2

    