from itertools import combinations, permutations
import numpy as np
from numpy import linalg as LA
from scipy import ndimage as ni
from numba import uint32, float32, float64    # import the types
from numba.experimental import jitclass

spec = [
    # class default attr
    ('n_cell', uint32),
    ('m_argon', float32),
    ('m_argon_kg', float32),
    ('kb', float32),
    ('sigma', float32),
    ('lennard_jones_cutoff', float32),
    # instance attr
    ('temp', float32),
    ('dt', float32),
    ('box_len', float32),
    ('lj_coef', float64),
    ('init_coord', float64[:,:]),
    ('_idx_array', uint32[:]),
    ('velo', float64[:,:]),
    ('id_pairs', uint32[:,:]),
    ('coords', float64[:,:]),
    ('r_ij', float64[:,:]),
    ('cutoff_ids', uint32[:,:]),
    ('_pair_potentials', float64[:,:]),
    ('potentials', float64[:,:]),
    ('last_coords', float64[:,:]),
    ('accl', float64[:,:])
]

@jitclass(spec)
class ArgonSim:
    """Simple molecular dynamics simulation engine for Argon systems. This is a
    reimplementation of the Verlet paper on Phys Review, 1976 except that this
    simulator does not use the reduced unit as Verlet did."""
    n: float = 6.0221415e23 # /mol
    m_argon: float = 39.948/n # gram
    m_argon_kg: float = m_argon*1e-3 # mass of one Argon atom in kg
    kb: float = 1.3806503e-23 # J/K
    # characteristic length for lennard jones
    sigma: float = 3.4e-10 # m
    # 2.5 sigma is set as lennard jones potential cutoff
    lennard_jones_cutoff: float = 2.5*sigma
    def __init__(
        self,
        n_cells : int,
        temp : float = 120,
        dt : float = 1e-14,
        rho : float = 1.374 # g/cm^3 @ 130K
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

            rho
            param : Density of the Argon system, unit in g/cm^3. Default to 
                    1.374 g/cm^3 (@130K)
            type : float
        """
        if not isinstance(n_cells, int):
            raise TypeError(
                "Integer value is required to specify the number of unit cells"
            )
        self.temp = temp # Kelvin
        self.dt = dt # second
        L = (self.m_argon*864/rho)**(1/3) # centimeter
        self.box_len = L * 1e-2 # system dimension in meters
        self.lj_coef = 4*120*self.kb # Lennard Jones energy coeff
        self.init_coords = self._init_coords(self.box_len, n_cells) # meters
        self._idx_array = np.arange(self.init_coords.shape[0])
        self.velo = self._init_velocities(self.temp, self.init_coords.shape)
        # combination of indices (i, j) for atom i and atom j
        self.id_pairs = np.array(
            list(combinations(range(len(self.init_coords)),2))
        )
        self.coords = np.empty_like(self.init_coords, dtype=float)
        self.r_ij = None
        self.cutoff_ids = None
        self._pair_potentials = None
        self.potentials = None
        self.last_coords = None
        self.accl = None
        self._init_step()
        
    def _init_coords(self, box_len, n_cells) -> np.ndarray:
        """
        Initialize coordinates in face centered cubic unit cells
        """
        cell_len = box_len/n_cells
        x_corner = np.linspace(0, box_len-cell_len, n_cells) # corner atom spacings
        x_face = np.linspace(cell_len/2,box_len-cell_len/2, n_cells) # face atom spacings

        grid_c = np.meshgrid(x_corner, x_corner, x_corner, indexing='ij')
        grid_f = np.meshgrid(x_face, x_corner, x_face, indexing='ij')
        # get all 3 face nodes by combinations of grid_f
        grid_3f = np.array(tuple(permutations(grid_f))[:3]).reshape((3,3,-1))
        face_coords = np.concatenate(grid_3f, axis = 1)
        corner_coords = np.array(grid_c).reshape((3,-1))
        # N x 3 coords array
        coords = np.concatenate([face_coords, corner_coords], axis = 1).T
        return coords
    
    def _init_velocities(self, temp, size) -> np.ndarray:
        """Initialize velocities for each atom from a Maxwell-Boltzmann 
        distribution"""
        kbT_over_m = 1.3806503*temp/(39.948e-3*6.0221415**(-1)) # m^2/s^2
        velo_std = (kbT_over_m)**0.5 # m/s
        init_velo = np.random.normal(0, velo_std, size)
        return init_velo
    
    def get_speeds(self, velocities) -> np.ndarray:
        """Get speed for a given array of velocities.
        - Args
            velocities
            param : array of velocity vetors with the shape (N, 3) for N atoms
            type : np.array
        
        - Return
            speed
            param : the vector norm calculated from the velocities with size of
            (N,)
            type : np.array
        """
        return LA.norm(velocities, axis = 1)
    
    def _apply_pbc_dist(self, d_vecs: float64) -> None:
        """Apply periodic boundary conditions on distance vectors. Shortest 
        distance will be determined from self.box_length and assigned to the 
        distance array directly"""
        mask1 = d_vecs > self.box_len/2
        mask2 = d_vecs < -self.box_len/2
        new_vecs1 = self.box_len - d_vecs[mask1]
        new_vecs2 = self.box_len + d_vecs[mask2]
        d_vecs[mask1] = new_vecs1
        d_vecs[mask2] = new_vecs2

    def _apply_pbc_coord(self, coords: float64) -> None:
        """Apply periodic boundary conditions on coordinates. Any out of bound 
        coord will be wrapped and place to the other side of the box. The new 
        coords will be assigned to the self.coords array directly"""
        mask1 = coords > self.box_len
        mask2 = coords < 0
        wrapped_coords1 = coords[mask1] % self.box_len
        wrapped_coords2 = self.box_len - ((-coords[mask2]) % self.box_len)
        coords[mask1] = wrapped_coords1
        coords[mask2] = wrapped_coords2

    @staticmethod
    def verlet(curr_x, last_x, dt, accl) -> np.ndarray:
        """Verlet algorithm"""
        return 2*(curr_x)-last_x+accl*dt**2

    def lennard_jones(self, r: float64[:]) -> np.ndarray:
        ''' 
        Lennard-Jones potentials
        v_LJ(r) = 4e[(sigma/r)^12-(sigma/4)^6]
        units in J
        epsilon/k_b = 120K => epsilon: J (m^2 kg s^-2)
        '''
        return self.lj_coef*((self.sigma/r)**12-(self.sigma/r)**6)
    
    def _find_cutoff_ids(self) -> np.ndarray:
        cutoff_mask = np.logical_and(
            self.r_ij <= self.lennard_jones_cutoff,
            self.r_ij > 0
        )
        return np.nonzero(cutoff_mask)[0]

    def get_cutoff_ids(self) -> np.ndarray:
        """Get the indices of r_ij that need are within the 2.5 sigma cutoff
        """
        if self.cutoff_ids is None:
            self.cutoff_ids = self._find_cutoff_ids()
        return self.cutoff_ids

    def _compute_pair_potentials(self):
        """Compute the potentials from the cutoff list only. Return pairwise
        potentials of atoms"""
        return self.lennard_jones(self.r_ij[self.cutoff_ids])

    @staticmethod
    def _sum_pair_values_1d(
            pair_values_1d: float64[:],
            pair_labels: uint32[:],
            idx_array: uint32[:]
        ) -> np.ndarray:
        """
        Sum pairwise values (i,j pairs) according the label ids. The directions
        of the pairwise values (e.g. forces, potentials) are taken into account.
        """
        # force/potential j to i
        val_in = ni.sum(pair_values_1d, pair_labels[:,0], idx_array)
        # force/potential i to j
        val_out = ni.sum(pair_values_1d, pair_labels[:,1], idx_array)
        val_total = val_in - val_out
        return val_total

    def get_potentials(self) -> np.ndarray:
        """Get Lennard Jones potentials from the distance matrix (list form) 
        based on the distance cutoffs (2.5 sigma). The overall potentials from 
        the id pairs in the current id_cutoff list will be evalutated and summed
        up for each atoms.
            
        Return
            potentials
            param : array of potentials of size N
            type : np.array
        """
        if self._pair_potentials is None:
            self._pair_potentials = self._compute_pair_potentials()
        
        atom_potentials = np.zeros(self.coords.shape[0])
        effective_id_pairs = self.id_pairs[self.cutoff_ids]
        atom_potentials = self._sum_pair_values_1d(
            self._pair_potentials, effective_id_pairs, self._idx_array
        )
        return atom_potentials

    def get_accelerations(self, coords) -> np.ndarray:
        """
        Get accelerations from a set of coordinates. The pairwaise distance 
        vectors and distances (vector norm) will be determined first. The 
        atom pairs that are within distance cutoff will be selected, and the 
        Lennard Jones potentials, the force vectors, and the accelerations on 
        each atom will be calculated. Only acceleration array is returned

        Args
            coords
            param : a set of coordinates of the Argon atoms at a given time step
            type: np.array
            
        Return
            acclerations
            param : acceleration vectors for each atom with shape (N, 3)
            type : np.array
        """
        id_pairs = self.id_pairs
        # dist vector from atom j to atom i
        dist_vecs = coords[id_pairs][:,0] - coords[id_pairs][:,1]
        # apply periodic boundary conditions
        self._apply_pbc_dist(dist_vecs)
        # update the pairwise euclidean distances from atom j to atom i
        self.r_ij = LA.norm(dist_vecs, axis = 1)
        # update the indices of r_ij that need are within the 2.5 sigma cutoff
        self.cutoff_ids = self._find_cutoff_ids()
        # update pairwise potentials
        self._pair_potentials = self._compute_pair_potentials()
        # calculate pairwise force vectors (x,y,z) from atom j to atom i
        pair_forces = (
            dist_vecs[self.cutoff_ids].T * self._pair_potentials / self.r_ij[self.cutoff_ids]
        ).T
        # indices pair that are within cutoff
        effective_id_pairs = id_pairs[self.cutoff_ids]
        # force vectors for each atom
        f_atoms = np.empty(coords.shape)
        fx, fy, fz = pair_forces.T
        for i, f in enumerate((fx, fy, fz)):
            # sum all pairwise forces by each dimension on each atom
            f_atoms[:,i] = self._sum_pair_values_1d(
                f, effective_id_pairs, self._idx_array
            )
        # acceleration
        accl = f_atoms/(self.m_argon_kg)
        return accl

    def _init_step(self) -> None:
        """The first two steps calculated using the taylor series expansion
        for t=0 and t+∆t"""
        self.accl = self.get_accelerations(self.init_coords)
        self.last_coords = self.init_coords
        self.coords = self.last_coords+self.velo*self.dt+self.accl*self.dt**2/2
        self._apply_pbc_coord(self.coords)
        self.accl = self.get_accelerations(self.coords)

    def step(self) -> None:
        """
        Advance the simulation for one time step ∆t. New velocities, 
        accelerations, coordinates, and last coordinates will be updated."""
        new_coords = self.verlet(
            self.coords, self.last_coords, self.dt, self.accl
        )
        self.velo = (new_coords - self.last_coords)/(2*self.dt)
        self._apply_pbc_coord(new_coords)
        self.accl = self.get_accelerations(new_coords)
        self.last_coords = self.coords
        self.coords = new_coords

    