import uuid
import numpy as np
import nglview as nv

class ArgonSimTrajectory(nv.Trajectory, nv.Structure):
    ext = "pdb"  # or gro, cif, mol2, sdf
    params = {}  # loading options passed to NGL
    def __init__(self, coords):
        self.ext = "pdb"
        self.params = {}
        self.id = str(uuid.uuid4())
        
        if len(coords.shape) != 3 or coords.shape[-1] != 3:
            raise ValueError(f'Incompatible coords shape: {coords.shape}')
        self.n_atoms = coords.shape[1]
        self.coords = coords*1e10 # PDB File unit in angstrom
        self._ATOM_FORMAT_STRING = (
            "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%s%6.2f      %4s%2s%2s\n"
        )
    def get_pdb_atom_line(self, atom_index):
        # initial coord is used here
        x, y, z = self.coords[0, atom_index]
        args = (
                'HETATM',       # record type
                atom_index+1,   # atom number
                'ARN',          # atom name
                ' ',            # alt loc
                'ARN',          # resname
                'A',            # chain id
                atom_index+1,   # res seq id
                ' ',            # insertion code
                x,
                y,
                z,
                " " * 6,        # occupancy
                0.0,            # b-factor
                "ARGN",         # segid
                "AR",           # element symbol
                '0',            # charge
            )
        return self._ATOM_FORMAT_STRING % args
    
    def get_structure_string(self):
        lines = ''
        for i in range(self.n_atoms):
            lines += self.get_pdb_atom_line(i)
        return lines
        
    def get_coordinates(self, index):
        # return 2D numpy array, shape=(n_atoms, 3)
        return self.coords[index]
    
    @property
    def n_frames(self):
        return self.coords.shape[0]  # return number of frames