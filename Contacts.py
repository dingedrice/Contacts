from openmm.app import *
from openmm import *
from openmm.unit import *

from OpenSMOG.OpenSMOG import SBM

import sys
import os
import numpy as np
from collections.abc import Iterable

def _isarrayint(index):
    if isinstance(index, (list, tuple)) and all(isinstance(i, int) for i in index):
        return True
    elif isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.integer):
        return True
    else:
        return False

class _Contact():
    def __init__(self, atom1, atom2, distance):
        self.atom1 = atom1
        self.atom2 = atom2
        self.distance = distance
        
    def __str__(self):
        return f"Atom1: {self.atom1:d}, Atom2: {self.atom2:d}, Distance: {self.distance:5.3f} nm"
    
    def __repr__(self):
        return self.__str__()

class Contacts():
    def __init__(self, Nbits = 32):
        self._atom1 = np.array([], dtype=f"int{Nbits}")
        self._atom2 = np.array([], dtype=f"int{Nbits}")
        self._distance = np.array([], dtype=f"float{Nbits}")

    def __len__(self):
        return len(self._atom1)

    def getNumContacts(self):
        return len(self._atom1)

    def getContact(self, index):
        if not isinstance(index, int):
            print(f"\n\ngetContact Error: index must be an integer")
            sys.exit(1)
        return _Contact(self._atom1[index], self._atom2[index], self._distance[index])
    
    def getAtom1Array(self):
        return self._atom1.copy()
    
    def getAtom2Array(self):
        return self._atom2.copy()
    
    def getDistanceArray(self):
        return self._distance.copy()
    
    def rescaledDistances(self, scale):
        return scale*self._distance
    
    def getIndices(self):
        return np.array([self._atom1, self._atom2]).T

    def toArray(self):
        return np.array([self._atom1, self._atom2, self._distance]).T
    
    def addContacts(self, atom1, atom2, distance):
        if isinstance(atom1, int) and isinstance(atom2, int) and isinstance(distance, (int, float)):
            pass
        elif _isarrayint(atom1) and _isarrayint(atom2):
            if len(atom1) != len(atom2) or len(atom1) != len(distance):
                print(f"addContacts Error: atom1 (len = {len(atom1)}), atom2 (len = {len(atom2)}) and distance (len = {len(atom2)}) must have the same length")
                sys.exit(1)
        else:
            print(f"addContacts Error: atom1 {atom1} and atom2 {atom2} must be integers or arrays of integers")
            sys.exit(1)

        self._atom1 = np.append(self._atom1, atom1)
        self._atom2 = np.append(self._atom2, atom2)
        self._distance = np.append(self._distance, distance)

    def removeContacts(self, index):
        self._atom1 = np.delete(self._atom1, index)
        self._atom2 = np.delete(self._atom2, index)
        self._distance = np.delete(self._distance, index)

    def setContacts(self, index, atom1, atom2, distance):
        if isinstance(index, int) and isinstance(atom1, int) and isinstance(atom2, int) and isinstance(distance, (int, float)):
            pass
        elif _isarrayint(atom1) and _isarrayint(atom2):
            if len(index) != len(atom1) or len(index) != len(atom2) or len(index) != len(distance):
                print(f"setContacts Error: index (len = {len(index)}), atom1 (len = {len(atom1)}), atom2 (len = {len(atom2)}) and distance (len = {len(distance)}) must have the same length")
                sys.exit(1)
        else:
            print(f"setContacts Error: atom1 {atom1} and atom2 {atom2} must be integers or arrays of integers")
            sys.exit(1)

        self._atom1[index] = atom1
        self._atom2[index] = atom2
        self._distance[index] = distance

class ContactsOnuchic(Contacts):
    def _loadContacts_openmm(self, openmm_forces, input_distance_calc_dict):
        if not isinstance(openmm_forces, Iterable):
            openmm_forces = (openmm_forces, )

        # map Openmm energy functions to distance calculation functions
        distance_calc_dict = {
            'A/r^12-B/r^6': lambda A, B: (2*A/B)**(1/6),
            'A/r^12-B/r^10': lambda A, B: np.sqrt(6/5*A/B),
            '-C/r^6+A/r^12': lambda C, A: (2*A/C)**(1/6)
            }
        if input_distance_calc_dict != None:
            distance_calc_dict.update(input_distance_calc_dict)

        for openmm_force in openmm_forces:
            if not isinstance(openmm_force, CustomBondForce):
                print(f"\n\nloadContacts Error: openmm_forces contains not only CustomBondForce: {openmm_force.getName()}")
                sys.exit(1)

            energy = openmm_force.getEnergyFunction()
            try:
                distance_function = distance_calc_dict[energy]
            except:
                print(f"\n\nloadContacts Error: energy function {openmm_force.getEnergyFunction()} is not supported, please provide a python dictionary mapping the energy function to the distance calculation for the calculation of distances with the force parameters")
                sys.exit(1)

            for i in range(openmm_force.getNumBonds()):
                atom1, atom2, distance_params = openmm_force.getBondParameters(i)
                try:
                    d = distance_function(*distance_params)
                except:
                    d = None
                    print(f"\n\nloadContacts Error: distance calculation function {distance_function} does not match the parameters {distance_params}")
                    sys.exit(1)
                self.addContacts(atom1, atom2, d)

    def _loadContacts_file(self, contacts_file, input_distance_calc_dict, coord):
        def ignore_dict_note(input_distance_calc_dict):
            if input_distance_calc_dict != None:
                print(f"\n\nloadContacts Note: input_distance_calc_dict will be ignored")

        if contacts_file.endswith('.xml'):
            # Construct forces from xml file 
            tmp = SBM(name='tmp',time_step=0.002,collision_rate=1.0,r_cutoff=0.65,temperature=0.5)
            tmp.system = System()
            tmp.loadXml(contacts_file)
            if not tmp.contacts_present:
                print(f"\n\nloadContacts Error: did find any contacts in the xml contacts_file {contacts_file}")
                sys.exit(1)
            forces = [tmp.forcesDict[i] for i in tmp.contacts]

            if coord.size == 0:
                self._loadContacts_openmm(forces, input_distance_calc_dict)
            else:
                ignore_dict_note(input_distance_calc_dict)
                for force in forces:
                    for i in range(force.getNumBonds()):
                        atom1, atom2, _ = force.getBondParameters(i)
                        d = np.linalg.norm(coord[atom1] - coord[atom2])
                        self.addContacts(atom1, atom2, d)
        elif contacts_file.endswith('.top'):
            if coord.size == 0:
                top = GromacsTopFile(contacts_file)
                system = top.createSystem()
                forces = [force for force in system.getForces() if isinstance(force, CustomBondForce)]
                print(f"\n\nloadContacts Note: Lennard-Jones potential is assumed")
                ignore_dict_note(input_distance_calc_dict)
                self._loadContacts_openmm(forces, input_distance_calc_dict)
            else:
                # When coordinates are available, openmm tools are not used
                # Thw reason is that we only need atom pairs and openmm only supports Lennard-Jones potential so that the loading may fail
                print(f"\n\nloadContacts Note: top file {contacts_file} detected, will read contacts from [ exclusions ]")
                ignore_dict_note(input_distance_calc_dict)

                with open(contacts_file, 'r') as read_obj:
                    exclusions_section = False
                    for line in read_obj:
                        if 'exclusions' in line:
                            exclusions_section = True
                        elif '[' in line:         
                            exclusions_section = False
                        elif exclusions_section:      
                            try:
                                atom1, atom2 = [int(i)-1 for i in line.split()]               
                                d = np.linalg.norm(coord[atom1] - coord[atom2])  
                                self.addContacts(atom1, atom2, d)
                            except:
                                pass
        else:
            print(f"\n\nloadContacts Error: contacts_file {contacts_file} must be a xml or top file")
            sys.exit(1)
        
    def loadContacts(self, openmm_forces = None, contacts_file = None, input_distance_calc_dict = None, gro_file = None, array = np.array([])):        
        def loadCoordinate(gro_file):
            gro = GromacsGroFile(gro_file)
            coord = gro.getPositions(asNumpy = True).value_in_unit(nanometer)
            print(f"\n\nloadContacts Note: gro file {gro_file} detected, will use the coordinates to calculate distances")
            return coord
        
        coord = np.array([])
        array = np.array(array)

        if openmm_forces != None:
            if contacts_file != None:
                print(f"\n\nloadContacts Note: openmm_forces detected, will ignore contacts_file {contacts_file}")
            if gro_file != None:
                print(f"\n\nloadContacts Note: openmm_forces detected, will ignore gro_file {gro_file}")
            if array.size != 0:
                print(f"\n\nloadContacts Note: openmm_forces detected, will ignore array {str(array)}")

            self._loadContacts_openmm(openmm_forces, input_distance_calc_dict)
        elif contacts_file != None:
            if not os.path.exists(contacts_file):
                print(f"\n\nloadContacts Error: contacts_file {contacts_file} does not exist")
                sys.exit(1)

            if gro_file != None:
                if not os.path.exists(gro_file):
                    print(f"\n\nloadContacts Error: gro_file {gro_file} does not exist")
                    sys.exit(1)
                elif array.size != 0:
                    print(f"\n\nloadContacts Error: gro_file and array cannot be both present")
                    sys.exit(1)
                else:
                    coord = loadCoordinate(gro_file)
            elif array.size != 0:
                coord = array.copy()
                if coord.shape[1] != 3:
                    print(f"\n\nloadContacts Error: array must have shape (N, 3)")
                    sys.exit(1)
                print(f"\n\nloadContacts WARNING: it could be dangerous to use array, and it is your responsibility to ensure that the coordinates are in the unit of nm and match the contact_file")
                
            self._loadContacts_file(contacts_file, input_distance_calc_dict, coord)
        else:
            print(f"\n\nloadContacts Error: openmm_forces and contacts_file cannot be both absent")
            sys.exit(1)

class QOnuchicReporter(StateDataReporter):
    def __init__(self, file, reportInterval, contacts, cutoff = 1.5, Qi = False, **kwargs):
        if not isinstance(contacts, Contacts):
            print(f"\n\nQOnuchicReporter Error: contacts must be of type Contacts")
            sys.exit(1)

        super().__init__(file, reportInterval, **kwargs)
        self._write_qi = Qi
        self._cutoff = cutoff
        # load contacts
        self._N = len(contacts)
        self._atom1 = contacts.getAtom1Array()
        self._atom2 = contacts.getAtom2Array()
        self._threshold = contacts.rescaledDistances(cutoff)
    
    def _constructHeaders(self):
        headers = super()._constructHeaders()
        headers.append("Q_Onuchic")
        if self._write_qi:
            headers.append("Qi_Onuchic")
        return headers
    
    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)
        positions = state.getPositions(asNumpy = True).value_in_unit(nanometer)

        displacement = positions[self._atom1] - positions[self._atom2]
        # check for periodic boundary conditions
        if simulation.system.usesPeriodicBoundaryConditions():
            box = state.getPeriodicBoxVectors(asNumpy = True).value_in_unit(nanometer)
            if np.allclose(box, np.diag(np.diagonal(box))):
                box = np.diagonal(box)
            else:
                print(f"\n\nQOnuchicReporter Error: Only orthorhombic boxes are supported, check your pbc")
                sys.exit(1)
            displacement -= np.round(displacement/box)*box

        distance = np.linalg.norm(displacement, axis = -1)
        qi = distance < self._threshold
        q = np.mean(qi, axis = -1)

        values.append(q)
        if self._write_qi:
            qi_str = "".join(map(str, qi.astype(int)))
            values.append(qi_str)
        return values
    
def qOnuchic(traj, contacts, cutoff = 1.5):
    if not isinstance(contacts, Contacts):
        print(f"\n\nqOnuchic Error: contacts must be of type Contacts")
        sys.exit(1)

    import mdtraj as md
    if not isinstance(traj, md.Trajectory):
        print(f"\n\nqOnuchic Error: traj must be of type md.Trajectory")
        sys.exit(1)
    
    indices = contacts.getIndices()
    threshold = contacts.rescaledDistances(cutoff)

    d = md.compute_distances(traj, indices)
    q0 = np.mean(d < threshold, axis = -1)

    return q0

    