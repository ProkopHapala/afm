from abc import ABC, abstractmethod

import numpy as np

from .. import elements
from ..atomicUtils import findBonds_
from ..ocl import field as FFcl
from ..ocl import relax as oclr


class AuxMapBase(ABC):
    """Base class for AuxMap subclasses.

    Each subclass must override the method `eval`, which gets called when the object instance is called.

    Arguments:
        scan_dim: tuple of two ints. Indicates the pixel size of the scan in x and y.
        scan_window: tuple ((start_x, start_y), (end_x, end_y)). The start and end coordinates of scan region in angstroms.
        zmin: float. Deepest coordinate that is still included. Top is defined to be at 0.
    """

    def __init__(self, scan_dim, scan_window, zmin=None):
        if not FFcl.oclu:
            raise RuntimeError("OpenCL context not initialized. Initialize with ocl.field.init before creating an AuxMap object.")
        self.scan_dim = scan_dim
        self.scan_window = scan_window
        self.projector = FFcl.AtomProjection()
        if zmin:
            self.projector.zmin = zmin
        self.nChan = 1

    def __call__(self, xyzqs=None, Zs=None, pot=None, rot=np.eye(3)):
        if xyzqs is not None:
            assert xyzqs.shape[1] == 4
            xyzqs = xyzqs.copy()
            xyz_center = xyzqs[:, :3].mean(axis=0)
            xyzqs[:, :3] = np.dot(xyzqs[:, :3] - xyz_center, rot.T) + xyz_center
        return self.eval(xyzqs, Zs, pot, rot)

    @abstractmethod
    def eval(xyzqs, Zs, pot, rot):
        """
        Arguments:
            xyzqs: numpy.ndarray of floats. xyz coordinates and charges of atoms in molecule
            Zs: numpy.ndarray of ints. Elements of atoms in molecule.
            pot: HartreePotential. Sample hartree potential.
            rot: np.ndarray of shape (3, 3). Sample rotation.
        """

    def prepare_projector(self, xyzqs, Zs, pos0, bonds2atoms=None, elem_channels=None):
        rot = np.eye(3)
        coefs = self.projector.makeCoefsZR(Zs.astype(np.int32), elements.ELEMENTS)
        self.projector.tryReleaseBuffers()
        self.projector.prepareBuffers(
            xyzqs.astype(np.float32),
            self.scan_dim[:2] + (self.nChan,),
            coefs=coefs,
            bonds2atoms=bonds2atoms,
            elem_channels=elem_channels,
        )
        return oclr.preparePossRot(
            self.scan_dim,
            pos0,
            rot[0],
            rot[1],
            self.scan_window[0],
            self.scan_window[1],
        )


class vdwSpheres(AuxMapBase):
    """
    Generate vdW Spheres descriptors for molecules. Each atom is represented by a projection of a sphere
    with radius equal to the vdW radius of the element.

    Arguments:
        Rpp: float. A constant that is added to the vdW radius of each atom.
    """

    def __init__(self, scan_dim=(128, 128), scan_window=((-8, -8), (8, 8)), zmin=-1.5, Rpp=-0.5):
        super().__init__(scan_dim, scan_window, zmin)
        self.projector.Rpp = Rpp

    def eval(self, xyzqs, Zs, pot=None, rot=None):
        coefs = self.projector.makeCoefsZR(Zs, elements.ELEMENTS)
        pos0 = [0, 0, (xyzqs[:, 2] + coefs[:, 3]).max() + self.projector.Rpp]
        poss = self.prepare_projector(xyzqs, Zs, pos0)
        return self.projector.run_evalSpheres(poss=poss, tipRot=oclr.mat3x3to4f(np.eye(3)))[:, :, 0]


class AtomicDisks(AuxMapBase):
    """
    Generate Atomic Disks descriptors for molecules. Each atom is represented by a conically decaying disk.

    Arguments:
        zmax_s: float. The maximum depth of vdW sphere shell when diskMode='sphere'.
        Rpp: float. A constant that is added to the vdW radius of each atom.
        diskMode: 'sphere' or 'center'. With 'center' only the center coordinates are considered, when deciding
                  whether an atom is too deep. With 'sphere' also the effective size of the atom is taken into
                  account.
    """

    def __init__(
        self,
        scan_dim=(128, 128),
        scan_window=((-8, -8), (8, 8)),
        zmin=-1.2,
        zmax_s=-1.2,
        Rpp=-0.5,
        diskMode="sphere",
    ):
        super().__init__(scan_dim, scan_window)
        self.projector.dzmax = -zmin
        self.projector.dzmax_s = -zmax_s
        self.projector.Rpp = Rpp
        self.diskMode = diskMode
        if diskMode == "sphere":
            self.offset = 0.0
        elif diskMode == "center":
            self.projector.dzmax_s = np.inf
            self.offset = -1.0
        else:
            raise ValueError(f"Unknown diskMode {diskMode}. Should be either sphere or center")

    def eval(self, xyzqs, Zs, pot=None, rot=None):
        coefs = self.projector.makeCoefsZR(Zs, elements.ELEMENTS)
        coords_sphere = xyzqs[:, 2] + coefs[:, 3] + self.projector.Rpp
        offset = coords_sphere.max() - xyzqs[:, 2].max() + self.offset
        pos0 = [0, 0, coords_sphere.max()]
        poss = self.prepare_projector(xyzqs, Zs, pos0)
        return self.projector.run_evaldisks(poss=poss, tipRot=oclr.mat3x3to4f(np.eye(3)), offset=offset)[:, :, 0]


class HeightMap(AuxMapBase):
    """
    Generate HeightMap descriptors for molecules. Represents the combined interaction of probe with atoms
    in molecule as a isosurface of the z-component of the forcefield around the molecule.

    The HeightMap and ESMap descriptors are different from the other ones in that they depend on the simulation parameters.
    Before calling HeightMap, first do a scan of the molecule with the scanner to get the forces. Giving the atoms xyzqs and
    elements Zs as input arguments for eval is optional, since they are not used for anything.

    Arguments:
        scanner: Instance of oclr.RelaxedScanner.
        iso: float. The value of the isosurface.
    """

    def __init__(self, scanner, zmin=-2.0, iso=0.1):
        self.scanner = scanner
        self.zrange = -zmin
        self.iso = iso
        self.nz = 100  # Number of steps in downwards scan

    def eval(self, xyzqs=None, Zs=None, pot=None, rot=None):
        self.scanner.prepareAuxMapBuffers(bZMap=True, bFEmap=False)
        Y = (self.scanner.run_getZisoTilted(iso=self.iso, nz=self.nz) * -1).copy()
        Y *= self.scanner.zstep
        Ymin = max(Y[Y <= 0].max() - self.zrange, Y.min())
        Y[Y > 0] = Ymin
        Y[Y < Ymin] = Ymin
        Y -= Ymin
        return Y


class ESMap(AuxMapBase):
    """
    Generate ESMap and HeightMap descriptors for molecules. Represents the charge distribution around
    the molecule as the z-component of the electrostatic field calculated on the surface defined by
    the HeightMap.

    The HeightMap and ESMap descriptors are different from the other ones in that they depend on the simulation parameters.
    Before calling ESMap, first do a scan of the molecule with the scanner to get the forces. Giving the elements Zs
    as an input argument for eval is optional, since it is not used for anything.

    Arguments:
        scanner: Instance of oclr.RelaxedScanner.
        iso: float. The value of the isosurface.
    """

    def __init__(self, scanner, zmin=-2.0, iso=0.1):
        self.scanner = scanner
        self.zrange = -zmin
        self.iso = iso
        self.nz = 100  # Number of steps in downwards scan

    def eval(self, xyzqs, Zs=None, pot=None, rot=None):
        self.scanner.prepareAuxMapBuffers(bZMap=True, bFEmap=True, atoms=xyzqs.astype(np.float32))
        zMap, feMap = self.scanner.run_getZisoFETilted(iso=self.iso, nz=self.nz)
        Ye = (feMap[:, :, 2]).copy()  # Fel_z
        zMap *= -(self.scanner.zstep)
        zMin = max(zMap[zMap <= 0].max() - self.zrange, zMap.min())
        zMap[zMap > 0] = zMin
        zMap[zMap < zMin] = zMin
        zMap -= zMin
        Ye[zMap == 0] = 0
        Y = np.stack([Ye, zMap], axis=2)
        return Y


class ESMapConstant(AuxMapBase):
    """
    Generate constant-height ESMap descriptor for molecules. Represents the charge distribution around
    the molecule as the z-component of the electrostatic field calculated on a constant-height surface.

    Arguments:
        height: float. The height of the constant-height slice, counted up from the center of the top atom.
        vdW_cutoff: float <0.0 or None. Use vdW-Spheres descriptor as a mask to cutoff regions without atoms. The cutoff
            is the same as zmin for the vdW-Spheres descriptor. If None, don't use cutoff, and calculate
            the ES Map descriptor for whole slice.
        Rpp: float. A constant that is added to the vdW radius of each atom if vdW_cutoff is set.
    """

    def __init__(
        self,
        scan_dim=(128, 128),
        scan_window=((-8, -8), (8, 8)),
        height=4.0,
        vdW_cutoff=None,
        Rpp=0.5,
    ):
        super().__init__(scan_dim, scan_window, vdW_cutoff)
        self.height = height
        self.nChan = 4
        self.vdW_cutoff = vdW_cutoff
        self.projector.Rpp = Rpp

    def eval(self, xyzqs, Zs=None, pot=None, rot=np.eye(3)):
        pos0 = [0, 0, xyzqs[:, 2].max() + self.height]
        if pot:
            self.nChan = 1
            # The scan window for the hartree potential needs to rotate to the opposite
            # direction compared to the molecule coordinates.
            rot = np.linalg.inv(rot)
            rot_center = xyzqs[:, :3].mean(axis=0)
            poss = self.prepare_projector(xyzqs, Zs, pos0)
            es = self.projector.run_evalHartreeGradient(pot, poss, rot=rot, rot_center=rot_center)
        else:
            self.nChan = 4
            poss = self.prepare_projector(xyzqs, Zs, pos0)
            es = self.projector.run_evalCoulomb(poss=poss)[:, :, 2]  # Last dim = (E_x, E_y, E_z, V)

        if self.vdW_cutoff:
            self.nChan = 1  # Projector needs only one channel for vdW Spheres
            coefs = self.projector.makeCoefsZR(Zs, elements.ELEMENTS)
            pos0 = [0, 0, (xyzqs[:, 2] + coefs[:, 3]).max() + self.projector.Rpp]
            poss = self.prepare_projector(xyzqs, Zs, pos0)
            vdW = self.projector.run_evalSpheres(poss=poss, tipRot=oclr.mat3x3to4f(np.eye(3)))[:, :, 0]
            es[vdW == vdW.min()] = 0.0

        return es


class MultiMapSpheres(AuxMapBase):
    """
    Generate Multimap vdW Spheres descriptors for molecules. Each atom is represented by a projection of a sphere
    with radius equal to the vdW radius of the element. Different sizes of atoms are separated into different
    channels based on their vdW radii.

    Arguments:
        Rpp: float. A constant that is added to the vdW radius of each atom.
        nChan: int. Number of channels.
        Rmin: float. Minimum radius.
        Rstep: float. Size range per bin.
        bOccl: 0 or 1. Switch occlusion of atoms 0=off 1=on.
    """

    def __init__(
        self,
        scan_dim=(128, 128),
        scan_window=((-8, -8), (8, 8)),
        zmin=-1.5,
        Rpp=-0.5,
        nChan=3,
        Rmin=1.4,
        Rstep=0.3,
        bOccl=0,
    ):
        super().__init__(scan_dim, scan_window, zmin)
        self.projector.Rpp = Rpp
        self.nChan = nChan
        self.Rmin = Rmin
        self.Rstep = Rstep
        self.bOccl = bOccl

    def eval(self, xyzqs, Zs, pot=None, rot=None):
        coefs = self.projector.makeCoefsZR(Zs, elements.ELEMENTS)
        pos0 = [0, 0, (xyzqs[:, 2] + coefs[:, 3]).max() + self.projector.Rpp]
        poss = self.prepare_projector(xyzqs, Zs, pos0)
        return self.projector.run_evalMultiMapSpheres(
            poss=poss,
            tipRot=oclr.mat3x3to4f(np.eye(3)),
            bOccl=self.bOccl,
            Rmin=self.Rmin,
            Rstep=self.Rstep,
        )


class MultiMapSpheresElements(AuxMapBase):
    """
    Generate Multimap vdW Spheres descriptors for molecules. Each atom is represented by a projection of a sphere
    with radius equal to the vdW radius of the element. Different elements can be separated arbitrarily into different
    channels.

    Arguments:
        Rpp: float. A constant that is added to the vdW radius of each atom.
        elems: list of lists of int or str. Lists of elements in each channel as the atomic numbers or symbols.
        bOccl: 0 or 1. Switch occlusion of atoms 0=off 1=on.
    """

    def __init__(
        self,
        scan_dim=(128, 128),
        scan_window=((-8, -8), (8, 8)),
        zmin=-1.5,
        Rpp=-0.5,
        elems=[["H"], ["N", "O", "F"], ["C", "Si", "P", "S", "Cl", "Br"]],
        bOccl=0,
    ):
        super().__init__(scan_dim, scan_window, zmin)
        self.projector.Rpp = Rpp
        self.elems = self.convert_elements(elems)
        self.all_elems = np.concatenate(self.elems)
        self.bOccl = bOccl
        self.nChan = len(elems)

    def convert_elements(self, elems):
        for el_list in elems:
            for i, e in enumerate(el_list):
                if isinstance(e, str):
                    el_list[i] = elements.ELEMENT_DICT[e][0]
        return elems

    def get_elem_channels(self, Zs):
        elem_channels = []
        for Z in Zs:
            if Z not in self.all_elems:
                raise RuntimeError(f"Element {Z} was not found in list of elements for any channel.")
            for i, c in enumerate(self.elems):
                if Z in c:
                    elem_channels.append(i)
        return elem_channels

    def eval(self, xyzqs, Zs, pot=None, rot=None):
        coefs = self.projector.makeCoefsZR(Zs, elements.ELEMENTS)
        elem_channels = self.get_elem_channels(Zs)
        pos0 = [0, 0, (xyzqs[:, 2] + coefs[:, 3]).max() + self.projector.Rpp]
        poss = self.prepare_projector(xyzqs, Zs, pos0, elem_channels=elem_channels)
        return self.projector.run_evalMultiMapSpheresElements(poss=poss, tipRot=oclr.mat3x3to4f(np.eye(3)), bOccl=self.bOccl)


class Bonds(AuxMapBase):
    """
    Generate Bonds descriptors for molecules. Bonds between atoms are represented by ellipses.

    Arguments:
        Rfunc: numpy.ndarray. Radial function of bonds&atoms potential. Converted to numpy.float32
        Rmax: float. Cutoff in angstroms for radial function. Make sure is smaller than maximum range of Rfunc - 3*drStep.
              The additional three steps are for spline interpolation.
        drStep: float. Step dx (dr) in angstroms for sampling of radial function Rfunc.
        ellipticity: float. Ratio between major and minor semiaxis.
    """

    def __init__(
        self,
        scan_dim=(128, 128),
        scan_window=((-8, -8), (8, 8)),
        zmin=-1.5,
        Rfunc=None,
        Rmax=10.0,
        drStep=0.1,
        ellipticity=0.5,
    ):
        super().__init__(scan_dim, scan_window, zmin)
        self.projector.Rmax = Rmax
        self.projector.drStep = drStep
        self.projector.elipticity = ellipticity
        if not Rfunc:
            xs = np.linspace(0.0, Rmax + 3 * drStep, int(Rmax / drStep) + 3 + 1)
            dx = xs[1] - xs[0]
            xs -= dx
            ys = np.exp(-5 * xs)
            self.projector.Rfunc = ys.astype(np.float32)
        else:
            assert len(Rfunc) * drStep >= Rmax + 3 * drStep
            self.projector.Rfunc = Rfunc.astype(np.float32)

    def eval(self, xyzqs, Zs, pot=None, rot=None):
        pos0 = [0, 0, xyzqs[:, 2].max()]
        bonds2atoms = np.array(
            findBonds_(xyzqs[:, :3], Zs.astype(np.int32), 1.2, ELEMENTS=elements.ELEMENTS),
            dtype=np.int32,
        )
        poss = self.prepare_projector(xyzqs, Zs, pos0, bonds2atoms=bonds2atoms)
        return self.projector.run_evalBondEllipses(poss=poss, tipRot=oclr.mat3x3to4f(np.eye(3)))[:, :, 0]


class AtomRfunc(AuxMapBase):
    """
    Generate AtomRfunc descriptors for molecules. Atoms are represented by disks with decay determined by Rfunc.

    Arguments:
        Rfunc: numpy.ndarray. Radial function of bonds&atoms potential. Converted to numpy.float32
        Rmax: float. Cutoff in angstroms for radial function. Make sure is smaller than maximum range of Rfunc - 3*drStep.
              The additional three steps are for spline interpolation.
        drStep: float. Step dx (dr) in angstroms for sampling of radial function Rfunc.
    """

    def __init__(
        self,
        scan_dim=(128, 128),
        scan_window=((-8, -8), (8, 8)),
        zmin=-1.5,
        Rfunc=None,
        Rmax=10.0,
        drStep=0.1,
    ):
        super().__init__(scan_dim, scan_window, zmin)
        self.projector.Rmax = Rmax
        self.projector.drStep = drStep
        if not Rfunc:
            xs = np.linspace(0.0, Rmax + 3 * drStep, int(Rmax / drStep) + 3 + 1)
            dx = xs[1] - xs[0]
            xs -= dx
            ys = np.exp(-5 * xs)
            self.projector.Rfunc = ys.astype(np.float32)
        else:
            assert len(Rfunc) * drStep >= Rmax + 3 * drStep
            self.projector.Rfunc = Rfunc.astype(np.float32)

    def eval(self, xyzqs, Zs, pot=None, rot=None):
        pos0 = [0, 0, xyzqs[:, 2].max()]
        bonds2atoms = np.array(
            findBonds_(xyzqs[:, :3], Zs.astype(np.int32), 1.2, ELEMENTS=elements.ELEMENTS),
            dtype=np.int32,
        )
        poss = self.prepare_projector(xyzqs, Zs, pos0, bonds2atoms=bonds2atoms)
        return self.projector.run_evalAtomRfunc(poss=poss, tipRot=oclr.mat3x3to4f(np.eye(3)))[:, :, 0]


aux_map_dict = {
    "vdwSpheres": vdwSpheres,
    "AtomicDisks": AtomicDisks,
    "HeightMap": HeightMap,
    "ESMap": ESMap,
    "ESMapConstant": ESMapConstant,
    "MultiMapSpheres": MultiMapSpheres,
    "Bonds": Bonds,
    "AtomRfunc": AtomRfunc,
}


class AuxMapFactory:
    def __call__(self, map_type, args={}):
        try:
            aux_map = aux_map_dict[map_type](**args)
        except KeyError:
            recognized_types = ", ".join([f"{key}" for key in aux_map_dict.keys()])
            raise ValueError(f"Unrecognized AuxMap type {map_type}. Should be one of {recognized_types}.")
        return aux_map


AuxMaps = AuxMapFactory()
