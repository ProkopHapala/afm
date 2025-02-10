#!/usr/bin/python

import os
import time
import warnings

import numpy as np
import pyopencl as cl

from .. import io
from ..common import genFFSampling
from ..defaults import d3
from ..fieldFFT import getProbeDensity
from ..HighLevel import _getAtomsWhichTouchPBCcell, subtractCoreDensities

try:
    from reikna import cluda
    from reikna.cluda import ocl_api
    from reikna.core import Annotation, Parameter, Transformation, Type
    from reikna.fft import FFT

    fft_available = True
except ModuleNotFoundError:
    fft_available = False

DEFAULT_FD_STEP = 0.05

cl_program = None
oclu = None


def init(env):
    global cl_program
    global oclu
    cl_program = env.loadProgram(env.CL_PATH / "FF.cl")
    oclu = env


verbose = 0
bRuntime = False


def makeDivisibleUp(num, divisor):
    rest = num % divisor
    if rest > 0:
        num += divisor - rest
    return num


# ========= classes


class D3Params:
    """
    pyopencl device buffer handles to Grimme-D3 parameters. Each buffer is allocated on first access.

    Arguments:
        ctx: pyopencl.Context. OpenCL context for device buffer. Defaults to oclu.ctx.
    """

    def __init__(self, ctx):
        self.ctx = ctx or oclu.ctx
        self._cl_rcov = None
        self._cl_rcut = None
        self._cl_ref_cn = None
        self._cl_ref_c6 = None
        self._cl_r4r2 = None

    def _alloc(self, buf):
        return cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=buf.astype(np.float32))

    @property
    def cl_rcov(self):
        """See :data:`.R_COV`."""
        if self._cl_rcov is None:
            self._cl_rcov = self._alloc(d3.R_COV)
        return self._cl_rcov

    @property
    def cl_rcut(self):
        """See :func:`.load_R0`."""
        if self._cl_rcut is None:
            self._cl_rcut = self._alloc(d3.load_R0())
        return self._cl_rcut

    @property
    def cl_ref_cn(self):
        """See :data:`.REF_CN`."""
        if self._cl_ref_cn is None:
            self._cl_ref_cn = self._alloc(d3.REF_CN)
        return self._cl_ref_cn

    @property
    def cl_ref_c6(self):
        """See :func:`.load_ref_c6`."""
        if self._cl_ref_c6 is None:
            self._cl_ref_c6 = self._alloc(d3.load_ref_c6())
        return self._cl_ref_c6

    @property
    def cl_r4r2(self):
        """See :data:`.R4R2`."""
        if self._cl_r4r2 is None:
            self._cl_r4r2 = self._alloc(d3.R4R2)
        return self._cl_r4r2

    def release(self):
        """Release device buffers."""
        for buf_name in ["_cl_rcov", "_cl_rcut", "_cl_ref_cn", "_cl_ref_c6", "_cl_r4r2"]:
            getattr(self, buf_name).release()
            setattr(self, buf_name, None)


class DataGrid:
    """
    Class for holding data on a grid. The data can be stored on the CPU host or an OpenCL device.

    Arguments:
        array: np.ndarray or pyopencl.Buffer. Array values on a 3D grid with possibly multiple components.
        lvec: array-like of shape (4, 3). Unit cell boundaries. First (row) vector specifies the origin,
            and the remaining three vectors specify the edge vectors of the unit cell.
        shape: array-like of length 3 or 4. Grid shape when array is a pyopencl.Buffer.
        ctx: pyopencl.Context. OpenCL context for device buffer. Defaults to oclu.ctx.
    """

    def __init__(self, array, lvec, shape=None, ctx=None):
        if isinstance(array, np.ndarray):
            if array.dtype != np.float32 or not array.flags["C_CONTIGUOUS"]:
                array = np.ascontiguousarray(array, dtype=np.float32)
            self.shape = tuple(array.shape)
            self._array = array
            self._cl_array = None
            self.nbytes = 0
        elif isinstance(array, cl.Buffer):
            if shape is None:
                raise ValueError("The shape of the grid has to be specified when the array is a pyopencl.Buffer.")
            nbytes = 4 * np.prod(shape)
            assert array.size >= nbytes, f"shape {shape} does not fit into the buffer of size {array.size}"
            self.shape = tuple(shape)
            self._array = None
            self._cl_array = array
            self.nbytes = nbytes
        else:
            raise ValueError(f"Invalid type `{type(array)}` for array.")
        if len(self.shape) not in [3, 4]:
            raise ValueError(f"Dimension of array should be 3 or 4, but got {len(self.shape)}")
        self.lvec = np.array(lvec)
        self.origin = self.lvec[0]
        assert self.lvec.shape == (4, 3), f"lvec should have shape (4, 3), but has shape {lvec.shape}"
        self.ctx = ctx or oclu.ctx

    @property
    def step(self):
        """Array of vectors pointing single steps along the grid for each lattice vector."""
        return np.stack([self.lvec[i + 1] / self.shape[i] for i in range(3)])

    @property
    def cell_vol(self):
        """The volume of a grid cell in angstrom^3."""
        a, b, c = self.step
        vol = abs(np.cross(a, b).dot(c))
        return vol

    @property
    def array(self):
        """Host array as np.ndarray. If the grid currently only exists on the device, it is copied to the host memory."""
        if self._array is None:
            self._array = np.empty(self.shape, dtype=np.float32)
            cl.enqueue_copy(oclu.queue, self._array, self._cl_array)
        return self._array

    @property
    def cl_array(self):
        """Device array as pyopencl.buffer. If the grid currently only exists on the host, it is copied to the device memory."""
        if self._cl_array is None:
            self._cl_array = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.array)
            self.nbytes += 4 * np.prod(self.shape)
            if verbose > 0:
                print(f"DataGrid.nbytes {self.nbytes}")
        return self._cl_array

    def update_array(self, array, lvec):
        """
        Update array contents. If the new array is the same size or smaller than the current array, the data is updated
        without a reallocation on the device.
        """
        if array.dtype != np.float32 or not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array, dtype=np.float32)
        if self._cl_array is not None:
            current_size = np.prod(self.shape)
            if array.size > current_size:
                if verbose > 0:
                    print(f"Reallocating buffers. Old size = {current_size}, new size = {array.size}")
                self._cl_array = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, 4 * array.size)
                self.nbytes += 4 * (array.size - current_size)
            self._enqueue_event = cl.enqueue_copy(oclu.queue, self._cl_array, array, is_blocking=False)
        self._array = array
        self.lvec = lvec
        self.shape = tuple(array.shape)

    def release(self, keep_on_host=True):
        """Release device buffer.

        Arguments:
        keep_on_host: bool. If the grid currently only exists on the device, it is copied to the host memory before release."""
        if self._cl_array is not None:
            if keep_on_host:
                self.array
            self._cl_array.release()
            self._cl_array = None
            self.nbytes -= 4 * np.prod(self.shape)

    @classmethod
    def from_file(cls, file_path, scale=1.0):
        """
        Load grid data and atoms from a .cube or .xsf file.

        Arguments:
            file_path: str. Path to file to load.
            scale: float. Scaling factor for the returned data grid values.

        Returns:
            data: class type. Data grid object.
            xyzs: np.ndarray of shape (num_atoms, 3). Atom coordinates.
            Zs: np.ndarray of shape (num_atoms,). Atomic numbers.
        """

        file_path = str(file_path)
        if file_path.endswith(".cube"):
            data, lvec, _, _ = io.loadCUBE(file_path, xyz_order=True, verbose=False)
            Zs, x, y, z, _ = io.loadAtomsCUBE(file_path)
        elif file_path.endswith(".xsf"):
            data, lvec, _, _ = io.loadXSF(file_path, xyz_order=True, verbose=False)
            try:
                (Zs, x, y, z, _), _, _ = io.loadXSFGeom(file_path)
            except ValueError:
                warnings.warn(f"Could not read geometry from {file_path} in DataGrid.from_file.")
                Zs = np.zeros(1)
                x = y = z = np.zeros((1, 1))
        else:
            raise ValueError(f"Unsupported file format in file `{file_path}`")

        if not np.allclose(scale, 1.0):
            data *= scale
        data = cls(data, lvec)
        xyzs = np.stack([x, y, z], axis=1)
        Zs = np.array(Zs)

        return data, xyzs, Zs

    def to_file(self, file_path, clamp=None):
        """
        Save data grid to file(s).

        Supported file types are .xsf and .npy.

        Arguments:
            file_path: str. Path to saved file. For a 4D data grid, letters x, y, z, w are appended
                to the file path for each component, respectively.
            clamp: float or None. If not None, all values greater than this are clamped to this value.
        """
        file_head, ext = os.path.splitext(file_path)
        if ext not in [".xsf", ".npy"]:
            raise ValueError(f"Unsupported file extension `{ext}` for saving data grid.")
        ext = ext[1:]
        array = self.array.copy()
        if len(self.shape) == 3:
            if clamp:
                array[array > clamp] = clamp
            io.save_scal_field(file_head, array.T, self.lvec, data_format=ext)
        if len(self.shape) == 4:
            assert self.shape[3] == 4, "Wrong number of components"
            if clamp:
                io.limit_vec_field(array, Fmax=clamp)
                array[:, :, :, 3][array[:, :, :, 3] > clamp] = clamp
            array = array.transpose(2, 1, 0, 3)
            io.save_vec_field(file_head, array[:, :, :, :3], self.lvec, data_format=ext)
            io.save_scal_field(file_head + "_w", array[:, :, :, 3], self.lvec, data_format=ext)

    def _prepare_same_size_output_grid(self, array_in, in_place):
        if in_place:
            grid_out = self
            self._array = None  # The current host array will be wrong after operation, so reset it
        else:
            array_out = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=array_in.size)
            array_type = type(self)  # This way so inherited classes return their own class type
            grid_out = array_type(array_out, lvec=self.lvec, shape=self.shape, ctx=self.ctx)
        return grid_out

    def clamp(self, minimum=-np.inf, maximum=np.inf, clamp_type="hard", soft_clamp_width=1.0, in_place=True, local_size=(32,), queue=None):
        """
        Clamp data grid values to a specified range. The ``'hard'`` clamp simply clips values that are out of range,
        and the ``'soft'`` clamp uses a sigmoid to smoothen the transition.

        Arguments:
            minimum: float. Values below minimum are set to minimum.
            maximum: float. Values above maximum are set to maximum.
            clamp_type: str. Type of clamp to use: ``'soft'`` or ``'hard'``.
            soft_clamp_width: float. Width of transition region for soft clamp.
            in_place: bool. Whether to do operation in place or to create a new array.
            local_size: tuple of a single int. Size of local work group on device.
            queue: pyopencl.CommandQueue. OpenCL queue on which operation is performed. Defaults to oclu.queue.

        Returns:
            grid_out: Same type as self. New data grid with result.
        """

        array_in = self.cl_array
        grid_out = self._prepare_same_size_output_grid(array_in, in_place)
        n = np.int32(array_in.size / 4)
        minimum = np.float32(minimum)
        maximum = np.float32(maximum)
        soft_clamp_width = np.float32(soft_clamp_width)

        queue = queue or oclu.queue
        global_size = [int(np.ceil(n / local_size[0]) * local_size[0])]

        if clamp_type == "hard":
            # fmt: off
            cl_program.clamp_hard(queue, global_size, local_size,
                array_in,
                grid_out.cl_array,
                n,
                minimum,
                maximum,
            )
            # fmt: on
        elif clamp_type == "soft":
            # fmt: off
            cl_program.clamp_soft(queue, global_size, local_size,
                array_in,
                grid_out.cl_array,
                n,
                minimum,
                maximum,
                soft_clamp_width,
            )
            # fmt: on
        else:
            raise ValueError(f"Unsupported clamp type `{clamp_type}`")

        return grid_out

    def add_mult(self, array, scale=1.0, in_place=True, local_size=(32,), queue=None):
        """
        Multiply the values of another data grid and add them to the values of this data grid.

        Arguments:
            array: DataGrid. Grid whose values to scale and add.
            scale: float. Value by which values in array are multiplied.
            in_place: bool. Whether to do operation in place or to create a new array.
            local_size: tuple of a single int. Size of local work group on device.
            queue: pyopencl.CommandQueue. OpenCL queue on which operation is performed.
                Defaults to oclu.queue.

        Returns:
            grid_out: Same type as self. New data grid with result.
        """

        array_in1 = self.cl_array
        array_in2 = array.cl_array
        grid_out = self._prepare_same_size_output_grid(array_in1, in_place)
        n = np.int32(array_in1.size / 4)
        scale = np.float32(scale)

        queue = queue or oclu.queue
        global_size = [int(np.ceil(n / local_size[0]) * local_size[0])]

        # fmt: off
        cl_program.addMult(queue, global_size, local_size,
            array_in1,
            array_in2,
            grid_out.cl_array,
            n,
            scale
        )
        # fmt: on
        return grid_out

    def power_positive(self, p=1.2, normalize=True, in_place=True, local_size=(32,), queue=None):
        """
        Raise every positive element in the grid into a power. Negative values are set to zero.

        Arguments:
            p: float. Power to rise to.
            normalize: bool. Whether to normalize the values after setting negative values to zero.
                The normalization is done by scaling the values such that the total sum of the values
                in the array remains unchanged after eliminating the negative values.
            in_place: bool. Whether to do operation in place or to create a new array.
            local_size: tuple of a single int. Size of local work group on device.
            queue: pyopencl.CommandQueue. OpenCL queue on which operation is performed.
                Defaults to oclu.queue.

        Returns:
            grid_out: Same type as self. New data grid with result.
        """

        if bRuntime:
            t0 = time.perf_counter()

        array_in = self.cl_array
        grid_out = self._prepare_same_size_output_grid(array_in, in_place)
        n = np.int32(array_in.size / 4)
        p = np.float32(p)
        if normalize:
            scale = self._get_normalization_factor(queue)
            assert scale > 0, "Normalizing scaling factor should be positive."
            scale = np.float32(scale) ** p
        else:
            scale = np.float32(1.0)

        queue = queue or oclu.queue
        global_size = [int(np.ceil(n / local_size[0]) * local_size[0])]

        # fmt: off
        cl_program.power(queue, global_size, local_size,
            array_in,
            grid_out.cl_array,
            n,
            p,
            scale
        )
        # fmt: on

        if bRuntime:
            print("runtime(DataGrid.power_positive) [s]: ", time.perf_counter() - t0)

        return grid_out

    def _get_normalization_factor(self, queue=None):
        queue = queue or oclu.queue
        n = np.int32(np.prod(self.shape))
        local_size = (256,)
        n_groups = np.int32(min(local_size[0], (n - 1) // local_size[0] + 1))
        global_size = (local_size[0] * n_groups,)
        array_in = self.cl_array
        array_out = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=8 * n_groups)
        # First do sums of the input array within each work group...
        # fmt: off
        cl_program.normalizeSumReduce(queue, global_size, local_size,
            array_in,
            array_out,
            n
        )
        # fmt: on
        # ... then sum the results of the first kernel call
        cl_program.sumSingleGroup(queue, local_size, local_size, array_out, n_groups)
        # Now the first element of array_out holds the final answer
        sums = np.empty((2,), dtype=np.float32)
        cl.enqueue_copy(queue, sums, array_out)
        return sums[0] / sums[1]

    def grad(self, scale=1.0, array_out=None, order="C", local_size=(32,), queue=None):
        """Get the centered finite difference gradient of the data grid. Uses periodic boundary conditions
        at the edge of the grid.

        The datagrid has to be either 3D, or 4D with self.shape[3] == 1.

        The resulting array adds a 4th dimension with size 4 to the grid such that at indices
        0-2 are the partial derivatives in x, y, and z directions, respectively, and at index
        3 is the original scalar field.

        Arguments:
            scale: float or array-like of size 4. Additional scaling factor for the output.
            array_out: pyopencl.Buffer or None. Output array. If None, then is created automatically.
            order: str, 'C' or 'F'. Whether to save values in C or Fortran order.
            local_size: tuple of a single int. Size of local work group on device.
            queue: pyopencl.CommandQueue. OpenCL queue on which operation is performed.
                Defaults to oclu.queue.

        Returns:
            grid_out: :class:`DataGrid`. New data grid with result.
        """

        if len(self.shape) == 4 and self.shape[3] > 1:
            raise RuntimeError(f"Can only take gradient of a datagrid with a single component.")

        if isinstance(scale, float):
            scale = scale * np.ones(4, dtype=np.float32)
        else:
            scale = np.array(scale, dtype=np.float32)
            if len(scale) != 4:
                raise ValueError(f"Scale should have length 4, but got {len(scale)}.")

        if not order in ["C", "F"]:
            raise ValueError(f"Unknown data order `{order}`")
        order = np.int32(1) if order == "C" else np.int32(0)

        array_in = self.cl_array
        queue = queue or oclu.queue

        shape_out = self.shape[:3] + (4,)
        if array_out is None:
            array_out = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=array_in.size * 4)
        else:
            assert array_out.size == array_in.size * 4, f"array size does not match ({array_out.size} != {array_in.size * 4})"
        grid_out = DataGrid(array_out, lvec=self.lvec, shape=shape_out, ctx=self.ctx)

        global_size = [int(np.ceil(np.prod(self.shape) / local_size[0]) * local_size[0])]
        step = np.append(np.diag(self.step), 0).astype(np.float32)
        # fmt: off
        cl_program.grad(queue, global_size, local_size,
            array_in,
            array_out,
            np.array(shape_out, dtype=np.int32),
            step,
            order,
            scale
        )
        # fmt: on

        return grid_out

    def interp_at(self, lvec_new, shape_new, array_out=None, rot=np.eye(3), rot_center=np.zeros(3), local_size=(32,), queue=None):
        """
        Interpolate grid values onto another grid. Uses periodic boundary conditions.

        Arguments:
            lvec_new: array-like of shape (4, 3). Unit cell boundaries for new grid.
            shape_new: array-like of length 3. New grid shape.
            array_out: pyopencl.Buffer or None. Output array. If None, then is created automatically.
            rot: np.ndarray of shape (3, 3). Rotation matrix to apply.
            rot_center: np.ndarray of shape (3,). Point around which rotation is performed.
            local_size: tuple of a single int. Size of local work group on device.
            queue: pyopencl.CommandQueue. OpenCL queue on which operation is performed.
                Defaults to oclu.queue.

        Returns:
            grid_out: Same type as self. New data grid with result.
        """

        if len(self.shape) == 4 and self.shape[3] > 1:
            raise NotImplementedError("Interpolation for 4D grids is not implemented.")

        queue = queue or oclu.queue

        size_new = 4 * np.prod(shape_new[:3])
        if array_out is None:
            array_out = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=size_new)
        else:
            assert array_out.size == size_new, f"array size does not match ({array_out.size} != {size_new})"
        array_out = type(self)(array_out, lvec_new, shape_new, self.ctx)

        global_size = [int(np.ceil(np.prod(shape_new[:3]) / local_size[0]) * local_size[0])]
        T = np.concatenate([np.linalg.inv(self.step).T.copy(), np.zeros((3, 1))], axis=1, dtype=np.float32)
        rot = np.concatenate([rot, np.zeros((3, 1))], axis=1, dtype=np.float32)
        dlvec = np.concatenate([array_out.step, np.zeros((3, 1))], axis=1, dtype=np.float32)

        # fmt: off
        cl_program.interp_at(queue, global_size, local_size,
            self.cl_array,
            array_out.cl_array,
            np.append(self.shape, 0).astype(np.int32),
            T[0],
            T[1],
            T[2],
            np.append(self.origin, 0).astype(np.float32),
            np.append(shape_new, 0).astype(np.int32),
            dlvec[0],
            dlvec[1],
            dlvec[2],
            np.append(array_out.origin, 0).astype(np.float32),
            rot[0],
            rot[1],
            rot[2],
            np.append(rot_center, 0).astype(np.float32),
        )
        # fmt: on

        return array_out


# Aliases for DataGrid
class HartreePotential(DataGrid):
    """Sample Hartree potential. Units should be in Volts."""


class ElectronDensity(DataGrid):
    """Sample electron density. Units should be in e/Å^3."""


class TipDensity(DataGrid):
    """Tip electron density. Units should be in e/Å^3."""

    def subCores(self, xyzs, Zs, Rcore=0.7, valElDict=None):
        """
        Subtract core densities from the tip density.

        Arguments:
            xyzs: np.ndarray of shape (n_atoms, 3). Coordinates of atoms.
            Zs: np.ndarray of shape (n_atoms,). Atomic numbers of atoms.
            Rcore: float. Width of core density distribution.
            valElDict: Dict or None. Dictionary of the number of valence electrons for elements.
                If None, then values in defaults.valelec_dict are used.

        Returns:
            TipDensity. New tip density with core densities subtracted.
        """
        array = np.ascontiguousarray(self.array.T, dtype=np.float64)  # 64 bit required by library
        Rs, elems = _getAtomsWhichTouchPBCcell(xyzs.T, Zs, self.shape, self.lvec, 1.0, False)
        subtractCoreDensities(array, self.lvec, elems=elems, Rs=Rs, valElDict=valElDict, Rcore=Rcore, bSaveDebugGeom=False)
        grid = TipDensity(array.T, self.lvec, ctx=self.ctx)
        return grid

    def interp_at(self, lvec_new, shape_new, array_out=None, local_size=(32,), queue=None):
        """
        Interpolate tip density onto a new grid. The tip is assumed to be cented on the origin,
        so the resizing of the grid happens in the middle of the grid and the corners remain
        fixed (the origin coordinates of the grids are ignored in the transformation).

        Arguments:
            lvec_new: array-like of shape (4, 3). Unit cell boundaries for new grid.
            shape_new: array-like of length 3. New grid shape.
            array_out: pyopencl.Buffer or None. Output array. If None, then is created automatically.
            local_size: tuple of a single int. Size of local work group on device.
            queue: pyopencl.CommandQueue. OpenCL queue on which operation is performed.
                Defaults to oclu.queue.

        Returns:
            grid_out: :class:`TipDensity`. New tip density grid.
        """

        if len(self.shape) == 4 and self.shape[3] > 1:
            raise NotImplementedError("Interpolation for 4D grids is not implemented.")

        queue = queue or oclu.queue

        size_new = 4 * np.prod(shape_new[:3])
        if array_out is None:
            array_out = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=size_new)
        else:
            assert array_out.size == size_new, f"array size does not match ({array_out.size} != {size_new})"
        array_out = TipDensity(array_out, lvec_new, shape_new, self.ctx)

        global_size = [int(np.ceil(np.prod(shape_new[:3]) / local_size[0]) * local_size[0])]
        T = np.concatenate([np.linalg.inv(self.step).T.copy(), np.zeros((3, 1))], axis=1, dtype=np.float32)
        lvec_in_inv = np.concatenate([np.linalg.inv(self.lvec[1:]), np.zeros((3, 1))], axis=1, dtype=np.float32)
        dlvec_out = np.concatenate([array_out.step, np.zeros((3, 1))], axis=1, dtype=np.float32)

        # fmt: off
        cl_program.interp_tip_at(queue, global_size, local_size,
            self.cl_array,
            array_out.cl_array,
            np.append(self.shape, 0).astype(np.int32),
            T[0],
            T[1],
            T[2],
            lvec_in_inv[0],
            lvec_in_inv[1],
            lvec_in_inv[2],
            np.append(shape_new, 0).astype(np.int32),
            dlvec_out[0],
            dlvec_out[1],
            dlvec_out[2],
        )
        # fmt: on

        return array_out


class MultipoleTipDensity(TipDensity):
    """
    Multipole probe tip charge density on a periodic grid.

    Arguments:
        lvec: np.ndarray of shape (3, 3). Grid lattice vectors.
        shape: array-like of length 3. Grid shape.
        center: array-like of length 3. Center position of charge density in the grid.
        sigma: float. Width of charge distribution.
        multipole: Dict. Charge multipole types. The dict should contain float entries for at least
            of one the following 's', 'px', 'py', 'pz', 'dz2', 'dy2', 'dx2', 'dxy' 'dxz', 'dyz'.
            The tip charge density will be a linear combination of the specified multipole types
            with the specified weights.
        tilt: float. Tip charge tilt angle in radians.
        ctx: pyopencl.Context. OpenCL context for device buffer. Defaults to oclu.ctx.
    """

    def __init__(self, lvec, shape, center=[0, 0, 0], sigma=0.71, multipole={"dz2": -0.1}, tilt=0.0, ctx=None):
        array = self._make_tip_density(lvec, shape, center, sigma, multipole, tilt)
        lvec = np.concatenate([[[0, 0, 0]], lvec], axis=0)
        super().__init__(array, lvec, ctx)

    def _make_tip_density(self, lvec, shape, center, sigma, multipole, tilt):
        if bRuntime:
            t0 = time.perf_counter()

        lvec_len = np.linalg.norm(lvec, axis=1)
        center = np.array(center)
        if (center < 0).any() or (center > lvec.sum(axis=0)).any():
            raise ValueError("Center position is outside the grid.")

        xyz = []
        for i in range(3):
            c = np.linspace(0, lvec_len[i] * (1 - 1 / shape[i]), shape[i]) - center[i]
            c[c >= lvec_len[i] / 2] -= lvec_len[i]
            c[c <= -lvec_len[i] / 2] += lvec_len[i]
            xyz.append(c)
        X, Y, Z = np.meshgrid(*xyz, indexing="ij")
        step = lvec_len / shape
        rho = getProbeDensity(lvec, X, Y, Z, step, sigma=sigma, multipole_dict=multipole, tilt=tilt)

        if bRuntime:
            print("runtime(MultipoleTipDensity._make_tip_density) [s]: ", time.perf_counter() - t0)

        return rho.astype(np.float32)


class FFTCrossCorrelation:
    """
    Do circular cross-correlation of sample Hartree potential or electron density with tip charge
    density via FFT.

    Arguments:
        rho: :class:`TipDensity`. Tip charge density.
        queue: pyopencl.CommandQueue. OpenCL queue on which operations are performed.
            Defaults to oclu.queue.
    """

    def __init__(self, rho, queue=None):
        if not fft_available:
            raise RuntimeError("Cannot do FFT because reikna is not installed.")
        self.shape = rho.array.shape
        self.queue = queue or oclu.queue
        self.ctx = self.queue.context
        self.nbytes = 0
        self._make_transforms()
        self._make_fft()
        self._set_rho(rho)
        if verbose > 0:
            print(f"FFTCrossCorrelation.nbytes {self.nbytes}")

    # https://github.com/fjarri/reikna/issues/57
    def _make_transforms(self):
        # fmt: off
        self.r2c = Transformation(
            [
                Parameter('output', Annotation(Type(np.complex64, self.shape), 'o')),
                Parameter('input',  Annotation(Type(np.float32,   self.shape), 'i'))
            ],
            """
            ${output.store_same}(
                COMPLEX_CTR(${output.ctype})(
                    ${input.load_same},
                    0));
            """,
        )
        self.c2r = Transformation(
            [
                Parameter("output", Annotation(Type(np.float32,   self.shape), "o")),
                Parameter("input",  Annotation(Type(np.complex64, self.shape), "i")),
                Parameter("scale",  Annotation(np.float32)),
            ],
            """
            ${output.store_same}(${input.load_same}.x * ${scale});
            """,
        )
        self.conj_mult = Transformation(
            [
                Parameter("input1", Annotation(Type(np.complex64, self.shape), "i")),
                Parameter("input2", Annotation(Type(np.complex64, self.shape), "i")),
                Parameter("output", Annotation(Type(np.complex64, self.shape), "o")),
            ],
            """
            ${output.store_same}(${MUL}(${CONJ}(${input1.load_same}), ${input2.load_same}));
            """,
            render_kwds={
                'CONJ': cluda.functions.conj(np.complex64),
                'MUL': cluda.functions.mul(np.complex64, np.complex64)
            }
        )
        # fmt: on

    def _make_fft(self):
        if bRuntime:
            t0 = time.perf_counter()

        thr = ocl_api().Thread(self.queue)
        size = 8 * np.prod(self.shape)
        self.pot_hat_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=size)
        self.rho_hat_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=size)
        self.nbytes += 2 * size

        fft_f = FFT(self.r2c.output)
        fft_f.parameter.input.connect(self.r2c, self.r2c.output, new_input=self.r2c.input)
        self.fft_f = fft_f.compile(thr)

        fft_i = FFT(self.c2r.input)
        fft_i.parameter.input.connect(self.conj_mult, self.conj_mult.output, new_input1=self.conj_mult.input1, new_input2=self.conj_mult.input2)
        fft_i.parameter.output.connect(self.c2r, self.c2r.input, new_output=self.c2r.output, scale=self.c2r.scale)
        self.fft_i = fft_i.compile(thr)

        if bRuntime:
            print("runtime(FFTCrossCorrelation._make_fft) [s]: ", time.perf_counter() - t0)

    def _set_rho(self, rho):
        self.rho = rho
        self.fft_f(self.rho_hat_cl, rho.cl_array, inverse=0)

    def correlate(self, array, E=None, scale=1):
        """
        Cross-correlate input array with tip charge density.

        Arguments:
            array: :class:`DataGrid` or pyopencl.Buffer. Sample potential/density
                to cross-correlate with tip density. Has to be the same shape as rho.
            E: :class:`DataGrid` or None. Output data grid. If None, is created automatically.
                The automatically created datagrid has the same lvec as self.rho.lvec.
            scale: float. Additional scaling factor for the output.

        Returns:
            E: :class:`DataGrid`. Result of cross-correlation.
        """

        if bRuntime:
            t0 = time.perf_counter()

        if isinstance(array, DataGrid):
            array = array.cl_array
        assert array.size == np.prod(self.shape) * 4, f"array size {array.size} does not match rho array shape {self.shape}"

        if E is None:
            E_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=4 * np.prod(self.shape))
            E = DataGrid(E_cl, lvec=self.rho.lvec, shape=self.shape, ctx=self.rho.ctx)
        else:
            assert E.shape == self.shape, "E data grid shape does not match"
            E_cl = E.cl_array

        if bRuntime:
            self.queue.finish()
            print("runtime(FFTCrossCorrelation.correlate.pre) [s]: ", time.perf_counter() - t0)

        # Do cross-correlation
        self.fft_f(output=self.pot_hat_cl, new_input=array, inverse=0)
        self.fft_i(new_output=E_cl, new_input1=self.rho_hat_cl, new_input2=self.pot_hat_cl, scale=scale * self.rho.cell_vol, inverse=1)

        if bRuntime:
            self.queue.finish()
            print("runtime(FFTCrossCorrelation.correlate) [s]: ", time.perf_counter() - t0)

        return E


class ForceField_LJC:
    """Evaluate Lennard-Jones based force fields on an OpenCL device."""

    verbose = 0

    def __init__(self):
        self.ctx = oclu.ctx
        self.queue = oclu.queue
        self.d3_params = D3Params(self.ctx)
        self.cl_poss = None
        self.cl_FE = None
        self.cl_Efield = None
        self.pot = None
        self.rho = None
        self.rho_delta = None
        self.rho_sample = None

    def initSampling(self, lvec, pixPerAngstrome=10, nDim=None):
        if nDim is None:
            nDim = genFFSampling(lvec, pixPerAngstrome=pixPerAngstrome)
        self.nDim = nDim
        self.setLvec(lvec, nDim=nDim)

    def initPoss(self, poss=None, nDim=None, lvec=None, pixPerAngstrome=10):
        if poss is None:
            self.initSampling(lvec, pixPerAngstrome=10, nDim=None)
        self.prepareBuffers(poss=poss)

    def setLvec(self, lvec, nDim=None):
        if nDim is not None:
            self.nDim = np.array([nDim[0], nDim[1], nDim[2], 4], dtype=np.int32)
        elif self.nDim is not None:
            nDim = self.nDim
        else:
            raise RuntimeError("nDim must be set somewhere")
        self.lvec0 = np.zeros(4, dtype=np.float32)
        self.lvec = np.zeros((3, 4), dtype=np.float32)
        self.dlvec = np.zeros((3, 4), dtype=np.float32)
        self.lvec0[:3] = lvec[0, :3]
        self.lvec[:, :3] = lvec[1:4, :3]
        self.dlvec[0, :] = self.lvec[0, :] / nDim[0]
        self.dlvec[1, :] = self.lvec[1, :] / nDim[1]
        self.dlvec[2, :] = self.lvec[2, :] / nDim[2]

    def setQs(self, Qs=[100, -200, 100, 0], QZs=[0.1, 0, -0.1, 0]):
        if (len(Qs) != 4) or (len(QZs) != 4):
            raise ValueError("Qs and Qzs must have length 4")
        self.Qs = np.array(Qs, dtype=np.float32)
        self.QZs = np.array(QZs, dtype=np.float32)

    def setPP(self, Z_pp):
        """Set the atomic number of the probe particle. Required for calculating DFT-D3 parameters."""
        self.iZPP = np.int32(Z_pp)

    def prepareBuffers(
        self,
        atoms=None,
        cLJs=None,
        REAs=None,
        Zs=None,
        poss=None,
        bDirect=False,
        nz=20,
        pot=None,
        E_field=False,
        rho=None,
        rho_delta=None,
        rho_sample=None,
        minimize_memory=False,
    ):
        """Allocate all necessary buffers in device memory."""

        if bRuntime:
            self.queue.finish()
            t0 = time.perf_counter()

        nbytes = 0
        mf = cl.mem_flags
        nb_float = np.dtype(np.float32).itemsize
        lvec = np.concatenate([self.lvec0[None, :3], self.lvec[:, :3]], axis=0)

        if atoms is not None:
            self.nAtoms = np.int32(len(atoms))
            atoms = atoms.astype(np.float32)
            self.cl_atoms = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=atoms)
            nbytes += atoms.nbytes
        if cLJs is not None:
            cLJs = cLJs.astype(np.float32)
            self.cl_cLJs = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cLJs)
            nbytes += cLJs.nbytes
        if REAs is not None:
            REAs = REAs.astype(np.float32)
            self.cl_REAs = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=REAs)
            nbytes += REAs.nbytes
        if Zs is not None:
            self.Zs = np.array(Zs, dtype=np.int32)
            self.cl_Zs = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.Zs)
            nbytes += self.Zs.nbytes
        if poss is not None:
            self.nDim = np.array(poss.shape, dtype=np.int32)
            self.cl_poss = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=poss)
            nbytes += poss.nbytes  # float4
        if (self.cl_FE is None) and not bDirect:
            nb = self.nDim[0] * self.nDim[1] * self.nDim[2] * 4 * nb_float
            self.cl_FE = cl.Buffer(self.ctx, mf.WRITE_ONLY, nb)
            nbytes += nb
            if self.verbose > 0:
                print(" forcefield.prepareBuffers() :  self.cl_FE  ", self.cl_FE)
        if pot is not None:
            assert isinstance(pot, HartreePotential), "pot should be a HartreePotential object"
            self.pot = pot
            self.pot.cl_array  # Accessing the cl_array attribute copies the pot to the device
        if E_field:
            self.cl_Efield = cl.Buffer(self.ctx, mf.READ_WRITE, size=4 * np.prod(self.nDim))
            nbytes += 4 * np.prod(self.nDim)
        if rho is not None:
            assert isinstance(rho, TipDensity), "rho should be a TipDensity object"
            self.rho = rho
            if not (np.allclose(self.rho.lvec, lvec) and np.allclose(self.rho.shape, self.nDim[:3])):
                self.rho = self.rho.interp_at(lvec, self.nDim[:3])
            if hasattr(self, "fft_corr") and np.allclose(self.rho.shape, self.fft_corr.shape):
                # We have an existing FFT prepared and it has the same shape as the new one, so we only need to update the array.
                self.fft_corr._set_rho(self.rho)
            else:
                self.fft_corr = FFTCrossCorrelation(self.rho)
            if minimize_memory:
                self.rho.release()  # We don't actually need this on device, only the FFT array
        if rho_delta is not None:
            assert isinstance(rho_delta, TipDensity), "rho_delta should be a TipDensity object"
            self.rho_delta = rho_delta
            if not (np.allclose(self.rho_delta.lvec, lvec) and np.allclose(self.rho_delta.shape, self.nDim[:3])):
                self.rho_delta = self.rho_delta.interp_at(lvec, self.nDim[:3])
            # self.fft_corr_delta = FFTCrossCorrelation(self.rho_delta)
            if hasattr(self, "fft_corr_delta") and np.allclose(self.rho_delta.shape, self.fft_corr_delta.shape):
                # We have an existing FFT prepared and it has the same shape as the new one, so we only need to update the array.
                self.fft_corr_delta._set_rho(self.rho_delta)
            else:
                self.fft_corr_delta = FFTCrossCorrelation(self.rho_delta)
            if minimize_memory:
                self.rho_delta.release()  # We don't actually need this on device, only the FFT array
        if rho_sample is not None:
            assert isinstance(rho_sample, ElectronDensity), "rho_sample should be an ElectronDensity object"
            self.rho_sample = rho_sample
            self.rho_sample.cl_array

        if self.verbose > 0:
            print("ForceField_LJC.prepareBuffers.nbytes", nbytes)

        if bRuntime:
            self.queue.finish()
            print("runtime(ForceField_LJC.prepareBuffers) [s]: ", time.perf_counter() - t0)

    def updateBuffers(self, atoms=None, cLJs=None, poss=None):
        """Update the content of device buffers."""
        if self.verbose > 0:
            print(" ForceField_LJC.updateBuffers ")
        oclu.updateBuffer(atoms, self.cl_atoms)
        oclu.updateBuffer(cLJs, self.cl_cLJs)
        oclu.updateBuffer(poss, self.cl_poss)

    def tryReleaseBuffers(self):
        """Release all device buffers."""
        if self.verbose > 0:
            print(" ForceField_LJC.tryReleaseBuffers ")
        try:
            self.cl_atoms.release()
            self.cl_atoms = None
        except:
            pass
        try:
            self.cl_cLJs.release()
            self.cl_cLJs = None
        except:
            pass
        try:
            self.cl_REAs.release()
            self.cl_REAs = None
        except:
            pass
        try:
            self.cl_poss.release()
            self.cl_poss = None
        except:
            pass
        try:
            self.cl_FE.release()
            self.cl_FE = None
        except:
            pass
        try:
            self.pot.release()
        except:
            pass
        try:
            self.cl_Efield.release()
            self.cl_Efield = None
        except:
            pass
        try:
            self.rho.release()
        except:
            pass
        try:
            self.rho_delta.release()
        except:
            pass
        try:
            self.rho_sample.release()
        except:
            pass

    def downloadFF(self, FE=None):
        """
        Get the force field array from the device.

        Arguments:
            FE: np.ndarray or None. Array where output force field is copied to. If None,
                will be created automatically.

        Returns:
            FE: np.ndarray. Force field and energy.
        """

        # Get numpy array
        if FE:
            if not np.allclose(FE.shape, self.nDim):
                raise ValueError(f"FE array dimensions {FE.shape} do not match with " f"force field dimensions {self.nDim}.")

            # Values are saved in Fortran order with the xyzw dimensions as the first index
            FE = FE.transpose(3, 0, 1, 2)
            if not FE.flags["F_CONTIGUOUS"]:
                FE = np.asfortranarray(FE)

        else:
            FE = np.empty((self.nDim[3],) + tuple(self.nDim[:3]), dtype=np.float32, order="F")

        if self.verbose:
            print("FE.shape ", FE.shape)

        # Copy from device to host
        cl.enqueue_copy(self.queue, FE, self.cl_FE)
        self.queue.finish()

        # Transpose xyzw dimension back to last index
        FE = FE.transpose(1, 2, 3, 0)

        return FE

    def initialize(self, value=0, bFinish=False):
        """Initialize the force field to a constant value.

        Arguments:
            value: float. Value assigned to every element of the force field grid.
            bFinish: Bool. Whether to wait for execution to finish.
        """
        cl.enqueue_copy(self.queue, self.cl_FE, np.full(self.nDim.prod(), value, dtype=np.float32))
        if bFinish:
            self.queue.finish()

    def run_evalLJ_noPos(self, FE=None, local_size=(32,), bCopy=True, bFinish=True):
        """
        Compute Lennard-Jones forcefield without charges at grid points.

        Arguments:
            FE: np.ndarray or None. Array where output force field is copied to if bCopy == True.
                If None and bCopy == True, will be created automatically.
            local_size: tuple of a single int. Size of local work group on device.
            bCopy: Bool. Whether to return the calculated electric field to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns:
            FE: np.ndarray if bCopy == True or None otherwise. Calculated force field and energy.
        """

        if bRuntime:
            t0 = time.perf_counter()

        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        cl_program.evalLJ_noPos(
            self.queue, global_size, local_size, self.nAtoms, self.cl_atoms, self.cl_cLJs, self.cl_FE, self.nDim, self.lvec0, self.dlvec[0], self.dlvec[1], self.dlvec[2]
        )

        if bCopy:
            FE = self.downloadFF(FE)
        if bFinish:
            self.queue.finish()
        if bRuntime:
            print("runtime(ForceField_LJC.run_evalLJ_noPos) [s]: ", time.perf_counter() - t0)

        return FE

    def run_evalLJC_QZs_noPos(self, FE=None, local_size=(32,), bCopy=True, bFinish=True):
        """Compute Lennard-Jones force field with several point-charges separated on the z-axis."""
        if bRuntime:
            t0 = time.time()
        if bCopy and (FE is None):
            ns = tuple(self.nDim[:3]) + (4,)
            FE = np.zeros(ns, dtype=np.float32)
            if self.verbose > 0:
                print("FE.shape", FE.shape, self.nDim)
        ntot = self.nDim[0] * self.nDim[1] * self.nDim[2]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        if bRuntime:
            print("runtime(ForceField_LJC.run_evalLJC_QZs_noPos.pre) [s]: ", time.time() - t0)
        # fmt: off
        cl_program.evalLJC_QZs_noPos(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_FE,
            self.nDim,
            self.lvec0,
            self.dlvec[0],
            self.dlvec[1],
            self.dlvec[2],
            self.Qs,
            self.QZs
        )
        # fmt: on
        if bCopy:
            cl.enqueue_copy(self.queue, FE, self.cl_FE)
        if bFinish:
            self.queue.finish()
        if bRuntime:
            print("runtime(ForceField_LJC.run_evalLJC_QZs_noPos) [s]: ", time.time() - t0)
        return FE

    def run_evalLJC_Hartree(self, FE=None, local_size=(32,), bCopy=True, bFinish=True):
        """
        Compute Lennard Jones force field at grid points and add to it the electrostatic force
        from an electric field precomputed from a Hartree potential.

        Arguments:
            FE: np.ndarray or None. Array where output force field is copied to if bCopy == True.
                If None and bCopy == True, will be created automatically.
            local_size: tuple of a single int. Size of local work group on device.
            bCopy: Bool. Whether to return the calculated forcefield to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns:
            FE: np.ndarray if bCopy == True or None otherwise. Calculated force field and energy.
        """

        if bRuntime:
            t0 = time.perf_counter()

        T = np.append(np.linalg.inv(self.dlvec[:, :3]).T.copy(), np.zeros((3, 1)), axis=1).astype(np.float32)

        if bRuntime:
            print("runtime(ForceField_LJC.run_evalLJC_Hartree.pre) [s]: ", time.perf_counter() - t0)

        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        cl_program.evalLJC_Hartree(
            self.queue,
            global_size,
            local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_Efield,
            self.cl_FE,
            self.nDim,
            self.lvec0,
            self.dlvec[0],
            self.dlvec[1],
            self.dlvec[2],
            T[0],
            T[1],
            T[2],
            self.Qs,
            self.QZs,
        )

        if bCopy:
            FE = self.downloadFF(FE)
        if bFinish:
            self.queue.finish()
        if bRuntime:
            print("runtime(ForceField_LJC.run_evalLJC_Hartree) [s]: ", time.perf_counter() - t0)

        return FE

    def run_gradPotentialGrid(self, pot=None, E_field=None, h=None, local_size=(32,), bCopy=True, bFinish=True):
        """
        Obtain electric field on the force field grid as the negative gradient of Hartree potential
        via centered difference.

        Arguments:
            pot: HartreePotential or None. Hartree potential to differentiate. If None, has to be initialized
                beforehand with prepareBuffers.
            E_field: np.ndarray or None. Array where output electric field is copied to if bCopy == True.
                If None and bCopy == True, will be created automatically.
            h: float > 0.0 or None. Finite difference step size (one-sided) in angstroms. If None, the default
                value DEFAULT_FD_STEP is used.
            local_size: tuple of a single int. Size of local work group on device.
            bCopy: Bool. Whether to return the calculated electric field to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns:
            E_field: np.ndarray if bCopy == True or None otherwise. Calculated electric field and potential.
        """

        if bRuntime:
            t0 = time.perf_counter()

        if pot:
            self.prepareBuffers(pot=pot)
        elif not self.pot:
            raise ValueError("Hartree potential not initialized on the device. " "Either initialize it with prepareBuffers or pass it here as a HartreePotential object.")

        if bCopy:
            E_field = E_field or np.empty(self.nDim, dtype=np.float32)
            if not np.allclose(E_field.shape, self.nDim):
                raise ValueError(f"E_field array dimensions {E_field.shape} do not match with " f"force field dimensions {self.nDim}.")

        if not self.cl_Efield:
            self.prepareBuffers(E_field=True)

        h = h or DEFAULT_FD_STEP

        # Check if potential grid matches the force field grid and is orthogonal.
        # If it does, we don't need to do interpolation.
        matching_grid = (
            np.allclose(self.pot.shape, self.nDim[:3])
            and (np.abs(self.pot.origin - self.lvec0[:3]) < 1e-3).all()
            and (np.abs(np.diag(self.pot.step) * self.pot.shape - np.diag(self.lvec[:, :3])) < 1e-3).all()
            and (self.pot.step == np.diag(np.diag(self.pot.step))).all()
        )
        if self.verbose > 0:
            print("Matching grid:", matching_grid)

        if bRuntime:
            print("runtime(ForceField_LJC.run_gradPotentialGrid.pre) [s]: ", time.perf_counter() - t0)

        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        if matching_grid:
            # fmt: off
            cl_program.grad(self.queue, global_size, local_size,
                self.pot.cl_array,
                self.cl_Efield,
                np.append(self.pot.shape, 0).astype(np.int32),
                np.append(np.diag(self.pot.step), 0).astype(np.float32),
                np.int32(1),
                np.array([-1.0, -1.0, -1.0, 1.0], dtype=np.float32)
            )
            # fmt: on
        else:
            T = np.append(np.linalg.inv(self.pot.step).T.copy(), np.zeros((3, 1)), axis=1).astype(np.float32)
            # fmt: off
            cl_program.gradPotentialGrid(self.queue, global_size, local_size,
                self.pot.cl_array,
                self.cl_Efield,
                np.append(self.pot.shape, 0).astype(np.int32),
                T[0],
                T[1],
                T[2],
                np.append(self.pot.origin, 0).astype(np.float32),
                self.nDim,
                self.dlvec[0],
                self.dlvec[1],
                self.dlvec[2],
                self.lvec0,
                np.array([h, h, h, 0.0], dtype=np.float32)
            )
            # fmt: on

        if bCopy:
            cl.enqueue_copy(self.queue, E_field, self.cl_Efield)
        if bFinish:
            self.queue.finish()
        if bRuntime:
            print("runtime(ForceField_LJC.run_gradPotentialGrid) [s]: ", time.perf_counter() - t0)

        return E_field

    def addLJ(self, local_size=(32,)):
        """Add Lennard-Jones force and energy to the current force field grid.

        Arguments:
            local_size: tuple of a single int. Size of local work group on device.
        """
        local_size = (min(local_size[0], 64),)
        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        # fmt: off
        cl_program.addLJ(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_FE,
            self.nDim,
            self.lvec0,
            self.dlvec[0],
            self.dlvec[1],
            self.dlvec[2]
        )
        # fmt: on

    def addvdW(self, damp_method=0, local_size=(32,)):
        """Add Lennard-Jones van der Waals force and energy to the current force field grid.

        Arguments:
            damp_method: int. Type of damping to use. -1: no damping, 0: constant, 1: R2, 2: R4, 3: invR4, 4: invR8.
            local_size: tuple of a single int. Size of local work group on device.
        """
        local_size = (min(local_size[0], 64),)
        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        # fmt: off
        cl_program.addvdW(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_cLJs,
            self.cl_REAs,
            self.cl_FE,
            self.nDim,
            self.lvec0,
            self.dlvec[0],
            self.dlvec[1],
            self.dlvec[2],
            np.int32(damp_method)
        )
        # fmt: on

    def _get_dftd3_params(self, params, local_size=(32,)):
        if not hasattr(self, "iZPP"):
            raise RuntimeError("Probe particle atomic number not set. Set it before DFT-D3 calculation using setPP()")
        if not hasattr(self, "cl_Zs") or not hasattr(self, "nAtoms"):
            raise RuntimeError("Atom positions or elements not set. Set them before DFT-D3 calculation using prepareBuffers(atoms=..., Zs=...)")

        params = d3.get_df_params(params)
        params = np.array([params["s6"], params["s8"], params["a1"], params["a2"] * io.bohrRadius2angstroem], dtype=np.float32)
        k = np.array([d3.K1, d3.K2, d3.K3, 0.0], dtype=np.float32)

        self.cl_cD3 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=(self.nAtoms * 4 * 4))

        global_size = (int(np.ceil(self.nAtoms / local_size[0]) * local_size[0]),)
        # fmt: off
        cl_program.d3_coeffs(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_Zs,
            self.d3_params.cl_rcov,
            self.d3_params.cl_rcut,
            self.d3_params.cl_ref_cn,
            self.d3_params.cl_ref_c6,
            self.d3_params.cl_r4r2,
            self.cl_cD3,
            k,
            params,
            self.iZPP
        )
        # fmt: on

    def add_dftd3(self, params="PBE", local_size=(64,)):
        """
        Add van der Waals force and energy to the force field grid using the DFT-D3 method. Uses the Becke-Johnson
        damping method. Mainly useful in conjunction with the full-density based model.

        The DFT-D3 parameters are adjusted based on the DFT functional. There are predefined scaling parameters for the
        following functionals:
        PBE, B1B95, B2GPPLYP, B3PW91, BHLYP, BMK, BOP, BPBE, CAMB3LYP, LCwPBE, MPW1B95, MPWB1K, mPWLYP, OLYP, OPBE,
        oTPSS, PBE38, PBEsol, PTPSS, PWB6K, revSSB, SSB, TPSSh, HCTH120, B2PLYP, B3LYP, B97D, BLYP, BP86, DSDBLYP,
        PBE0, PBE, PW6B95, PWPB95, revPBE0, revPBE38, revPBE, rPW86PBE, TPSS0, TPSS.
        See also :data:`.DF_DEFAULT_PARAMS`.

        Otherwise, the parameters can be manually specified in a dict with the following entries:

         - 's6': Scaling parameter for r^-6 term.
         - 's8': Scaling parameter for r^-8 term.
         - 'a1': Scaling parameter for cutoff radius.
         - 'a2': Additive parameter for cutoff radius. Unit should be Bohr.

        Arguments:
            params: str or dict. Functional-specific scaling parameters. Can be a str with the
                functional name or a dict with manually specified parameters.
            local_size: tuple of a single int. Size of local work group on device.
        """
        if bRuntime:
            t0 = time.perf_counter()
        local_size = (min(local_size[0], 64),)  # The kernel uses shared memory arrays with size 64. Let's not overflow.
        self._get_dftd3_params(params, local_size)
        if bRuntime:
            self.queue.finish()
            print("runtime(ForceField_LJC.add_dftd3.get_params) [s]: ", time.perf_counter() - t0)
        global_size = [int(np.ceil(np.prod(self.nDim[:3]) / local_size[0]) * local_size[0])]
        # fmt: off
        cl_program.addDFTD3_BJ(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_cD3,
            self.cl_FE,
            self.nDim,
            self.lvec0,
            self.dlvec[0],
            self.dlvec[1],
            self.dlvec[2]
        )
        # fmt: on
        if bRuntime:
            self.queue.finish()
            print("runtime(ForceField_LJC.add_dftd3) [s]: ", time.perf_counter() - t0)

    def calc_force_hartree(self, FE=None, rot=np.eye(3), rot_center=np.zeros(3), local_size=(32,), bCopy=True, bFinish=True):
        """
        Calculate force field for LJ + Hartree cross-correlated with tip density via FFT.

        Arguments:
            FE: np.ndarray or None. Array where output force field is copied to if bCopy == True.
                If None and bCopy == True, will be created automatically.
            rot: np.ndarray of shape (3, 3). Rotation matrix applied to the atom coordinates.
            rot_center: np.ndarray of shape (3,). Point around which rotation is performed.
            local_size: tuple of a single int. Size of local work group on device.
            bCopy: Bool. Whether to copy the calculated forcefield field to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns:
            FE: np.ndarray if bCopy==True or None otherwise. Calculated force field and energy.
        """

        if bRuntime:
            t0 = time.perf_counter()

        if (self.lvec[:, :3] != np.diag(np.diag(self.lvec[:, :3]))).any():
            raise NotImplementedError(
                "Forcefield calculation via FFT for non-rectangular grids is not implemented. " "Note that the forcefield grid does not need to match the Hartree potential grid."
            )

        # Interpolate Hartree potential onto the correct grid
        lvec = np.concatenate([self.lvec0[None, :3], self.lvec[:, :3]], axis=0)
        pot = self.pot.interp_at(lvec, self.nDim[:3], rot=rot, rot_center=rot_center, local_size=local_size, queue=self.queue)

        if bRuntime:
            print("runtime(ForceField_LJC.calc_force_hartree.interpolate) [s]: ", time.perf_counter() - t0)

        # Cross-correlate Hartree potential and tip charge density
        E = self.fft_corr.correlate(pot)

        if bRuntime:
            self.queue.finish()
            print("runtime(ForceField_LJC.calc_force_hartree.cross-correlation) [s]: ", time.perf_counter() - t0)

        # Take gradient to get electrostatic force field
        E.grad(scale=[-1.0, -1.0, -1.0, 1.0], array_out=self.cl_FE, order="F", local_size=local_size, queue=self.queue)

        if bRuntime:
            self.queue.finish()
            print("runtime(ForceField_LJC.calc_force_hartree.gradient) [s]: ", time.perf_counter() - t0)

        # Add Lennard-Jones force
        self.addLJ(local_size=local_size)

        if bCopy:
            FE = self.downloadFF(FE)
        if bFinish or bRuntime:
            self.queue.finish()
        if bRuntime:
            print("runtime(ForceField_LJC.calc_force_hartree) [s]: ", time.perf_counter() - t0)

        return FE

    def calc_force_fdbm(
        self, A=18.0, B=1.0, vdw_type="D3", d3_params="PBE", lj_vdw_damp=2, FE=None, rot=np.eye(3), rot_center=np.zeros(3), local_size=(32,), bCopy=True, bFinish=True
    ):
        """
        Calculate force field using the full density-based model.

        Arguments:
            A: float. Prefactor for Pauli repulsion.
            B: float. Exponent used for Pauli repulsion.
            vdw_type: ``'D3'`` or ``'LJ'``. Type of vdW interaction to use with the FDBM. ``'D3'`` is for Grimme-D3 and ``'LJ'`` uses
                standard Lennard-Jones vdW.
            d3_params: str or dict. Functional-specific scaling parameters for DFT-D3. Can be a str with the
                functional name or a dict with manually specified parameters. See :meth:`add_dftd3`.
            lj_vdw_damp:  int. Type of damping to use in vdw calculation ``fdbm_vdw_type=='LJ'``.
                ``-1``: no damping, ``0``: constant, ``1``: R2, ``2``: R4, ``3``: invR4, ``4``: invR8.
            FE: np.ndarray or None. Array where output force field is copied to if ``bCopy==True``.
                If ``None`` and ``bCopy==True``, will be created automatically.
            rot: np.ndarray of shape ``(3, 3)``. Rotation matrix applied to the atom coordinates.
            rot_center: np.ndarray of shape ``(3,)``. Point around which rotation is performed.
            local_size: tuple of a single int. Size of local work group on device.
            bCopy: Bool. Whether to copy the calculated forcefield field to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns:
            FE: np.ndarray if ``bCopy==True`` or ``None`` otherwise. Calculated force field and energy.
        """

        if bRuntime:
            t0 = time.perf_counter()

        if (self.lvec[:, :3] != np.diag(np.diag(self.lvec[:, :3]))).any():
            raise NotImplementedError(
                "Forcefield calculation via FFT for non-rectangular grids is not implemented. " "Note that the forcefield grid does not need to match the Hartree potential grid."
            )

        # Interpolate sample Hartree potential and electron density onto the correct grid
        lvec = np.concatenate([self.lvec0[None, :3], self.lvec[:, :3]], axis=0)
        pot_lvec_same = np.allclose(lvec, self.pot.lvec) and np.allclose(self.pot.shape, self.nDim[:3])
        rho_sample_lvec_same = np.allclose(lvec, self.rho_sample.lvec) and np.allclose(self.rho_sample.shape, self.nDim[:3])
        if not pot_lvec_same:
            pot = self.pot.interp_at(lvec, self.nDim[:3], rot=rot, rot_center=rot_center, local_size=local_size, queue=self.queue)
        else:
            pot = self.pot
        if not rho_sample_lvec_same:
            rho_sample = self.rho_sample.interp_at(lvec, self.nDim[:3], rot=rot, rot_center=rot_center, local_size=local_size, queue=self.queue)
        else:
            rho_sample = self.rho_sample

        if bRuntime:
            self.queue.finish()
            print("runtime(ForceField_LJC.calc_force_fdbm.interpolate) [s]: ", time.perf_counter() - t0)

        # Cross-correlate Hartree potential and tip electron delta density for electrostatic energy
        E_es = self.fft_corr_delta.correlate(pot, scale=-1.0)  # scale=-1.0, because the electron density has positive sign.
        if not pot_lvec_same:
            pot.release(keep_on_host=False)

        # Cross-correlate sample electron density and tip electron density for Pauli energy
        if not np.allclose(B, 1.0):
            rho_sample = rho_sample.power_positive(p=B, in_place=False)
        E_pauli = self.fft_corr.correlate(rho_sample)
        if not rho_sample_lvec_same:
            rho_sample.release(keep_on_host=False)

        if bRuntime:
            self.queue.finish()
            print("runtime(ForceField_LJC.calc_force_fdbm.cross-correlation) [s]: ", time.perf_counter() - t0)

        # Add the two energy contributions together
        E = E_es.add_mult(E_pauli, scale=A, in_place=True, local_size=local_size, queue=self.queue)
        E_pauli.release(keep_on_host=False)

        if bRuntime:
            self.queue.finish()
            print("runtime(ForceField_LJC.calc_force_fdbm.add_mult) [s]: ", time.perf_counter() - t0)

        # Take gradient to get the force field
        E.grad(scale=[-1.0, -1.0, -1.0, 1.0], array_out=self.cl_FE, order="F", local_size=local_size, queue=self.queue)
        E.release(keep_on_host=False)

        if bRuntime:
            self.queue.finish()
            print("runtime(ForceField_LJC.calc_force_fdbm.gradient) [s]: ", time.perf_counter() - t0)

        # Add vdW force
        if vdw_type == "D3":
            self.add_dftd3(params=d3_params, local_size=local_size)
        elif vdw_type == "LJ":
            self.addvdW(damp_method=lj_vdw_damp, local_size=local_size)
        else:
            raise ValueError(f"Invalid vdw type `{vdw_type}`")

        if bCopy:
            FE = self.downloadFF(FE)
        if bFinish or bRuntime:
            self.queue.finish()
        if bRuntime:
            print("runtime(ForceField_LJC.calc_force_fdbm) [s]: ", time.perf_counter() - t0)

        return FE

    def makeFF(
        self,
        xyzs,
        cLJs,
        REAs=None,
        Zs=None,
        method="point-charge",
        FE=None,
        qs=None,
        pot=None,
        rho_sample=None,
        rho=None,
        rho_delta=None,
        A=18.0,
        B=1.0,
        fdbm_vdw_type="D3",
        d3_params="PBE",
        lj_vdw_damp=2,
        rot=np.eye(3),
        rot_center=np.zeros(3),
        local_size=(32,),
        bRelease=True,
        bCopy=True,
        bFinish=True,
    ):
        """
        Generate the force field for a tip-sample interaction.

        There are several methods for generating the force field:

            - ``'point-charge'``: Lennard-Jones + point-charge electrostatics for both tip and sample.
            - ``'hartree'``: Lennard-Jones + sample hartree potential cross-correlated with tip charge density
              for electrostatic interaction.
            - ``'fdbm'``: Approximated full density-based model. Pauli repulsion is calculated by tip-sample
              electron density overlap + attractive vdW like in Lennard-Jones. Electrostatic
              interaction is same as in 'hartree', except tip delta-density is used instead.

        If ``pot``, ``rho``, or ``rho_delta`` is ``None`` and is required for the specified method, it has to be
        initialized beforehand with :meth:`prepareBuffers`.

        Arguments:
            xyzs: np.ndarray of shape ``(n_atoms, 3)``. xyz positions.
            cLJs: np.ndarray of shape ``(n_atoms, 2)``. Lennard-Jones interaction parameters in AB form for each atom.
            REAs: np.ndarray of shape ``(n_atoms, 4)`` or None. Lennard-Jones interaction parameters in RE form for each atom.
                Required when method is 'fdbm', fdbm_vdw_type is 'LJ', and vdw_damp_method >= 1.
            Zs: np.ndarray of shape ``(n_atoms,)``. Atomic numbers. Required when method is 'fdbm'.
            method: 'point-charge', 'hartree' or 'fdbm'. Method for generating the force field.
            FE: np.ndarray or None. Array where output force field is copied to if bCopy == True.
                If None and bCopy == True, will be created automatically.
            qs: np.ndarray of shape ``(n_atoms,)`` or None. Point charges of atoms. Used when method
                is 'point-charge'.
            pot: :class:`HartreePotential` or None. Hartree potential used for electrostatic interaction when
                method is 'hartree' or 'fdbm'.
            rho_sample: :class:`ElectronDensity` or None. Sample electron density. Used for Pauli repulsion
                when method is 'fdbm'.
            rho: :class:`TipDensity` or None. Probe tip charge density. Used for electrostatic interaction when
                method is 'hartree' and Pauli repulsion when method is 'fdbm'.
            rho_delta: :class:`TipDensity` or None. Probe tip electron delta-density. Used for electrostatic
                interaction when method is 'fdbm'.
            A: float. Prefactor for Pauli repulsion when method is 'fdbm'.
            B: float. Exponent used for Pauli repulsion when method is 'fdbm'.
            fdbm_vdw_type: 'D3' or 'LJ'. Type of vdW interaction to use when method is 'fdbm'. 'D3' is for Grimme-D3 and
                'LJ' uses standard Lennard-Jones vdW.
            d3_params: str or dict. Functional-specific scaling parameters for DFT-D3. Can be a str with the functional name
                or a dict with manually specified parameters. Used when method is 'fdbm. See :meth:`add_dftd3`.
            lj_vdw_damp: int. Type of damping to use in vdw calculation when method is 'fdbm' and fdbm_vdw_type is 'LJ'.
                -1: no damping, 0: constant, 1: R2, 2: R4, 3: invR4, 4: invR8.
            rot: np.ndarray of shape ``(3, 3)``. Rotation matrix applied to the atom coordinates.
            rot_center: np.ndarray of shape ``(3,)``. Point around which rotation is performed.
            local_size: tuple of a single int. Size of local work group on device.
            bRelease: Bool. Whether to delete data on device after computation is done.
            bCopy: Bool. Whether to copy the calculated forcefield field to host.
            bFinish: Bool. Whether to wait for execution to finish.

        Returns:
            FE: np.ndarray if ``bCopy==True`` or ``None`` otherwise. Calculated force field and energy.
        """

        if bRuntime:
            t0 = time.perf_counter()

        if not hasattr(self, "nDim") or not hasattr(self, "lvec"):
            raise RuntimeError("Forcefield position is not initialized. Initialize with initSampling.")

        # Rotate atoms
        xyzs = xyzs.copy()
        xyzs -= rot_center
        xyzs = np.dot(xyzs, rot.T)
        xyzs += rot_center
        rot_ff = np.linalg.inv(rot)  # Force field rotation is in opposite direction to atoms

        # Prepare data on device
        if qs is None:
            qs = np.zeros(len(xyzs))
        self.atoms = np.concatenate([xyzs, qs[:, None]], axis=1)
        self.prepareBuffers(self.atoms, cLJs, REAs=REAs, Zs=Zs, pot=pot, rho=rho, rho_delta=rho_delta, rho_sample=rho_sample)
        if bRuntime:
            print("runtime(ForceField_LJC.makeFF.pre) [s]: ", time.perf_counter() - t0)

        if method == "point-charge":
            if np.allclose(self.atoms[:, -1], 0):  # No charges
                FF = self.run_evalLJ_noPos()
            else:
                FF = self.run_evalLJC_QZs_noPos(FE=FE, local_size=(32,), bCopy=bCopy, bFinish=bFinish)

        elif method == "hartree":
            if rho == None and self.rho is None:
                if not np.allclose(rot, np.eye(3)):
                    raise NotImplementedError("Force field calculation with rotation for Hartree potential with " "point charges tip density is not implemented.")
                self.run_gradPotentialGrid(local_size=local_size, bCopy=False, bFinish=False)
                FF = self.run_evalLJC_Hartree(FE=FE, local_size=local_size, bCopy=bCopy, bFinish=bFinish)
            else:
                FF = self.calc_force_hartree(rot=rot_ff, rot_center=rot_center, local_size=local_size, bCopy=bCopy, bFinish=bFinish)

        elif method == "fdbm":
            FF = self.calc_force_fdbm(
                A=A,
                B=B,
                rot=rot_ff,
                rot_center=rot_center,
                vdw_type=fdbm_vdw_type,
                d3_params=d3_params,
                lj_vdw_damp=lj_vdw_damp,
                local_size=local_size,
                bCopy=bCopy,
                bFinish=bFinish,
            )

        else:
            raise ValueError(f"Unknown method for force field calculation: `{method}`.")

        if bRelease:
            self.tryReleaseBuffers()
        if bRuntime:
            print("runtime(ForceField_LJC.makeFF.tot) [s]: ", time.perf_counter() - t0)

        return FF


class AtomProjection:
    """
    to generate reference output maps ( Ys )  in generator for Neural Network training
    """

    Rpp = 2.0  #  probe-particle radius
    zmin = -3.0  #  minim position of pixels sampled in SphereMaps
    dzmax = 2.0  #  maximum distance of atoms from sampling screen for Atomic Disk maps ( similar )
    dzmax_s = np.inf  #  maximum depth of vdW shell in Atomic Disks

    Rmax = 10.0  #  Radial function of bonds&atoms potential  ; used in Bonds
    drStep = 0.1  #  step dx (dr) for sampling of radial function; used in Bonds
    elipticity = 0.5
    #  ration between major and minor semiaxi;   used in Bonds

    # occlusion
    zmargin = 0.2  #  zmargin
    tgMax = 0.5  #  tangens of angle limiting occlusion for SphereCaps
    tgWidth = 0.1  #  tangens of angle for limiting rendered area for SphereCaps
    Rfunc = None

    def __init__(self):
        self.ctx = oclu.ctx
        self.queue = oclu.queue

    def makeCoefsZR(self, Zs, ELEMENTS):
        """
        make atomic coeficients used e.g. in MultiMap
        """
        na = len(Zs)
        coefs = np.zeros((na, 4), dtype=np.float32)
        if verbose > 0:
            print("Zs", Zs)
        for i, ie in enumerate(Zs):
            coefs[i, 0] = 1.0
            coefs[i, 1] = ie
            coefs[i, 2] = ELEMENTS[ie - 1][6]
            coefs[i, 3] = ELEMENTS[ie - 1][7]
        if verbose > 0:
            print("coefs[:,2]", coefs[:, 2])
        return coefs

    def prepareBuffers(self, atoms, prj_dim, coefs=None, bonds2atoms=None, Rfunc=None, elem_channels=None):
        """
        allocate GPU buffers
        """
        if verbose > 0:
            print("AtomProjection.prepareBuffers prj_dim", prj_dim)
        self.prj_dim = prj_dim
        nbytes = 0
        self.nAtoms = np.int32(len(atoms))
        mf = cl.mem_flags
        self.cl_atoms = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=atoms)
        nbytes += atoms.nbytes

        if (Rfunc is not None) or (self.Rfunc is not None):
            if Rfunc is None:
                Rfunc = self.Rfunc
            self.Rfunc = Rfunc
            Rfunc = Rfunc.astype(np.float32, copy=False)
            self.cl_Rfunc = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Rfunc)

        if bonds2atoms is not None:
            self.nBonds = np.int32(len(bonds2atoms))
            bondPoints = np.empty((self.nBonds, 8), dtype=np.float32)
            bondPoints[:, :4] = atoms[bonds2atoms[:, 0]]
            bondPoints[:, 4:] = atoms[bonds2atoms[:, 1]]
            self.bondPoints = bondPoints
            self.bonds2atoms = bonds2atoms
            self.cl_bondPoints = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bondPoints)
            nbytes += bondPoints.nbytes

        if coefs is None:
            coefs = np.zeros((self.nAtoms, 4), dtype=np.float32)
            coefs[:, 0] = 1.0  # amplitude
            coefs[:, 1] = 0.1  # width

        self.cl_coefs = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=coefs)
        nbytes += coefs.nbytes

        npostot = prj_dim[0] * prj_dim[1]

        bsz = np.dtype(np.float32).itemsize * npostot
        self.cl_poss = cl.Buffer(self.ctx, mf.READ_ONLY, bsz * 4)
        nbytes += bsz * 4  # float4
        self.cl_Eout = cl.Buffer(self.ctx, mf.WRITE_ONLY, bsz * prj_dim[2])
        nbytes += bsz  # float

        self.cl_itypes = cl.Buffer(self.ctx, mf.READ_ONLY, 200 * np.dtype(np.int32).itemsize)
        nbytes += bsz  # float

        if elem_channels:
            elem_channels = np.array(elem_channels).astype(np.int32)
            self.cl_elem_channels = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=elem_channels)
            nbytes += elem_channels.nbytes

        if verbose > 0:
            print("AtomProjection.prepareBuffers.nbytes ", nbytes)

    def updateBuffers(self, atoms=None, coefs=None, poss=None):
        """
        upload data to GPU
        """
        oclu.updateBuffer(atoms, self.cl_atoms)
        oclu.updateBuffer(coefs, self.cl_coefs)
        oclu.updateBuffer(poss, self.cl_poss)

    def setAtomTypes(self, types, sel=[1, 6, 8]):
        """
        setup selection of atomic types for SpheresType kernel and upload them to GPU
        """
        self.nTypes = np.int32(len(sel))
        dct = {typ: i for i, typ in enumerate(sel)}
        itypes = np.ones(200, dtype=np.int32)
        itypes[:] *= -1
        for i, typ in enumerate(types):
            if typ in dct:
                itypes[i] = dct[typ]
        cl.enqueue_copy(self.queue, self.cl_itypes, itypes)
        return itypes, dct

    def releaseBuffers(self):
        """
        deallocated all GPU buffers
        """
        if verbose > 0:
            print(" AtomProjection.releaseBuffers ")
        self.cl_atoms.release()
        self.cl_coefs.release()
        self.cl_poss.release()
        self.cl_FE.release()

    def tryReleaseBuffers(self):
        """
        deallocated all GPU buffers (those which exists)
        """
        if verbose > 0:
            print(" AtomProjection.releaseBuffers ")
        try:
            self.cl_atoms.release()
        except:
            pass
        try:
            self.cl_coefs.release()
        except:
            pass
        try:
            self.cl_poss.release()
        except:
            pass
        try:
            self.cl_FE.release()
        except:
            pass

    def run_evalLorenz(self, poss=None, Eout=None, local_size=(32,)):
        """
        kernel producing lorenzian function around each atom
        """
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
            if verbose > 0:
                print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalLorenz(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evaldisks(self, poss=None, Eout=None, tipRot=None, offset=0.0, local_size=(32,)):
        """
        kernel producing atomic disks with conical profile
        """
        if tipRot is not None:
            self.tipRot = tipRot
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
            if verbose > 0:
                print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalDisk(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32(self.dzmax),
            np.float32(self.dzmax_s),
            np.float32(offset),
            np.float32(self.Rpp),
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2]
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evaldisks_occlusion(self, poss=None, Eout=None, tipRot=None, local_size=(32,)):
        """
        kernel producing atomic disks occluded by higher nearby atoms
        """
        if tipRot is not None:
            self.tipRot = tipRot
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
            if verbose > 0:
                print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalDisk_occlusion(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32(self.Rpp),
            np.float32(self.zmin),
            np.float32(self.zmargin),
            np.float32(self.dzmax),
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2]
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evalSpheres(self, poss=None, Eout=None, tipRot=None, local_size=(32,)):
        """
        kernel producing van der Waals spheres
        """
        if tipRot is not None:
            self.tipRot = tipRot
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
            if verbose > 0:
                print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalSpheres(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32(self.Rpp),
            np.float32(self.zmin),
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2]
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evalSphereCaps(self, poss=None, Eout=None, tipRot=None, local_size=(32,)):
        """
        kernel producing spherical caps (just to top most part of vdW sphere)
        """
        if tipRot is not None:
            self.tipRot = tipRot
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
            if verbose > 0:
                print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalSphereCaps(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32(self.Rpp),
            np.float32(self.zmin),
            np.float32(self.tgMax),
            np.float32(self.tgWidth),
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2]
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evalQdisks(self, poss=None, Eout=None, tipRot=None, local_size=(32,)):
        """
        kernel producing atoms disks with positive and negative value encoding charge
        """
        if tipRot is not None:
            self.tipRot = tipRot
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
            if verbose > 0:
                print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalQDisk(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_poss,
            self.cl_Eout,
            np.float32(self.dzmax),
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2]
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evalMultiMapSpheres(self, poss=None, Eout=None, tipRot=None, bOccl=0, Rmin=1.4, Rstep=0.1, local_size=(32,)):
        """
        kernel to produce multiple channels of vdW Sphere maps each containing atoms with different vdW radius
        """
        if tipRot is not None:
            self.tipRot = tipRot
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalMultiMapSpheres(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32(self.Rpp),
            np.float32(self.zmin),
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2],
            np.int32(bOccl),
            np.int32(self.prj_dim[2]),
            np.float32(Rmin),
            np.float32(Rstep)
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evalMultiMapSpheresElements(self, poss=None, Eout=None, tipRot=None, bOccl=0, local_size=(32,)):
        """
        kernel to produce multiple channels of vdW Sphere maps each containing atoms with different vdW radius
        """
        if tipRot is not None:
            self.tipRot = tipRot
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalMultiMapSpheresElements(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_elem_channels,
            self.cl_poss,
            self.cl_Eout,
            np.float32(self.Rpp),
            np.float32(self.zmin),
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2],
            np.int32(bOccl),
            np.int32(self.prj_dim[2]),
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evalSpheresType(self, poss=None, Eout=None, tipRot=None, bOccl=0, local_size=(32,)):
        """
        kernel to produce multiple channels of vdW Sphere maps each coresponding to different atom type
        """
        if tipRot is not None:
            self.tipRot = tipRot
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalSpheresType(self.queue, global_size, local_size,
            self.nAtoms,
            self.nTypes,
            self.cl_atoms,
            self.cl_itypes,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            np.float32(self.Rpp),
            np.float32(self.zmin),
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2],
            np.int32(bOccl),
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evalBondEllipses(self, poss=None, Eout=None, tipRot=None, local_size=(32,)):
        """
        kernel to produce multiple channels of vdW Sphere maps each coresponding to different atom type
        """
        if tipRot is not None:
            self.tipRot = tipRot
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalBondEllipses(self.queue, global_size, local_size,
            self.nBonds,
            self.cl_bondPoints,
            self.cl_poss,
            self.cl_Eout,
            self.cl_Rfunc,
            np.float32(self.drStep),
            np.float32(self.Rmax),
            np.float32(self.elipticity),
            np.float32(self.zmin),
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2]
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evalAtomRfunc(self, poss=None, Eout=None, tipRot=None, local_size=(32,)):
        """
        kernel to produce multiple channels of vdW Sphere maps each coresponding to different atom type
        """
        if tipRot is not None:
            self.tipRot = tipRot
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalAtomRfunc( self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_coefs,
            self.cl_poss,
            self.cl_Eout,
            self.cl_Rfunc,
            np.float32(self.drStep),
            np.float32(self.Rmax),
            np.float32(self.zmin),
            self.tipRot[0],
            self.tipRot[1],
            self.tipRot[2]
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evalCoulomb(self, poss=None, Eout=None, local_size=(32,)):
        """
        kernel producing coulomb potential and field
        """
        if Eout is None:
            Eout = np.zeros(self.prj_dim, dtype=np.float32)
            if verbose > 0:
                print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)
        ntot = self.prj_dim[0] * self.prj_dim[1]
        ntot = makeDivisibleUp(ntot, local_size[0])  # TODO: - we should make sure it does not overflow
        global_size = (ntot,)  # TODO make sure divisible by local_size
        # fmt: off
        cl_program.evalCoulomb(self.queue, global_size, local_size,
            self.nAtoms,
            self.cl_atoms,
            self.cl_poss,
            self.cl_Eout,
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()
        return Eout

    def run_evalHartreeGradient(self, pot, poss=None, Eout=None, h=None, rot=np.eye(3), rot_center=None, local_size=(32,)):
        """
        Get electric field as the negative gradient of a Hartree potential.

        Arguments:
            pot: HartreePotential. Hartree potential to differentiate.
            poss: np.ndarray or None. Position grid for points to get the field at.
            Eout: np.ndarray or None. Output array. If None, will be created automatically.
            h: float > 0.0 or None. Finite difference step size (one-sided) in angstroms. If None, the default
                value DEFAULT_FD_STEP is used.
            rot: np.ndarray of shape (3, 3). Rotation matrix to apply to the position coordinates.
            rot_center: np.ndarray of shape (3,). Point around which rotation is performed.
            local_size: tuple of a single int. Size of local work group on device.
        """

        if Eout is None:
            Eout = np.zeros(self.prj_dim[:2], dtype=np.float32)
            if verbose > 0:
                print("FE.shape", Eout.shape, self.nDim)
        if poss is not None:
            if verbose > 0:
                print("poss.shape ", poss.shape, self.prj_dim, poss.nbytes, poss.dtype)
            oclu.updateBuffer(poss, self.cl_poss)

        global_size = (int(np.ceil(np.prod(self.prj_dim[:2]) / local_size[0]) * local_size[0]),)
        T = np.append(np.linalg.inv(pot.step).T.copy(), np.zeros((3, 1)), axis=1).astype(np.float32)
        rot = np.append(rot, np.zeros((3, 1)), axis=1).astype(np.float32)
        h = h or DEFAULT_FD_STEP

        # fmt: off
        cl_program.evalHartreeGradientZ(self.queue, global_size, local_size,
            pot.cl_array,
            self.cl_poss,
            self.cl_Eout,
            np.append(pot.shape, 0).astype(np.int32),
            T[0],
            T[1],
            T[2],
            np.append(pot.origin, 0).astype(np.float32),
            rot[0],
            rot[1],
            rot[2],
            np.append(rot_center, 0).astype(np.float32),
            np.float32(h),
        )
        # fmt: on
        cl.enqueue_copy(self.queue, Eout, self.cl_Eout)
        self.queue.finish()

        return Eout
