(generator-tutorial-title)=
# Generating machine learning training data

The previous tutorials described the [`AFMulator`](afmulator-tutorial) for producing AFM simulations and the [`AuxMap`](auxmap-tutorial) image descriptors.
The simulations and image descriptors can be used as input/output pairs for training a machine learning model for atomic structure prediction.
To make the generation of large databases such training data easier, the ppafm Python API provides a generator that wraps both the AFM simulation and the image descriptor calculation into a single object that produces batches of training data.

The generator comes in two forms, the {class}`.InverseAFMtrainer` which can be used for simple Lennard-Jones (+ point-charge electrostatics) simulations based on xyz files, and the {class}`.GeneratorAFMtrainer` which works with any of the force-field models, but requires a bit more setup.
The following describes how to use both of them.


## Data generation from xyz files

In the simplest case you may have just a directory full of xyz files that contain molecule geometries and possibly point charges for the atoms.
In this case we can use the {class}`.InverseAFMtrainer` as follows, assuming that the xyz files are in the folder `example_molecules`:

```python
from pathlib import Path

import numpy as np

from ppafm.common import sphereTangentSpace
from ppafm.ml.AuxMap import AtomicDisks, AtomRfunc, Bonds, HeightMap, vdwSpheres
from ppafm.ml.Generator import InverseAFMtrainer
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.oclUtils import init_env


class ExampleTrainer(InverseAFMtrainer):
    # We override this callback method in order to augment the samples with randomized tip distance and tilt
    def on_sample_start(self):
        self.randomize_distance(delta=0.2)
        self.randomize_tip(max_tilt=0.3)


if __name__ == "__main__":

    init_env(i_platform=0)

    # Define simulator with parameters
    afmulator = AFMulator(
        pixPerAngstrome=10,
        scan_dim=(128, 128, 19),
        scan_window=((2.0, 2.0, 5.0), (18.0, 18.0, 6.9)),
        df_steps=10,
        npbc=(0, 0, 0),
    )

    # Define descriptors to generate
    auxmap_args = {"scan_window": ((2.0, 2.0), (18.0, 18.0)), "scan_dim": (128, 128)}
    aux_maps = [
        vdwSpheres(**auxmap_args),
        AtomicDisks(**auxmap_args),
        HeightMap(afmulator.scanner),
        Bonds(**auxmap_args),
        AtomRfunc(**auxmap_args),
    ]

    # Paths to molecule xyz files
    xyzs_dir = Path("example_molecules")
    paths = list(xyzs_dir.glob("*.xyz"))

    # Combine everything to one
    trainer = ExampleTrainer(
        afmulator,
        aux_maps,
        paths,
        batch_size=20,
        distAbove=4.5,
        iZPPs=[8],
        QZs=[[0.1, 0, -0.1, 0]],
        Qs=[[-10, 20, -10, 0]],
    )

    # Augment molecule list with rotations and shuffle
    trainer.augment_with_rotations_entropy(sphereTangentSpace(n=100), 30)
    trainer.shuffle_molecules()

    # Iterate over batches
    for ib, (afm, desc, mols) in enumerate(trainer):
        # Do stuff with the data...

```

Here we first define the {class}`.AFMulator` and various `AuxMap` instances as before.
Additionally we gather a list of paths to the xyz files in the variable `paths`.
We combine all of the above into the generator class:
```python
trainer = ExampleTrainer(
    afmulator,
    aux_maps,
    paths,
    batch_size=20,
    distAbove=4.5,
    iZPPs=[8], # CO
    QZs=[[0.1, 0, -0.1, 0]],
    Qs=[[-10, 20, -10, 0]],
)
```
There are a couple of things to note here.
First is that the `iZPPs`, `QZs`, and `Qs` which define the properties of the probe particle (PP) are defined here instead of the `AFMulator`.
This is because it is possible to generate the AFM image for multiple different PPs at once.
Notice that the arguments here are lists.
If we wanted to add for example a Xe PP in addition to CO, we could just add the corresponding parameters to the lists:
```python
trainer = ExampleTrainer(
    ...
    iZPPs=[8, 54], # CO, Xe
    QZs=[[0.1, 0, -0.1, 0], [0.3, 0.0, 0.0, 0.0]],
    Qs=[[-10, 20, -10, 0], [0.0, 0.0, 0.0, 0.0]],
    ...
)
```
```{note}
The ordering of the PPs matters for the {class}`.HeightMap` descriptor which uses the generated force field. It will use the force field of the last PP in the list.
```

The other two important arguments are `batch_size`, which sets the number of samples in a batch (last batch might be smaller), and `distAbove` which sets the tip-sample distance.
The tip-sample distance here is measured as the distance between the top of the scan and the top atom in the sample subtracted by the vdW radius of the PP and the top atom of the sample.
This is a heuristic which gives a roughly similar level of contrast in the simulated AFM images regardless of the sample.


Secondly, instead of using the {class}`.InverseAFMtrainer` directly, we subclass it:
```python
class ExampleTrainer(InverseAFMtrainer):
    # We override this callback method in order to augment the samples with randomized tip distance and tilt
    def on_sample_start(self):
        self.randomize_distance(delta=0.2)
        self.randomize_tip(max_tilt=0.3)
```
This enables us to add random augmentations to the samples as they are generated. This is done by overriding callback methods that are automatically called at various parts of the generation process. The overridable methods are {meth}`.InverseAFMtrainer.on_afm_start`, {meth}`.InverseAFMtrainer.on_sample_start`, and {meth}`.InverseAFMtrainer.on_batch_start`.
In pseudo-code the generation procedure looks like the following:
```
on_batch_start()
for each sample:
    on_sample_start()
    for each PP:
        set_afmulator_PP_params()
        on_afm_start()
        afm = afmulator(...)
    for each descriptor:
        desc = descriptor(...)
```
By default the callbacks don't do anything.
In this case we randomize the tip-sample distance and the tip tilt for each sample by overriding the `on_sample_start` callback.
When writing the overriding methods, it is useful to know that the `AFMulator` instance is available in `self.afmulator`, the list of `AuxMap` is available in `self.aux_maps`, and the currently set tip-sample distance is available in `self.distAboveActive`.

We additionally augment the molecule list before the generation starts by taking random rotations for each molecule:
```python
trainer.augment_with_rotations_entropy(sphereTangentSpace(n=100), 30)
trainer.shuffle_molecules()
```
The {func}`.sphereTangentSpace` function first generates `n = 100` rotations (more-or-less) uniformly over a sphere and then the {meth}`InverseAFMtrainer.augment_with_rotations_entropy` method picks the 30 best ones by a certain measure of "entropy" that prefers directions that have more atoms in a plane parallel to the scan plane.
We also shuffle the list of molecules with `InverseAFMtrainer.shuffle_molecules` so that the samples within a batch are uncorrelated.

The constructed instance of the generator class is an iterator which we can use in a for-loop:
```python
for ib, (afm, desc, mols) in enumerate(trainer):
    # Do stuff with the data...
```
The returned arrays contain the simulated AFM images `afm`, the image descriptors `desc`, and the arrays of coordinates and types of the atoms `mols`.

A complete example with the script and the data can be found [in the examples folder](https://github.com/Probe-Particle/ppafm/blob/main/examples/Generator/inverse_trainer_xyz.py).

## General data generation

The above only works with xyz files and basic Lennard-Jones force field.
If we want to use a Hartree potential for the electrostatics of the FDBM for more accurate Pauli repulsion, then we will use the {class}`.GeneratorAFMtrainer` class instead.

Since the Hartree potentials and electron densities come in many different formats, in this case the file loading is separated from the generator class and is instead implemented in a separate sample generator provided by the user.
A complete example for generating data with the FDBM is provided [in the examples folder](https://github.com/Probe-Particle/ppafm/blob/main/examples/Generator/generator_trainer.py):

```python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ppafm.data import download_dataset
from ppafm.io import saveXYZ
from ppafm.ml.AuxMap import AtomicDisks, ESMapConstant, HeightMap, vdwSpheres
from ppafm.ml.Generator import GeneratorAFMtrainer
from ppafm.ocl import oclUtils as oclu
from ppafm.ocl.AFMulator import AFMulator, ElectronDensity, HartreePotential, TipDensity


class ExampleTrainer(GeneratorAFMtrainer):
    # Override the on_sample_start method to randomly modify simulation parameters for each sample.
    def on_sample_start(self):
        self.randomize_distance(delta=0.2)
        self.randomize_tip(max_tilt=0.3)


def prepare_dataset(data_dir: Path):
    """
    Download an example dataset and repackage it to a format that is better suited for fast reading from the disk.
    """
    if data_dir.exists():
        print("Data directory already exists. Skipping data preparation.")
        sample_dirs = [d for d in data_dir.glob("*") if d.is_dir()]
        return sample_dirs
    download_dataset("hartree-density", data_dir)
    sample_dirs = [d for d in data_dir.glob("*") if d.is_dir()]
    for sample_dir in sample_dirs:
        # Load data from .xsf files
        hartree, xyzs, Zs = HartreePotential.from_file(sample_dir / "LOCPOT.xsf", scale=-1.0)
        density, _, _ = ElectronDensity.from_file(sample_dir / "CHGCAR.xsf")

        # In this dataset the molecules are centered in the corner of the box. Let's shift them to the center so
        # that the distance calculation works correctly in the generator.
        box_size = np.diag(hartree.lvec[1:])
        shift = (np.array(hartree.shape) / 2).astype(np.int32)
        xyzs = (xyzs + box_size / 2) % box_size
        hartree_array = np.roll(hartree.array, shift=shift, axis=(0, 1, 2))
        density_array = np.roll(density.array, shift=shift, axis=(0, 1, 2))

        # Save into .npz files.
        np.savez(sample_dir / "hartree.npz", data=hartree_array.astype(np.float32), lvec=hartree.lvec, xyzs=xyzs, Zs=Zs)
        np.savez(sample_dir / "density.npz", data=density_array.astype(np.float32), lvec=density.lvec, xyzs=xyzs, Zs=Zs)

    return sample_dirs


def load_sample(sample_dir: Path):
    """
    Load a sample from a directory. The directory contains the files hartree.npz and density.npz, which contain
    the sample Hartree potential and electron density in the format that we prepared in prepare_dataset.
    """
    hartree_data = np.load(sample_dir / "hartree.npz")
    density_data = np.load(sample_dir / "density.npz")
    xyzs = hartree_data["xyzs"]
    Zs = hartree_data["Zs"]
    hartree = HartreePotential(hartree_data["data"], lvec=hartree_data["lvec"])
    density = ElectronDensity(density_data["data"], lvec=density_data["lvec"])
    return xyzs, Zs, hartree, density


def generate_samples(sample_dirs, rotations):
    """A simple generator that yields samples from directories."""

    for sample_dir in sample_dirs:
        # Fetch a sample from file
        xyzs, Zs, hartree, density = load_sample(sample_dir)

        for rot in rotations:
            # We yield samples as dicts containing the input arguments to AFMulator.
            sample_dict = {
                "xyzs": xyzs,
                "Zs": Zs,
                "qs": hartree,
                "rho_sample": density,
                "rot": rot,
            }

            yield sample_dict


if __name__ == "__main__":

    oclu.init_env(i_platform=0)

    # Output directory
    save_dir = Path("./generator_images_fdbm/")
    save_dir.mkdir(exist_ok=True, parents=True)

    # Get example data
    sample_dirs = prepare_dataset(data_dir=Path("./generator_data"))

    # Load tip densities
    co_dir = Path("./CO_densities")
    download_dataset("CO-tip-densities", co_dir)
    rho_tip, _, _ = TipDensity.from_file(co_dir / "density_CO.xsf")
    rho_tip_delta, _, _ = TipDensity.from_file(co_dir / "CO_delta_density_aims.xsf")

    # Create the simulator
    afmulator = AFMulator(
        pixPerAngstrome=10,
        scan_dim=(160, 160, 19),
        scan_window=((0, 0, 0), (15.9, 15.9, 1.9)),
        df_steps=10,
        npbc=(0, 0, 0),
        A_pauli=12,
        B_pauli=1.2,
    )

    # Create auxmap objects to generate image descriptors of the samples
    auxmap_args = {"scan_window": ((0, 0), (15.9, 15.9)), "scan_dim": (160, 160)}
    aux_maps = [vdwSpheres(**auxmap_args), AtomicDisks(**auxmap_args), HeightMap(afmulator.scanner), ESMapConstant(**auxmap_args)]

    # Create a sample generator that yields samples by loading them from the disk. We can augment the dataset with rotations.
    rot_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sample_generator = generate_samples(sample_dirs, rotations=[np.eye(3), rot_180])

    # Combine everything into a trainer object
    trainer = ExampleTrainer(
        afmulator,
        aux_maps,
        sample_generator,
        batch_size=6,  # Number of samples per batch
        distAbove=2.5,  # Tip-sample distance, taking into account the effective size of the tip and the sample atoms
        iZPPs=[8],  # Tip atomic numbers
        rhos=[rho_tip],  # Tip electron densities, used for the Pauli overlap integral
        rho_deltas=[rho_tip_delta],  # Tip electron delta densities, used for the electrostatic interaction
    )

    # Get samples from the trainer by iterating over it
    for afm, desc, mols, scan_windows in trainer:
        # Do stuff with the data...
```

We use here the data from the [ppafm review paper](https://doi.org/10.1016/j.cpc.2024.109341) as an example.
The data is saved in the xsf format, which is a text-based format.
Loading from disk and parsing the text files will become the biggest bottleneck of the whole process, so we first convert the data to the binary numpy npz format in `prepare_dataset`.
This step is not strictly necessary, but it becomes increasingly important with larger datasets.

Next we create the `AFMulator` and `AuxMap` instances as before.
The difference comes when we create the generator object, where instead of a list of files we provide the `sample_generator` function as an input.
The `sample_generator` function iterates over the sample directories and yields samples as dicts:

```python
def generate_samples(sample_dirs, rotations):
    """A simple generator that yields samples from directories."""

    for sample_dir in sample_dirs:
        # Fetch a sample from file
        xyzs, Zs, hartree, density = load_sample(sample_dir)

        for rot in rotations:
            # We yield samples as dicts containing the input arguments to AFMulator.
            sample_dict = {
                "xyzs": xyzs,
                "Zs": Zs,
                "qs": hartree,
                "rho_sample": density,
                "rot": rot,
            }

            yield sample_dict


if __name__ == "__main__":

    ...

    # Create a sample generator that yields samples by loading them from the disk. We can augment the dataset with rotations.
    rot_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sample_generator = generate_samples(sample_dirs, rotations=[np.eye(3), rot_180])

    # Combine everything into a trainer object
    trainer = ExampleTrainer(
        ...
        sample_generator,
        sim_type="FDBM",
        ...
        rhos=[rho_tip],  # Tip electron densities, used for the Pauli overlap integral
        rho_deltas=[rho_tip_delta],  # Tip electron delta densities, used for the electrostatic interaction
    )
```
Notice that the names of the entries in the `sample_dict` match the arguments of the {meth}`.AFMulator.eval` method.
Which entries are required depends on the type of force field requested, which is specified by the `sim_type` argument.
Notice that we could also load xyz files in the `sample_generator` and use a Lennard-Jones force field with `sim_type="LJ"` or `sim_type="LJ+PC"` as in the first example.
Or if we want to use Hartree potential for electrostatics without the FDBM for the Pauli interaction, then we can specify `sim_type="LJ+Hartree"` and skip the `"rho_sample"` in the yielded `sample_dict`.

The sample generator here is a simple function that runs in the main process same as the simulation.
This means that the simulator is waiting for the next input when it's being loaded.
To overcome this limitation, see [a more advanced example here](https://github.com/SINGROUP/ml-spm/blob/8471fe281a75fff00ea972f1bf923d0e4c3919b2/mlspm/data_generation.py#L168) where the input files are loaded in multiple parallel processes.

Also notice that the tip electron densities, similar to the point charges in the first example, are given to `ExampleTrainer` instead of `AFMulator`.
A list of electron densities can be used for generating samples for multiple tips at the same time.

Finally, notice that when we iterate over the generated samples, there is one more returned value, `scan_windows`:
```python
for afm, desc, mols, scan_windows in trainer:
    # Do stuff with the data...
```
The `scan_windows` contains a list of the used scan areas for each sample (in the same format as the `scan_window` argument to the {class}`.AFMulator`), which is varying as the scan window is centered on the molecules.
