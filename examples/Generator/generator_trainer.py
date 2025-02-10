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


if __name__ == "__main__":
    # Initialize OpenCL environment on chosen platform
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
        sim_type="FDBM",
        batch_size=6,  # Number of samples per batch
        distAbove=2.5,  # Tip-sample distance, taking into account the effective size of the tip and the sample atoms
        iZPPs=[8],  # Tip atomic numbers
        rhos=[rho_tip],  # Tip electron densities, used for the Pauli overlap integral
        rho_deltas=[rho_tip_delta],  # Tip electron delta densities, used for the electrostatic interaction
    )

    # Get samples from the trainer by iterating over it
    counter = 0
    for ib, (afms, descriptors, mols, scan_windows) in enumerate(trainer):
        print(f"Batch {ib+1}")

        # Loop over samples in the batch
        for afm, desc, mol, sw in zip(afms, descriptors, mols, scan_windows):
            # afm: AFM images
            # desc: Image descriptors
            # mol: Sample atom coordinates, atomic numbers, and charges
            # sw: Scan window bounds

            # Plot AFM images
            for i, x in enumerate(afm):
                rows, cols = 2, 5
                fig = plt.figure(figsize=(3.2 * cols, 2.5 * rows))
                for k in range(afm.shape[-1]):
                    fig.add_subplot(rows, cols, k + 1)
                    plt.imshow(x[..., k].T, cmap="afmhot", origin="lower")
                    plt.colorbar()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{counter}_afm{i}.png"))
                plt.close()

            # Plot images descriptors
            fig, axes = plt.subplots(1, len(aux_maps))
            fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
            fig.set_size_inches(3 * len(aux_maps), 3)
            for y, ax in zip(desc, axes):
                im = ax.imshow(y.T, origin="lower")
                fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{counter}_auxmaps.png"))
            plt.close()

            # Save molecule into a xyz file
            saveXYZ(os.path.join(save_dir, f"{counter}_mol.xyz"), mol[:, :3], mol[:, 4].astype(np.int32), mol[:, 3])

            counter += 1
