import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ppafm.common import sphereTangentSpace
from ppafm.io import saveXYZ
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
    # Initialize OpenCL environment on chosen device platform
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
    # trainer.bRuntime = True

    # Augment molecule list with rotations and shuffle
    trainer.augment_with_rotations_entropy(sphereTangentSpace(n=100), 30)
    trainer.shuffle_molecules()

    # Image output directory
    save_dir = Path("./test_images")
    save_dir.mkdir(exist_ok=True, parents=True)

    # Iterate over batches
    counter = 0
    for ib, (afm, desc, mols) in enumerate(trainer):
        print(f"Batch {ib}. Plotting...")

        # The batch contains:
        # - afm: AFM images
        # - desc: descriptors
        # - mols: molecule geometries

        # Iterate over samples in the batch
        for j in range(len(desc[0])):

            # Plot AFM
            for i, x in enumerate(afm):
                rows, cols = 2, 5
                fig = plt.figure(figsize=(3.2 * cols, 2.5 * rows))
                for k in range(x.shape[-1]):
                    fig.add_subplot(rows, cols, k + 1)
                    plt.imshow(x[j, :, :, k].T, cmap="afmhot", origin="lower")
                    plt.colorbar()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{counter}_afm{i}.png"))
                plt.close()

            # Plot descriptors
            fig, axes = plt.subplots(1, len(aux_maps))
            fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
            fig.set_size_inches(3 * len(aux_maps), 3)
            for i, ax in enumerate(axes):
                im = ax.imshow(desc[i][j].T, origin="lower")
                fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{counter}_auxmaps.png"))
            plt.close()

            # Save molecule geometry
            mol = mols[j]
            saveXYZ(os.path.join(save_dir, f"{counter}_mol.xyz"), mol[:, :3], mol[:, 4].astype(np.int32), mol[:, 3])

            counter += 1
