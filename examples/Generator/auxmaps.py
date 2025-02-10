import numpy as np
from matplotlib import pyplot as plt

from ppafm.io import loadXYZ
from ppafm.ml.AuxMap import AtomicDisks, ESMapConstant, HeightMap, vdwSpheres
from ppafm.ocl.AFMulator import AFMulator

if __name__ == "__main__":
    # Load a sample
    xyzs, Zs, qs, _ = loadXYZ("example_molecules/bcb.xyz")

    # The AuxMaps expect the xyz coordinates and point charges to be in the same array
    xyzqs = np.concatenate([xyzs, qs[:, None]], axis=1)

    # We need an instance of the AFMulator for the HeightMap specifically
    afmulator = AFMulator(scan_dim=(128, 128, 30), scan_window=((0, 0, 6), (16, 16, 9)))

    # The scan region for the AuxMaps is given in similar way as for the AFMulator, except that it's 2D instead of 3D.
    scan_dim = (128, 128)
    scan_window = ((0, 0), (16, 16))

    # Construct instances of each AuxMap
    vdw_spheres = vdwSpheres(scan_dim=scan_dim, scan_window=scan_window, zmin=-1.5, Rpp=-0.5)
    atomic_disks = AtomicDisks(scan_dim=scan_dim, scan_window=scan_window, zmin=-1.2)
    height_map = HeightMap(scanner=afmulator.scanner, zmin=-2.0)
    es_map = ESMapConstant(scan_dim=scan_dim, scan_window=scan_window, vdW_cutoff=-2.0, Rpp=1.0)

    # Evaluate each AuxMap for the molecule
    y_spheres = vdw_spheres(xyzqs, Zs)
    y_disks = atomic_disks(xyzqs, Zs)
    y_es = es_map(xyzqs, Zs)

    # The HeightMap is special in that it requires the force field for the probe particle to be calculated beforehand,
    # so we first run afmulator with the molecule to generate the force field
    afm = afmulator(xyzs, Zs, qs)
    y_height = height_map(xyzqs, Zs)

    # Make a plot
    fig, axes = plt.subplots(1, 5, figsize=(14, 4), gridspec_kw={"wspace": 0.02})
    axes[0].imshow(afm[:, :, -1].T, origin="lower", cmap="gray")
    axes[1].imshow(y_spheres.T, origin="lower")
    axes[2].imshow(y_disks.T, origin="lower")
    axes[3].imshow(y_height.T, origin="lower")
    vmax = max(y_es.max(), -y_es.min())  # Make the ES Map value range symmetric so that zero is in the middle of the color range (white)
    vmin = -vmax
    axes[4].imshow(y_es.T, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)

    axes[0].set_title("AFM sim.")
    axes[1].set_title("vdW Spheres")
    axes[2].set_title("Atomic Disks")
    axes[3].set_title("Height Map")
    axes[4].set_title("ES Map")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Save image to disk
    plt.savefig("auxmaps.png", bbox_inches="tight")
