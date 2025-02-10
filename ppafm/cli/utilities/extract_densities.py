#!/usr/bin/python
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
from optparse import OptionParser

import numpy as np

import ppafm.GridUtils as GU
from ppafm import common, io

parameters = common.PpafmParameters()

parser = OptionParser()
# fmt: off
parser.add_option( "-i",               action="store", type="string",  help="input file",                                 default='CHGCAR.xsf' )
parser.add_option( "-n", "--same_name",action="store_false",           help="save the file with same name as input file", default= True        )
parser.add_option( "--zmin",           action="store", type="float",   help="From what height to extract densities",      default=+0.0         )
parser.add_option( "--zmax",           action="store", type="float",   help="To which height extract densities",          default=-1.0         )
parser.add_option( "--dz",             action="store", type="float",   help="use dz with step",                           default=-0.1         )
parser.add_option( "--plot",           action="store_false"        ,   help="plot extracted ?",                           default=True         )
# fmt: on
(options, args) = parser.parse_args()

fname, fext = os.path.splitext(options.i)
fext = fext[1:]

atoms, nDim, lvec = io.loadGeometry(options.i, parameters=parameters)
GU.lib.setGridN(np.array(nDim[::-1], dtype=np.int32))
GU.lib.setGridCell(np.array(lvec[1:], dtype=np.float64))

F, lvec, nDim, atomic_info_or_head = io.load_scal_field(fname, data_format=fext)

# zs = np.linspace( 0, lvec[3,2], nDim[0] )
zs = np.arange(options.zmin, options.zmax if options.zmax > 0.0 else lvec[3, 2], options.dz if options.dz > 0.0 else lvec[3, 2] / nDim[0])
print(lvec)


if fext == "cube":
    F /= io.Hartree2eV

dlines = [
    zs,
]

byType = {}

natoms = len(atoms[0])
for i in range(natoms):
    elem = atoms[0][i]
    x = atoms[1][i]
    y = atoms[2][i]
    p1 = (x, y, zs[0])
    p2 = (x, y, zs[-1])
    vals = GU.interpolateLine(F, p1, p2, sz=len(zs), cartesian=True)
    rec = byType.get(elem, [np.zeros(vals.shape), 1])
    rec[0] += vals
    rec[1] += 1
    byType[elem] = rec
    dlines.append(vals)

# byType = zip(byType.items())
# byType =  zip(*byType.items())
fname = "atom_density" if options.same_name else fname
byType = list(map(list, list(zip(*list(byType.items())))))
ntypes = len(byType[0])
for i in range(ntypes):
    byType[1][i] = byType[1][i][0] / byType[1][i][1]
np.savetxt(
    fname + "_zlines_type.dat",
    np.transpose(
        np.array(
            [
                zs,
            ]
            + byType[1]
        )
    ),
    header="# types " + str(byType[0]),
)
np.savetxt(fname + "_zlines.dat", np.transpose(np.array(dlines)), header="# types " + str(atoms[0]))

if options.plot:
    import matplotlib.pyplot as plt

    plt.figure()
    for i in range(ntypes):
        plt.plot(zs, byType[1][i], label=str(byType[0][i]))
    plt.yscale("log")
    plt.legend()
    plt.figure()
    for i in range(natoms):
        plt.plot(zs, dlines[i + 1], label=str(atoms[0][i]) + "_" + str(i))
    plt.yscale("log")
    plt.legend()

    plt.show()
