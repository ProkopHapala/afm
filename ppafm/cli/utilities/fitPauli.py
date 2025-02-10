#!/usr/bin/python
# This is a scirpt using the geometry in *.xsf or *.cube file and extracted density lines stored in atom_density_zlines.dat produced via extract_densities.py
# for fitting the size atom in the L-J force fields. Can be useful for ionic material surfaces, used in:
# J. Phys. Chem. Lett. 2023, 14, 7, 1983–1989
# Publication Date:February 16, 2023
# https://doi.org/10.1021/acs.jpclett.2c03243
# It will create new_xyz.xyz geometry file with each atom having its own number and
# atomtypes.ini with fitted sizes and standard e0 parameters for L-J force-field.
# !!! then adjust your params.ini with proper grid vectors and use LETTER for specification of your PP !!!
# after this use ppafm-generate-ljff -i new_xyz.xyz
# It will automatically load the parameters. You can continue as usually
# ppafm-gui not tested !
# last test 26th July 2023
# Ondrej Krejci (with help of Prokop Hapala): https://github.com/ondrejkrejci

import os
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np

import ppafm as PPU
from ppafm import common, io

parser = OptionParser()
# fmt: off
parser.add_option( "-i",           action="store", type="string", help="input file - *.xsf or *.cube - where the densities and geometries are originally", default= 'CHGCAR.xsf' )
parser.add_option( "-o",           action="store", type="string", help="output xyz file name",                    default= 'new_xyz.xyz'   )
parser.add_option( "-z", "--zcut", action="store", type="float",  help="cut-out atoms bellow this height",        default= -100.           )
parser.add_option( "-n", "--no_morse", action="store_false"    ,  help="do not-fit the morse parameters -- they can cause problems for atoms on top of each other",   default= True            )
parser.add_option( "--height",     action="store", type="float",  help="how far above atom is isosurface fitted", default= +5.0            )
parser.add_option( "--old",        action="store_true",           help="use old version of atomtypes.ini - useful when combining with old-style double PP", default= False )
parser.add_option( "--not_plot",   action="store_false"        ,  help="do not-plot the lines above atoms",       default= True            )
parser.add_option( "--debug",      action="store_true"         ,  help="plot and pr. all lines and data from fit",default= False           )
# fmt: on
(options, args) = parser.parse_args()

# ====== functions


def fitExp(zs, fs, zmin, zmax):
    fs = np.log(fs)
    f0, f1 = np.interp([zmin, zmax], zs, fs)
    alpha = (f1 - f0) / (zmax - zmin)
    A = np.exp(f0 - alpha * zmin)
    return alpha, A


def getIsoArg(zs_, fs, iso=0.01, atom_z=0.0):
    zi = int((atom_z - zs_[0]) // (zs_[1] - zs_[0]))
    ai = int((atom_z + options.height + 0.1 - zs_[0]) // (zs_[1] - zs_[0]))
    zs = zs_ - atom_z
    i = np.searchsorted(-fs[zi:ai], -iso)
    x0 = zs[zi + i - 1]
    f0 = fs[zi + i - 1]
    dx = zs[zi + i] - x0
    df = fs[zi + i] - f0
    if options.debug or x0 + dx * (iso - f0) / df >= 5.0 or x0 + dx * (iso - f0) / df <= 1.0:
        print("atom_z", atom_z)
        print("zs_[0]", zs_[0])
        print("(atom_z -zs_[0])", (zs_[1] - zs_[0]))
        print("(zs_[1]-zs_[0])", (zs_[1] - zs_[0]))
        print((x0, f0, dx, df, i, zi, ai))
        plt.plot(zs_[zi:ai], fs[zi:ai], [zs_[zi], zs_[ai]], [iso, iso])
        plt.show()
    return x0 + dx * (iso - f0) / df


def getMorse(r, R0=3.5, eps=0.03, alpha=-1.8, cpull=1.0, cpush=1.0):
    expar = np.exp(alpha * (r - R0))
    return eps * (expar * expar * cpush - 2 * expar * cpull)


def getLJ(r, R0=3.5, eps=0.03, cpull=1.0, cpush=1.0):
    rmr6 = (R0 / r) ** 6
    return eps * (rmr6 * rmr6 * cpush - 2 * rmr6 * cpull)


# ====== Main
parameters = common.PpafmParameters()
fname, fext = os.path.splitext(options.i)
fext = fext[1:]
atoms, nDim, lvec = io.loadGeometry(options.i, parameters=parameters)

data = np.transpose(np.genfromtxt("atom_density_zlines.dat"))

zs_bare = data[0]

zmin = 1.2
zmax = 2.2

FFparams = PPU.loadSpecies(fname=None)

# remove atoms lower than zcut:
mask = np.array(atoms[3]) >= float(options.zcut)

for i in range(len(atoms)):
    atoms[i] = np.array(atoms[i])[mask]

atoms_z = atoms[3]
atoms_e = atoms[0]
iZs = atoms[0]

REAs = PPU.getSampleAtomsREA(iZs, FFparams)

print("REAs:")
print(REAs)
print()

mask = np.append([True], mask)

data = data[mask]

del mask

if options.not_plot:
    import matplotlib.pyplot as plt

# f1 . pseudo-xyz file with all atoms about z-cut
ilist = list(range(len(atoms[0])))
f1 = open(options.o, "w")
f1.write(str(len(atoms[0])) + "\n")
f1.write("\n")
# f2 - atomtypes.ini file with Riso (and Alpha) coresponding to each atom
f2 = open("atomtypes.ini", "w")
number_of_atoms = len(ilist)
for i in ilist:
    fs = data[1 + i]
    zs = zs_bare  # - atoms_z[i]
    if options.old or not options.no_morse:  # Morse is not compatible with the old style and also adding options to be removed from the new style
        alpha = 0.0
        A = 0.0
    else:
        alpha, A = fitExp(zs - atoms_z[i], fs, zmin, zmax)

    Riso = getIsoArg(zs, fs, iso=0.017, atom_z=atoms_z[i])
    if not (
        -3.0 < Riso - REAs[i][0] < 0.6
    ):  # prevent oversizing of atoms when on top of each other, but allows to lower down the size of atoms a lot (e.g. Mx+ ions, can be up to 1.5 A smaller) #
        print("!!! Problem with Riso for atom no. %i : Riso %f, we will use tabled number." % (i, Riso))
        Riso = REAs[i][0]
    f1.write(str(i + 1) + " " + str(atoms[1][i]) + " " + str(atoms[2][i]) + " " + str(atoms[3][i]) + "\n")
    if options.old:  # old verison of atomtypes.ini
        f2.write(str(Riso) + " " + str(REAs[i][1]) + " " + str(i + 1) + " " + str(FFparams[iZs[i] - 1][4].decode("UTF-8")) + str(i) + "\n")
    else:  # ocl version of atomtypes.ini
        f2.write(str(Riso) + " " + str(REAs[i][1]) + " " + str(alpha / 2) + " " + str(i + 1) + " " + str(FFparams[iZs[i] - 1][4].decode("UTF-8")) + str(i) + "\n")
    print(" elem %i a_z %f Riso %f " % (atoms_e[i], atoms_z[i], Riso), REAs[i], FFparams[iZs[i] - 1][4].decode("UTF-8"))

    REAs[i][0] = Riso
    REAs[i][2] = alpha / 2.0

    if options.not_plot:
        plt.plot(zs - atoms_z[i], fs, label=("%i" % atoms_e[i]))

number_of_original_elements = len(FFparams)
print()
for ie in range(number_of_original_elements):
    # This part is important so the PP force-field can be created without a problem ...
    tmp = FFparams[ie]
    print(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4])  # if needed for debugging - tmp[4] is not there for options.old ...
    if options.old:  # old verison of atomtypes.ini
        f2.write(f"{tmp[0]} {tmp[1]} {ie+1+number_of_atoms} {tmp[4].decode('UTF-8')}\n")
    else:  # ocl version of atomtypes.ini
        f2.write(f"{tmp[0]} {tmp[1]} {tmp[2]} {ie+1+number_of_atoms} {tmp[4].decode('UTF-8')}\n")

f1.close()
f2.close()

if options.not_plot:
    plt.plot([0.0, 5.0], [0.017, 0.017], label=("threshold"))
    plt.legend()
    plt.xlim(0.0, options.height)
    plt.ylim(1e-8, 1e5)
    plt.yscale("log")
    plt.axvline(zmin, ls="--", c="k")
    plt.axvline(zmax, ls="--", c="k")
    plt.grid()

atoms = np.transpose(np.array(atoms))
print(atoms.shape, REAs.shape)
data = np.concatenate((atoms[:, :4], REAs), axis=1)
np.savetxt("atom_REAs.xyz", data, header=("%i \n # e,xyz,REA" % len(data)))

if options.not_plot:
    plt.show()

print("!!! Do not forget to adjust your params.ini with proper lattice vectors and use LETTER for a probeType !!!")
print(" -- Fitting done -- BB")
