#!/usr/bin/python
import os

import numpy as np

import ppafm as PPU
import ppafm.cpp_utils as cpp_utils
import ppafm.fieldFFT as fFFT
import ppafm.HighLevel as PPH
from ppafm import io

HELP_MESSAGE = f"""Use this program in the following way:
ppafm-generate-elff -i <filename> [ --sigma <value> ]
Supported file fromats are:
    * cube
    * xsf
"""

def main():

    parser = PPU.CLIParser(
        description='Generate electrostatic force field by cross-correlation of sample Hartree potential with tip charge density. '
            'The generated force field is saved to FFel_{x,y,z}.[ext].'
    )

    parser.add_arguments(['input', 'output_format', 'tip', 'sigma', 'Rcore', 'energy', 'noPBC'])
    parser.add_argument("--tip_dens", action="store", type=str, default=None,
        help="Use tip density from a file (.xsf). Overrides --tip.")
    parser.add_argument("--doDensity", action="store_true", help="Do density overlap")
    parser.add_argument( "--tilt", action="store", type=float, default=0, help="Tilt of tip electrostatic field (radians)")
    parser.add_argument("--KPFM_tip", action="store", type=str, default='Fit', help="Read tip density under bias")
    parser.add_argument("--KPFM_sample", action="store", type=str, help="Read sample hartree under bias")
    parser.add_argument("--Vref", action="store", type=float, help="Field under the KPFM dens. and Vh was calculated in V/Ang")
    parser.add_argument("--z0", action="store", type=float, default=0.0,
        help="Heigth of the topmost layer of metallic substrate for E to V conversion (Ang)")
    args = parser.parse_args()

    if os.path.isfile( 'params.ini' ):
        PPU.loadParams( 'params.ini' )
    else:
        print(">> LOADING default params.ini >> 's' =")
        PPU.loadParams( cpp_utils.PACKAGE_PATH / 'defaults' / 'params.ini' )
    PPU.apply_options(vars(args))

    if os.path.isfile( 'atomtypes.ini' ):
        print(">> LOADING LOCAL atomtypes.ini")
        PPU.loadSpecies( 'atomtypes.ini' )
    else:
        PPU.loadSpecies( cpp_utils.PACKAGE_PATH / 'defaults' / 'atomtypes.ini' )

    bSubstractCore =  ( (args.doDensity) and (args.Rcore > 0.0) and (args.tip_dens is not None) )
    if bSubstractCore:  # We do it here, in case it crash we don't want to wait for all the huge density files to load
        if args.tip_dens is None: raise Exception( " Rcore>0 but no tip density provided ! " )
        valElDict        = PPH.loadValenceElectronDict()
        Rs_tip,elems_tip = PPH.getAtomsWhichTouchPBCcell( args.tip_dens, Rcut=args.Rcore )

    atoms_samp,nDim_samp,lvec_samp = io.loadGeometry( args.input, params=PPU.params )
    head_samp                      = io.primcoords2Xsf( atoms_samp[0], [atoms_samp[1],atoms_samp[2],atoms_samp[3]], lvec_samp )

    V=None
    if(args.input.lower().endswith(".xsf") ):
        print(">>> loading Hartree potential from  ",args.input,"...")
        print("Use loadXSF")
        V, lvec, nDim, head = io.loadXSF(args.input)
    elif(args.input.lower().endswith(".cube") ):
        print(" loading Hartree potential from ",args.input,"...")
        print("Use loadCUBE")
        V, lvec, nDim, head = io.loadCUBE(args.input)
    V *= -1 # Unit conversion, energy to potential (eV -> V)

    if PPU.params['tip']==".py":
        #import tip
        exec(compile(open("tip.py", "rb").read(), "tip.py", 'exec'))
        print(tipMultipole)
        PPU.params['tip'] = tipMultipole
        print(" PPU.params['tip'] ", PPU.params['tip'])

    if args.tip_dens is not None:
        ###  NO NEED TO RENORMALIZE : fieldFFT already works with density
        print(">>> loading tip density from ",args.tip_dens,"...")
        rho_tip, lvec_tip, nDim_tip, head_tip = io.loadXSF( args.tip_dens )

        if bSubstractCore:
            print(">>> subtracting core densities from rho_tip ... ")
            PPH.subtractCoreDensities( rho_tip, lvec_tip, elems=elems_tip, Rs=Rs_tip, valElDict=valElDict, Rcore=args.Rcore, head=head_tip )

        PPU.params['tip'] = -rho_tip # Negative sign, because the electron density needs to be negative and but the input density is positive

    if (args.KPFM_sample is not None):
        V_v0_aux = V.copy()
        V_v0_aux2 = V.copy()

        V_kpfm=None
        sigma=PPU.params['sigma']
        print(PPU.params['sigma'])
        if(args.KPFM_sample.lower().endswith(".xsf") ):
            Vref_s = args.Vref
            print(">>> loading Hartree potential  under bias from  ",args.KPFM_sample,"...")
            print("Use loadXSF")
            V_kpfm, lvec, nDim, head = io.loadXSF(args.KPFM_sample)

        elif(args.KPFM_sample.lower().endswith(".cube") ):
            Vref_s = args.Vref
            print(" loading Hartree potential under bias from ",args.KPFM_sample,"...")
            print("Use loadCUBE")
            V_kpfm, lvec, nDim, head = io.loadCUBE(args.KPFM_sample)

        dV_kpfm = (V_kpfm - V_v0_aux)

        print(">>> loading tip density under bias from ",args.KPFM_tip,"...")
        if (args.KPFM_tip.lower().endswith(".xsf")):
            Vref_t = args.Vref
            rho_tip_v0_aux = rho_tip.copy()
            rho_tip_kpfm, lvec_tip, nDim_tip, head_tip = io.loadXSF( args.KPFM_tip )
            drho_kpfm = (rho_tip_kpfm - rho_tip_v0_aux)
        elif(args.KPFM_tip.lower().endswith(".cube")):
            Vref_t = args.Vref
            rho_tip_v0_aux = rho_tip.copy()
            rho_tip_kpfm, lvec_tip, nDim_tip, head_tip = io.loadCUBE( args.KPFM_tip, hartree=False, borh = args.borh )
            drho_kpfm = (rho_tip_kpfm - rho_tip_v0_aux)
        elif args.KPFM_tip in {'Fit', 'fit', 'dipole', 'pz'}: #To be put on a library in the near future...
            Vref_t = -0.1
            if ( PPU.params['probeType'] == '8' ):
                drho_kpfm={'pz':-0.045}
                sigma = 0.48
                print(" Select CO-tip polarization ")
            if ( PPU.params['probeType'] == '47' ):
                drho_kpfm={'pz':-0.21875}
                sigma = 0.7
                print(" Select Ag polarization with decay sigma", sigma)
            if ( PPU.params['probeType'] == '54' ):
                drho_kpfm={'pz':-0.250}
                sigma = 0.67
                print(" Select Xe-tip polarization")

        if args.tip_dens is not None: #This copy is made to avoid El and kpfm conflicts because during the computeEl, the tip is been put upside down
            tip_aux_2 = PPU.params['tip'].copy()
        else:
            tip_aux_2 = PPU.params['tip']
        FFkpfm_t0sV,Eel_t0sV=PPH.computeElFF(dV_kpfm,lvec,nDim,tip_aux_2,computeVpot=args.energy , tilt=args.tilt ,)
        FFkpfm_tVs0,Eel_tVs0=PPH.computeElFF(V_v0_aux2,lvec,nDim,drho_kpfm,computeVpot=args.energy , tilt=args.tilt, sigma=sigma )

        print("Linear E to V")
        zpos = np.linspace(lvec[0,2]-args.z0,lvec[3,2]-args.z0,nDim[0])
        for i in range(nDim[0]):
            FFkpfm_t0sV[i,:,:]=FFkpfm_t0sV[i,:,:]/((Vref_s)*(zpos[i]+0.1))
            FFkpfm_tVs0[i,:,:]=FFkpfm_tVs0[i,:,:]/((Vref_t)*(zpos[i]+0.1))

        print(">>> saving electrostatic forcefiled ... ")
        io.save_vec_field('FFkpfm_t0sV',FFkpfm_t0sV,lvec_samp ,data_format=args.output_format, head=head_samp)
        io.save_vec_field('FFkpfm_tVs0',FFkpfm_tVs0,lvec_samp ,data_format=args.output_format, head=head_samp)

    print(">>> calculating electrostatic forcefiled with FFT convolution as Eel(R) = Integral( rho_tip(r-R) V_sample(r) ) ... ")
    FFel,Eel=PPH.computeElFF(V,lvec,nDim,PPU.params['tip'],computeVpot=args.energy , tilt=args.tilt )

    print(">>> saving electrostatic forcefiled ... ")

    io.save_vec_field('FFel',FFel,lvec_samp ,data_format=args.output_format, head=head_samp, atomic_info = (atoms_samp[:4],lvec_samp))
    if args.energy:
        io.save_scal_field( 'Eel', Eel, lvec_samp, data_format=args.output_format, head=head_samp, atomic_info = (atoms_samp[:4],lvec_samp))
    del FFel,V


if __name__ == "__main__":
    main()
