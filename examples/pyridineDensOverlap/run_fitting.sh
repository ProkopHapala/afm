#! /bin/bash

# Can be runned only after the run.sh script !

echo " ====== STEP 0 : extract_densities and fit the atomic sizes "

python3 ../../ppafm/cli/utilities/extract_densities.py -i CHGCAR.xsf --zmax 5.0

python3 ../../ppafm/cli/utilities/fitPauli.py # --old


echo "======= STEP 1 : Generate force field grid."
#ppafm-conv-rho       -s sample/CHGCAR.xsf -t tip/CHGCAR.xsf -B 1.0 -E
ppafm-generate-elff  -i LOCPOT.xsf -t "dz2" -f npy
ppafm-generate-ljff  -i new_xyz.xyz -f npy
#ppafm-generate-dftd3 -i /LOCPOT.xsf --df_name PBE
#
echo "======= STEP 2 : Relax Probe Particle using that force field grid."
ppafm-relaxed-scan -f npy -k 0.25 -q -0.05# k - to be the same as the overlap example
#

echo "======= STEP 3 : Plot the results."
ppafm-plot-results --df -f npy -k 0.25 -q 0.05 -a 2.0 # a - to be also the same as the overlap example`````
