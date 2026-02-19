#!/bin/bash
#SBATCH --job-name=catt_pickle
#SBATCH --account=stf
#SBATCH --partition=ckpt-all

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=02:00:00
#SBATCH --mem=200GB

#SBATCH --mail-type=END
#SBATCH --mail-user=asyed4@uw.edu

module load cesg/python/3.8.10

cd /gscratch/anantram/asyed4/DNADataSet

#for d in */; do
 #   if [ -f "$d/${d%/}.gjf" ]; then
  #      echo "Running pickle generation in $d"
   #     python convert_to_pickle.py "$d"
    #fi
#done
python convert_to_pickle.py catt
