#!/bin/bash
#SBATCH --clusters=gpu
#SBATCH --qos=gpu_hi_normal # gpu_def_long
#SBATCH --account=vanthieghem_hi # vanthieghem
#SBATCH --partition=hi
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#SBATCH --gpus-per-node=1                                                                                                                                                             
#SBATCH --ntasks-per-node=1


#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --job-name=ShockPIC1D
##SBATCH --nodelist=node0298 # A100
#SBATCH --nodelist=node0256


module purge
module load cmake/3.26.0
module load cuda/12.6
module load gcc/12.1.0
module load openmpi/5.0.5
module load hdf5/1.14.6
module load libfabric/1.22.0
#module load adios2-mpi/2.10.2
#export HDF5_ROOT=/shared/apps/hdf5/1.14.6/openmpi/5.0.5/gcc/12.1.0
export HDF5_ROOT=/shared/apps/hdf5/1.10.5/gcc/12.1.0

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/shared/apps/hpc_sdk/Linux_x86_64/24.3/REDIST/cuda/12.3/compat/

RUNDIR=/travail/xgong/output_1105 #_mpi
mkdir -p ${RUNDIR}
cd ${RUNDIR}


#${RUNDIR}/../../build/src/entity.xc -input ${RUNDIR}/../../pgens/mcp/mypgen.toml
srun ${RUNDIR}/../../../obs/xgong/entity/build/src/entity.xc -input ${RUNDIR}/../../../obs/xgong/entity/pgens/mcp/mypgen.toml  -continue
#mpiexec -np N ${RUNDIR}/../../../obs/xgong/entity/build_mpi/src/entity.xc -input ${RUNDIR}/../../../obs/xgong/entity/pgens/mcp/mypgen.toml #-restart
