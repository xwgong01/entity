export GIT_FLUSH_LINE_ENDINGS=1
#export ADIOS2_DIR=/obs/xgong/dependency/ADIOS2_MPI/
module load cmake/3.26.0
module load cuda/12.6
module load gcc/12.1.0
module load openmpi/5.0.5
module load hdf5/1.14.6
module load libfabric/1.22.0
#module load adios2-mpi/2.10.2
export HDF5_ROOT=/shared/apps/hdf5/1.14.6/openmpi/5.0.5/gcc/12.1.0
#export HDF5_ROOT=/shared/apps/hdf5/1.10.5/gcc/12.1.0
cmake -B build_sh \
   -DCMAKE_PREFIX_PATH=/obs/xgong/dependency/ADIOS2_MPI \
   -D pgen=shock -D precision=double -D mpi=ON -D output=ON -D ADIOS2_USE_CUDA=OFF -D Kokkos_ENABLE_CUDA=ON -D gpu_aware_mpi=OFF -D Kokkos_ARCH_VOLTA70=ON \
   -D deposit=esirkepov -D shape_order=2
cmake --build build_sh -j

#Kokkos_ARCH_AMPERE80=ON for A100
#............VOLTA70=ON for V100
