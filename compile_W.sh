module purge
module load cmake
module load PrgEnv-cray
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load rocm

cmake -B build_W \
  -D pgen=shock_integratework \
  -D mpi=ON \
  -D gpu_aware_mpi=OFF \
  -D CMAKE_C_COMPILER=cc \
  -D CMAKE_CXX_COMPILER=hipcc \
  -D MPI_C_COMPILER=mpicc \
  -D MPI_CXX_COMPILER=mpicxx \
  -D Kokkos_ENABLE_HIP=ON \
  -D Kokkos_ARCH_AMD_GFX90A=ON \
  -D CMAKE_CXX_FLAGS="--offload-arch=gfx90a -Wno-c++11-narrowing -munsafe-fp-atomics" \
  -D CMAKE_C_FLAGS="-Wno-c++11-narrowing -munsafe-fp-atomics" \
  -D deposit=esirkepov \
  -D shape_order=2 \
  -D Kokkos_ROOT=$HOME/.entity/kokkos-trento/ \
  -D adios2_ROOT=$HOME/.entity/adios2-trento/ \

cmake --build build_W -j

