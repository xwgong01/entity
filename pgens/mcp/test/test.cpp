#include "weibel_profile.hpp"
#include "arch/kokkos_aliases.h"
#include "poststep.hpp"
#include <iostream>
 /* -------------------------------------------------------------------------- */
/* Local macros                                                               */
/* -------------------------------------------------------------------------- */
#define from_Xi_to_i(XI, I)                                                    \
  { I = static_cast<int>((XI + 1)) - 1; }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    DI = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);               \
  }

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

inline auto init_pool(int seed) -> unsigned int {
    if (seed < 0) {
      unsigned int new_seed = static_cast<unsigned int>(rand());
#if defined(MPI_ENABLED)
      MPI_Bcast(&new_seed, 1, MPI_UNSIGNED, MPI_ROOT_RANK, MPI_COMM_WORLD);
#endif // MPI_ENABLED
      return new_seed;
    } else {
      return static_cast<unsigned int>(seed);
    }
  }


real_t calgm(real_t u1, real_t u2, real_t u3){
    real_t norm = std::sqrt(u1 * u1 + u2*u2+u3*u3);
    return std::sqrt(norm*norm + 1.);
}

int main(){
    using namespace ntt;
    Kokkos::initialize();{
        array_t<int*> i1("i1", 2);
        array_t<prtldx_t*> dx1("dx", 2);
        array_t<real_t*> ux1("ux1", 2);
        array_t<real_t*> ux2("ux2", 2);
        array_t<real_t*> ux3("ux3", 2);
        array_t<short*> tag("tag", 2);
        i1(0) = 2;
        i1(1) = 1;
        dx1(0) = 0.1;
        dx1(1) = 0.2;
        ux1(0) = 0.99;
        ux1(1) = 0.9;
        ux2(0) = 0.0;
        ux2(1) = 0.0;
        ux3(0) = 0.0;
        ux3(1) = 0.9;
        real_t global_max = 10.;
        real_t global_min = 0.;
        real_t Lsh = 2.;
        real_t dt = 0.05;
        real_t shock_fill = 0.3;
        real_t drift_ux = 0.1;
        random_number_pool_t pool = init_pool(1234); 
        kernel::mcp::UpdateVelKernel MyKernel(
               i1, dx1, ux1, ux2, ux3, tag,  1., 0, dt, shock_fill, global_min, global_max, drift_ux, Lsh, pool 
                );

        real_t gm = calgm(ux1(0), ux2(0), ux3(0));
        std::cout << ux1(0)/gm <<'\t' << ux2(0)/gm << '\t' << ux3(0)/gm <<  '\t'<<  gm<< std::endl;
        MyKernel(0);
        gm = calgm(ux1(0), ux2(0), ux3(0));
        std::cout << ux1(0)/gm <<'\t' << ux2(0)/gm << '\t' << ux3(0)/gm <<  '\t'<<  gm<<std::endl;

    }
    Kokkos::finalize();
    return 0;
}

#undef from_Xi_to_i
#undef from_Xi_to_i_di
#undef i_di_to_Xi
