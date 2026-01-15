/**
 * @file kernels/inject_single_prtls.hpp
 * @brief A kernel to inject certain species, certain number of prtls at certain location
 * @implements
 *   - kernel::injector::InjectSinglePrtls_kernel
 * @namespaces:
 *   - kernel::injector::
 * @macros:
 *   - MPI_ENABLED
 * @note
 */

#ifndef KERNELS_INJECT_SINGLE_PRTLS_HPP
#define KERNELS_INJECT_SINGLE_PRTLS_HPP

#include <cmath>
#include <Kokkos_Core.hpp>
#include "enums.h"
#include "global.h"
#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"
#include "archetypes/problem_generator.h"
#include "kernels/injectors.hpp"


namespace kernel::injector{
template <class M, class ED> 
struct InjectSinglePrtls_kernel {
    
    private:
        npart_t               p, cntr;
        array_t<int*>&        i1;
        array_t<prtldx_t*>&   dx1;
        array_t<real_t*>&     ux1;
        array_t<real_t*>&     ux2;
        array_t<real_t*>&     ux3;
        array_t<real_t*>&     phi;
        array_t<real_t*>&     weight;
        array_t<short*>&      tag;
        array_t<npart_t**>&   pld_i;
        tuple_t<int, Dim::_1D>      x_targ {static_cast<int>(ZERO)};
        tuple_t<prtldx_t, Dim::_1D> dx_targ {static_cast<prtldx_t>(ZERO)};
        coord_t<Dim::_1D>&    x_Cd;
        real_t                weight_prtl       = ONE;
        real_t                phi_prtl          = ZERO;
        npart_t               domain_idx   = 0u;
        npart_t               offset;
        const ED              energy_distribution;
        const bool           use_tracking;

    public:
      InjectSinglePrtls_kernel(
          Particles<M::Dim, M::CoordType>& species,
          const ED&                        energy_distribution,
          coord_t<Dim::_1D>&               x_Cd       
      ):
      i1 {species.i1},
      dx1 {species.dx1},
      ux1 {species.ux1},
      ux2 {species.ux2},
      ux3 {species.ux3},
      phi {species.phi},
      x_Cd {x_Cd},
      weight {species.weight},
      tag {species.tag},
      pld_i {species.pld_i},
      cntr {species.counter()},
      x_targ {static_cast<int>(x_Cd[0])},
      dx_targ {static_cast<prtldx_t>(x_Cd[0] - static_cast<int>(x_Cd[0]))},
      offset {species.npart()},
      energy_distribution {energy_distribution},
      use_tracking {species.use_tracking()} {}
    
    Inline void operator()(npart_t p) const {
      //if constexpr (M::CoordType == Coord::Cart){
      vec_t<Dim::_3D>       v_Cd {ZERO,ZERO,ZERO};
      
      if (M::Dim == Dim::_1D){
        energy_distribution(x_Cd, v_Cd);
        }
      if (not use_tracking) {
        InjectParticle<M::Dim, M::CoordType, false>(p + offset,
                                                    i1, i1, i1, 
                                                    dx1, dx1, dx1, 
                                                    ux1, ux2, ux3,
                                                    phi, weight, tag, pld_i,
                                                    x_targ, dx_targ, v_Cd, weight_prtl, ZERO);
      } else {
        InjectParticle<M::Dim, M::CoordType, true>(p + offset,
                                                  i1, i1, i1, 
                                                  dx1, dx1, dx1, 
                                                  ux1, ux2, ux3,
                                                  phi, weight, tag, pld_i,
                                                  x_targ, dx_targ, v_Cd, weight_prtl, ZERO,
                                                  domain_idx, p + cntr);
      }
    }
  };// struct InjectSinglePrtls_kernel


} // namespace kernel::injector
#endif