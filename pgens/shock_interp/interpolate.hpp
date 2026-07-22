#ifndef USER_INTERPOLATE_HPP
#define USER_INTERPOLATE_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/archetypes.h"
#include "traits/metric.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/particle_shapes.hpp"
#include "kernels/pushers/context.h"


namespace kernels::user {
  using namespace ntt;
  
  template <class M, Dimension D,unsigned short O>
  struct InterpolateKernel{ 
  
      private:
	 array_t<int*>      i1, i2;
	 array_t<prtldx_t*> dx1, dx2;
         ndfield_t<D, 6> EB;  
  
      public: 
      InterpolateKernel(array_t<int*>      i1,
		      array_t<int*>        i2,
		      array_t<prtldx_t*>   dx1,
		      array_t<prtldx_t*>   dx2,
		      ndfield_t<D, 6> EB):
	  i1 {i1},
	  i2 {i2},
	  dx1 {dx1},
	  dx2 {dx2},
  	  EB {EB} {}	
      
      Inline void InterpolatedEMFields(prtlidx_t          p,
                                        vec_t<Dim::_3D>& e0,
                                        vec_t<Dim::_3D>& b0) const {

 
        if constexpr (O >= 1u) {
	  if constexpr (D == Dim::_2D) {

            const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
            const int  j { i2(p) + static_cast<int>(N_GHOSTS) };
            const auto dx1_ { static_cast<real_t>(dx1(p)) };
            const auto dx2_ { static_cast<real_t>(dx2(p)) };
  
            // primal and dual shape function
            real_t S1p[O + 1], S1d[O + 1];
            real_t S2p[O + 1], S2d[O + 1];
            // minimum contributing cells
            int    ip_min, id_min;
            int    jp_min, jd_min;
  
            // primal shape function - not staggered
            prtl_shape::order<false, O>(i, dx1_, ip_min, S1p);
            prtl_shape::order<false, O>(j, dx2_, jp_min, S2p);
            // dual shape function - staggered
            prtl_shape::order<true, O>(i, dx1_, id_min, S1d);
            prtl_shape::order<true, O>(j, dx2_, jd_min, S2d);
  
            // Ex1 -- dual, primal
            e0[0] = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c0 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c0 += S1d[idx1] * EB(id_min + idx1, jp_min + idx2, em::ex1);
              }
              e0[0] += c0 * S2p[idx2];
            }
  
            // Ex2 -- primal, dual
            e0[1] = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c0 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c0 += S1p[idx1] * EB(ip_min + idx1, jd_min + idx2, em::ex2);
              }
              e0[1] += c0 * S2d[idx2];
            }
  
            // Ex3 -- primal, primal
            e0[2] = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c0 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c0 += S1p[idx1] * EB(ip_min + idx1, jp_min + idx2, em::ex3);
              }
              e0[2] += c0 * S2p[idx2];
            }
  
            // Bx1 -- primal, dual
            b0[0] = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c0 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c0 += S1p[idx1] * EB(ip_min + idx1, jd_min + idx2, em::bx1);
              }
              b0[0] += c0 * S2d[idx2];
            }
  
            // Bx2 -- dual, primal
            b0[1] = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c0 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c0 += S1d[idx1] * EB(id_min + idx1, jp_min + idx2, em::bx2);
              }
              b0[1] += c0 * S2p[idx2];
            }
  
            // Bx3 -- dual, dual
            b0[2] = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c0 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c0 += S1d[idx1] * EB(id_min + idx1, jd_min + idx2, em::bx3);
              }
              b0[2] += c0 * S2d[idx2];
            }
  
        } 
	}
      } // void interpolateEM
  }; // struct interpolatekernel
} // namespace kernels::user

#endif // USER_INTERPOLATE_HPP
