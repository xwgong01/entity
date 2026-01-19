/**
 * @file kernels/poststep.hpp
 * @brief Particle pusher for stochastic scattering and inertial corrections
 * @implements
 *   - kernel::mcp::UpdateVelKernel<>
 * @namespaces:
 *   - kernel::mcp::
 * @macros:
 *   - MPI_ENABLED
 * @note
 * At the end of the boundary condition call, if MPI is enabled particles
 * are additionally tagged depending on which direction they are leaving
 */

#ifndef KERNELS_POSTSTEP_HPP
#define KERNELS_POSTSTEP_HPP
#include "enums.h"
#include "global.h"
#include <Kokkos_Random.hpp>

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"
#include "utils/log.h"
#include "weibel_profile.hpp"

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"
#endif

namespace user_plds{
    enum {
      pldi = 2,
      pldr = 0
      };
}

/* -------------------------------------------------------------------------- */
/* Local macros                                                               */
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */

namespace kernel::mcp {
  using namespace ntt;
  
  // define the vector type for velocity
  struct vector_t {
      real_t data[3];
      Inline real_t& operator[](int i){
          return data[i];
      }
      Inline constexpr int size() const {
          return 3;
      }
  };

  template <class M, Dimension D>
  struct UpdateVelKernel {

  private:
    const M                       metric;
    array_t<int*>                 i1;
    array_t<prtldx_t*>            dx1;
    array_t<real_t*>              ux1, ux2, ux3;
    array_t<real_t**>             pld_r;
    array_t<npart_t**>            pld_i;
    array_t<short*>               tag;
    array_t<real_t*>              weight;
    const real_t                  charge;
    array_t<int>                  prtl_to_inject;
    random_number_pool_t          random_pool;
    const real_t  dt, mp, mi;
    simtime_t time;
    real_t shock_filling_fraction, global_min, global_max, drift_ux, Lsh;
    real_t nu0, nu_coeff,Bmag;
    kernel::weibel::WeibelKernel weibel_kernel;
    const real_t x_wait;
    bool DEBUG;
    
  public:
    UpdateVelKernel(
                  M                              metric,
                  array_t<int*>&                 i1,
                  array_t<prtldx_t*>&            dx1,
                  array_t<real_t*>&              ux1,
                  array_t<real_t*>&              ux2,
                  array_t<real_t*>&              ux3,
                  array_t<real_t**>&             pld_r,
                  array_t<npart_t**>&            pld_i,
                  array_t<short*>&               tag,
                  array_t<real_t*>&              weight,
                  real_t                         charge,
                  real_t                         mp,
                  real_t                         mi,
                  simtime_t                      time,
                  real_t                         dt,
                  real_t                         shock_filling_fraction,
                  real_t                         global_min,
                  real_t                         global_max,
                  real_t                         drift_ux,
                  real_t                         Lsh,
                  real_t                         Bmag,
                  real_t                         nu0,
                  real_t                         nu_coeff,
                  real_t                         x_wait,
                  array_t<int>                   prtl_to_inject,
                  random_number_pool_t          &random_pool,
                  bool                           DEBUG): 
      metric {metric}
      , i1 { i1 }
      , dx1 { dx1 }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , pld_r {pld_r}
      , pld_i {pld_i}
      , tag { tag }
      , mp  { mp }  // particle (electron) mass
      , mi  { mi }  // ion mass
      , time { time }
      , dt  { dt }
      , shock_filling_fraction {shock_filling_fraction}
      , global_min {global_min}
      , global_max {global_max}
      , drift_ux {drift_ux}
      , weight {weight}
      , Lsh {Lsh}
      , Bmag {Bmag}
      , nu0 {nu0}
      , x_wait {x_wait}
      , nu_coeff {nu_coeff}
      , charge {charge}
      , prtl_to_inject {prtl_to_inject}
      , random_pool {random_pool}
      , DEBUG {DEBUG} {
          weibel_kernel = kernel::weibel::WeibelKernel(shock_filling_fraction, global_min, global_max, drift_ux, Lsh);
      }


// Calculte vector norm 
Inline auto norm(vector_t vec) const -> real_t{
    double result = math::sqrt(math::pow(vec[0],2)+math::pow(vec[1],2)+math::pow(vec[2],2));
    return result;
}

// Calculate vector multiply with a scalar
Inline auto mul(vector_t vec, real_t a) const -> vector_t{
    for (int i=0;i<vec.size();i++){
        vec[i] *= a;
    }
    return vec;
}

// Calculate inner produkt of vectors
Inline auto dot(vector_t v1,vector_t v2) const -> real_t{
    real_t sum=0;
    for (int i=0;i<v1.size();i++){
        sum += v1[i] * v2[i];
    }
    return sum;
}

// Calculate cross product of vectors
Inline auto cross(vector_t v1,vector_t v2) const -> vector_t{
    vector_t result;
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return result;
}


// Rotate vector for an angle around a given axis
Inline auto rotate(vector_t v, vector_t k, real_t rotangle) const -> vector_t{
        k = mul(k, 1/norm(k));  // rotation axis
        
	// Rodriguez Rotation Formula
        vector_t v_new;
        vector_t term1, term2, term3;
        term1 = mul(v, math::cos(rotangle));
        term2 = mul(k, dot(k, v) * (1-  math::cos(rotangle)));
        term3 = mul(cross(k,v), math::sin(rotangle));

        for (int i=0;i<k.size();i++){
            v_new[i] =  term1[i] + term2[i] + term3[i];
        }
    
        return mul(v_new , norm(v) / norm(v_new)); // guarantee that the normaization is conserved
    }

//Lorentz transformation
Inline auto boostVel(vector_t u, real_t brel, real_t LFrel) const -> vector_t{
        // assume that brel is the relative velocity between the moving frame and the lab frame
        vector_t unew;
        real_t gm = math::sqrt(math::pow(norm(u),2.) + 1);
        unew[0] = LFrel * (u[0] - gm * brel);
        unew[1] = u[1];
        unew[2] = u[2];
        return unew;
    }

  
// Manipulate particle velocity 
Inline void operator()(index_t p) const {
        if (tag(p) == ParticleTag::dead){
            return;
        }
        real_t dtw;
        vector_t u, uw, ufin, k;
        real_t u_weibel, duwdx, brel, lfrel;
        real_t theta, phi, rotangle;
        real_t nu, gm, gmw;
        // real_t esc;

        // the velocity in lab frame
        u[0] = ux1(p);
        u[1] = ux2(p);
        u[2] = ux3(p);

        // get particle location
        real_t x_prtl = ZERO;
        if constexpr (D == Dim::_1D) {
          coord_t<Dim::_1D> x_Cd { ZERO };
          x_Cd[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));

          coord_t<Dim::_1D> x_Ph { ZERO };
          metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
        
          x_prtl = x_Ph[0];
        }
         
        gm = math::sqrt(math::pow(norm(u),2.) + 1.); // Particle Lorentz factor in lab frame

        nu = nu0 * mi / mp * ((mi == mp) ? 1.0:nu_coeff); // TODO: define the model of scattering freq.

        u_weibel = weibel_kernel.getux(x_prtl);
        duwdx = weibel_kernel.getdudx(x_prtl);
        brel = -math::sqrt(math::pow(u_weibel, 2.)/(1. + math::pow(u_weibel, 2.)));  // relative speed between lab and weibel frame (always negative)
        lfrel = 1.0 / math::sqrt(1 - math::pow(brel, 2.));     // relative Lorentz factor between lab and weibel frame
       
	    dtw = dt * lfrel * (1 - u[0]/gm * brel);     // timestep in weibel frame

        uw = boostVel(u, brel, lfrel);
	    gmw = math::sqrt(math::pow(norm(uw),2. ) +1.); // Particle Lorentz factor in Weibel frame
        
	// ----------------- scatter --------------
	// random generate k, rotation angle and normalize:
        auto generator  = random_pool.get_state();
        {
            // esc = Random<real_t>(generator);
            theta = math::acos(2. * Random<real_t>(generator) - 1);
            phi = 2. * constant::PI * Random<real_t>(generator);
            rotangle = math::sqrt(2. * nu * dtw) * math::sqrt(-2. * math::log(Random<real_t>(generator))) * math::cos(2. * constant::PI * Random<real_t>(generator));
        }
        random_pool.free_state(generator);
	    k[0] = math::sin(theta) * math::cos(phi);
        k[1] = math::sin(theta) * math::sin(phi);
        k[2] = math::cos(theta);

        uw = rotate(uw, k, rotangle); // Apply scattering
         
	
	// ------------------- final velocity -----------------
        ufin = boostVel(uw, -brel, lfrel);

        // if (DEBUG){
        //     Kokkos::printf("Prtl inside main box , pldi=%d\n",pld_i(p, 0));
        //     Kokkos::printf("Prtl inside main box , pldi=%d\n",pld_i(p, 1));
        //     Kokkos::printf("Prtl inside main box , pldi=%d\n",pld_i(p, 2));
        // }

    // ------------------- reflect beams to avoid being absorbed ---------------

    if (x_prtl <= x_wait){
        
        pld_i(p, user_plds::pldi) = 0; // reset the tag whether prtl is inside the isolated box
    }
    else { // prtl enters the isolated box
        if (DEBUG){
            // Kokkos::printf("Prtl in isolated box , x=%.4f, xwait=%.4f\n", x_prtl, x_wait );
        }
        if (pld_i(p, user_plds::pldi) == 0) // if it is still active (enters the box for the first time)
        {
            pld_r(p,user_plds::pldr) = x_prtl - x_wait; // the real position relative to x_wait.
            pld_i(p,user_plds::pldi) = 1; // mark prtl as in the isolated box
            weight(p) = ZERO; // not letting prtl affect the field
            // for particle entering box for first time, initialize. 
            Kokkos::atomic_add(&prtl_to_inject(),(charge > 0) ? ONE: -ONE);
        }
        else{
            pld_r(p,user_plds::pldr) += x_prtl - (x_wait + global_max)/2.0;
        }

        if  (pld_r(p, user_plds::pldr) > 0){ // prtl still outside the main simulation
            //drag prtl back to a safe place
            if constexpr (D == Dim::_1D) {
                coord_t<Dim::_1D> x_wait_Cd { ZERO };
                coord_t<Dim::_1D> x_wait_Ph { ZERO };
                x_wait_Ph[0] = static_cast<real_t>(x_wait/2 + global_max / 2);
                metric.template convert<Crd::Ph,Crd::Cd>(x_wait_Ph, x_wait_Cd);
                i1(p) = static_cast<int>(x_wait_Cd[0]);
                dx1(p) = x_wait_Cd[0] - static_cast<int>(x_wait_Cd[0]);
            }
            
            if (DEBUG){assert(weight(p) == ZERO);}
        }
        else { //  prtl leave the isolated box and return to the main simulation
            pld_i(p,user_plds::pldi) = 0; //  mark prtl as back
            if constexpr (D == Dim::_1D) {
                coord_t<Dim::_1D> x_wait_Cd { ZERO };
                coord_t<Dim::_1D> x_wait_Ph { ZERO };
                x_wait_Ph[0] = static_cast<real_t>(x_wait +pld_r(p, user_plds::pldr) );
                metric.template convert<Crd::Ph,Crd::Cd>(x_wait_Ph, x_wait_Cd);
                i1(p) = static_cast<int>(x_wait_Cd[0]);
                dx1(p) = x_wait_Cd[0] - static_cast<int>(x_wait_Cd[0]);
            }
            if (DEBUG){assert(weight(p) == ZERO);}
            weight(p) = ONE; // to re-enable prtl feedback on fld

            //Inject a thermal particle
            Kokkos::atomic_add(&prtl_to_inject(),(charge > 0) ? -ONE:ONE);
        }
    }
 // ------------------- print diagnostic, default false -----------------
        
    ux1(p) = ufin[0];
    ux2(p) = ufin[1];
    ux3(p) = ufin[2];
        

	return;
    }   // operator 
  }; // UpdateVelKernel 
}// namespace mcp



#endif // KERNELS_POSTSTEP_HPP
