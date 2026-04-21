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
    const real_t x_wait, x_wait_l;
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
                  real_t                         x_wait_l,
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
      , x_wait_l {x_wait_l}
      , nu_coeff {nu_coeff}
      , charge {charge}
      , prtl_to_inject {prtl_to_inject}
      , random_pool {random_pool}
      , DEBUG {DEBUG} {
          weibel_kernel = kernel::weibel::WeibelKernel(shock_filling_fraction, global_min, global_max, drift_ux, Lsh);
      }


// Calculate cross product of vectors
Inline void cross(real_t v11,real_t v12,real_t v13, real_t v21, real_t v22, real_t v23, real_t &res1, real_t &res2, real_t &res3) const {
    res1 = v12 * v23 - v13 * v22;
    res2 = v13 * v21 - v11 * v23;
    res3 = v11 * v22 - v12 * v21;
}


// Rotate vector for an angle around a given axis
Inline void rotate(real_t v1, real_t v2, real_t v3,
                   real_t k1, real_t k2, real_t k3,
                   real_t &res1,real_t &res2,real_t &res3,
                   real_t rotangle) const {
        real_t normk = math::sqrt(k1*k1 +k2*k2+k3*k3);
        if (normk <= 0.0) {
	    res1 = v1;
	    res2 = v2;
	    res3 = v3;
	    return;
	}
        k1 = k1/normk; 
        k2 = k2/normk; 
        k3 = k3/normk; 
	// Rodriguez Rotation Formula
        real_t v_new1, v_new2, v_new3;
        real_t term11, term12, term13;
        real_t term21, term22, term23;
        real_t term31, term32, term33;
        real_t kxv1, kxv2, kxv3;   
        real_t normv, normvnew;
        
        cross(k1,k2,k3,v1,v2,v3,kxv1,kxv2,kxv3); 
           
        term11 = v1 * math::cos(rotangle);
        term12 = v2 * math::cos(rotangle);
        term13 = v3 * math::cos(rotangle);
        term21 = k1 * (k1*v1+k2*v2+k3*v3)* (1-  math::cos(rotangle));
        term22 = k2 * (k1*v1+k2*v2+k3*v3)* (1-  math::cos(rotangle));
        term23 = k3 * (k1*v1+k2*v2+k3*v3)* (1-  math::cos(rotangle));
        term31 = kxv1 *  math::sin(rotangle);
        term32 = kxv2 *  math::sin(rotangle);
        term33 = kxv3 *  math::sin(rotangle);
        
        v_new1 = term11+term21+term31; 
        v_new2 = term12+term22+term32; 
        v_new3 = term13+term23+term33;
        normv = math::sqrt(v1*v1+v2*v2+v3*v3); 
        normvnew = math::sqrt(v_new1*v_new1+v_new2*v_new2+v_new3*v_new3);
        res1 = v_new1 * normv / normrvnew;
        res2 = v_new2 * normv / normrvnew;
        res3 = v_new3 * normv / normrvnew;
    }

//Lorentz transformation
Inline void boostVel(real_t u1, real_t u2, real_t u3,
                     real_t brel, real_t LFrel,
                     real_t &res1,real_t &res2,real_t &res3) const{
        // assume that brel is the relative velocity between the moving frame and the lab frame
        real_t gm = math::sqrt(u1*u1+u2*u2+u3*u3 + 1);
        res1 = LFrel * (u1 - gm * brel);
        res2 = u2;
        res3 = u3;
    }

  
// Manipulate particle velocity 
Inline void operator()(index_t p) const {
        if (tag(p) == ParticleTag::dead){
            return;
        }
        real_t dtw;
	real_t u1,u2,u3;
	real_t uw1,uw2,uw3;
	real_t ufin1,ufin2,ufin3;
        real_t k1,k2,k3;
        real_t u_weibel, duwdx, brel, lfrel;
        real_t theta, phi, rotangle;
        real_t nu, gm, gmw;
        // real_t esc;

        // the velocity in lab frame
        u1 = ux1(p);
        u2 = ux2(p);
        u3 = ux3(p);

        // get particle location
        real_t x_prtl = ZERO;
        if constexpr (D == Dim::_1D) {
          coord_t<Dim::_1D> x_Cd { ZERO };
          x_Cd[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));

          coord_t<Dim::_1D> x_Ph { ZERO };
          metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
        
          x_prtl = x_Ph[0];
        }
         
        gm = math::sqrt((u1*u1+u2*u2+u3*u3) + 1.); // Particle Lorentz factor in lab frame

        nu = nu0 * mi / mp * ((mi == mp) ? 1.0:nu_coeff); // TODO: define the model of scattering freq.

        u_weibel = weibel_kernel.getux(x_prtl);
        duwdx = weibel_kernel.getdudx(x_prtl);
        brel = -math::sqrt(math::pow(u_weibel, 2.)/(1. + math::pow(u_weibel, 2.)));  // relative speed between lab and weibel frame (always negative)
        lfrel = 1.0 / math::sqrt(1 - math::pow(brel, 2.));     // relative Lorentz factor between lab and weibel frame
       
	dtw = dt * lfrel * (1 - u1/gm * brel);     // timestep in weibel frame

        boostVel(u1, u2, u3, brel, lfrel, uw1,uw2,uw3);
        gmw = math::sqrt((uw1*uw1+uw2*uw2+uw3*uw3) +1.); // Particle Lorentz factor in Weibel frame
        
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
	k1 = math::sin(theta) * math::cos(phi);
        k2 = math::sin(theta) * math::sin(phi);
        k3 = math::cos(theta);

        rotate(uw1,uw2,uw3, k1,k2,k3, uw1,uw2,uw3, rotangle); // Apply scattering
         
	
	// ------------------- final velocity -----------------
        boostVel(uw1, uw2, uw3, -brel, lfrel, ufin1, ufin2, ufin3);


    // ------------------- reflect beams to avoid being absorbed by right boundary ---------------

    assert(x_wait_l < x_wait); // x_wait_l is the waiting zone at left boundary, where prtls should be frozen and gyrate. 
   
    if ((x_prtl <= x_wait) && (x_prtl >= x_wait_l)){
        pld_i(p, user_plds::pldi) = 0; // reset the tag whether prtl is inside the isolated box
    }
    else if (x_prtl > x_wait) { // prtl enters the isolated box right
        if (DEBUG){
            // Kokkos::printf("Prtl in isolated box , x=%.4f, xwait=%.4f\n", x_prtl, x_wait );
        }
        if (pld_i(p, user_plds::pldi) == 0) // if it is still active (enters the box for the first time)
        {
            pld_r(p,user_plds::pldr) = x_prtl - x_wait; // the real position relative to x_wait.
            pld_i(p,user_plds::pldi) = 1; // mark prtl as in the right isolated box
            weight(p) = ZERO; // not letting prtl affect the field
            // for particle entering box for first time, initialize. 
            Kokkos::atomic_add(&prtl_to_inject(),(charge > 0) ? ONE: -ONE);
        }
        else{
            // the prtl has already been in waiting box. Update its true position. 
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
    else if (x_prtl < x_wait_l){ //prtl enters the isolated box left
        if (pld_i(p, user_plds::pldi) == 0) // if it is still active (enters the box for the first time)
        {
            pld_r(p,user_plds::pldr) = x_prtl - x_wait_l; // the real position relative to x_wait.
            pld_i(p,user_plds::pldi) = 2; // mark prtl as in the right isolated box
            weight(p) = ZERO; // not letting prtl affect the field
            // for particle entering box for first time, initialize. 
        }
        else{
            pld_r(p,user_plds::pldr) += x_prtl - (x_wait_l + global_min)/2.0;
        }

        if  (pld_r(p, user_plds::pldr) < 0){ // prtl still outside the main simulation
            //drag prtl back to a safe place
            if constexpr (D == Dim::_1D) {
                coord_t<Dim::_1D> x_wait_l_Cd { ZERO };
                coord_t<Dim::_1D> x_wait_l_Ph { ZERO };
                x_wait_l_Ph[0] = static_cast<real_t>(x_wait_l/2 + global_min / 2);
                metric.template convert<Crd::Ph,Crd::Cd>(x_wait_l_Ph, x_wait_l_Cd);
                i1(p) = static_cast<int>(x_wait_l_Cd[0]);
                dx1(p) = x_wait_l_Cd[0] - static_cast<int>(x_wait_l_Cd[0]);
            }
            
            if (DEBUG){assert(weight(p) == ZERO);}
        }
        else { //  prtl leave the isolated box and return to the main simulation
            pld_i(p,user_plds::pldi) = 0; //  mark prtl as back
            if constexpr (D == Dim::_1D) {
                coord_t<Dim::_1D> x_wait_l_Cd { ZERO };
                coord_t<Dim::_1D> x_wait_l_Ph { ZERO };
                x_wait_l_Ph[0] = static_cast<real_t>(x_wait_l + pld_r(p, user_plds::pldr));
                metric.template convert<Crd::Ph,Crd::Cd>(x_wait_l_Ph, x_wait_l_Cd);
                i1(p) = static_cast<int>(x_wait_l_Cd[0]);
                dx1(p) = x_wait_l_Cd[0] - static_cast<int>(x_wait_l_Cd[0]);
            }
            if (DEBUG){assert(weight(p) == ZERO);}
            weight(p) = ONE; // to re-enable prtl feedback on fld

        }

        real_t rg = math::sqrt(uw1*uw1+uw2*uw2+uw3*uw3) * mp / ((Bmag+1e-10) * 4.0);
        real_t ldiff = (uw1*uw1+uw2*uw2+uw3*uw3) / math::abs(u_weibel) / nu;

        if (math::abs(pld_r(p, user_plds::pldr)) > math::min(rg, ldiff)){
            tag(p) = ParticleTag::dead; // kill particles too far away
        }
    }
 // ------------------- print diagnostic, default false -----------------
        
    ux1(p) = ufin1;
    ux2(p) = ufin2;
    ux3(p) = ufin3;

	return;
    }   // operator 
  }; // UpdateVelKernel 
}// namespace mcp



#endif // KERNELS_POSTSTEP_HPP
