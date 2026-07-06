#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/field_setter.h"
#include "framework/parameters/parameters.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"
#include <Kokkos_Random.hpp>

#include <algorithm>
#include <utility>

#include "interpolate.hpp"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    /*
      Sets up magnetic and electric field components for the simulation.
      Must satisfy E = -v x B for Lorentz Force to be zero.

      @param bmag: magnetic field scaling
      @param btheta: magnetic field polar angle
      @param bphi: magnetic field azimuthal angle
      @param drift_ux: drift velocity in the x direction
    */
    InitFields(real_t bmag, real_t btheta, real_t bphi, real_t drift_ux)
      : Bmag { bmag }
      , Btheta { btheta * static_cast<real_t>(convert::deg2rad) }
      , Bphi { bphi * static_cast<real_t>(convert::deg2rad) }
      , Vx { drift_ux } {}

    // magnetic field components
    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return Bmag * math::cos(Btheta);
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    // electric field components
    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return -Vx * Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return Vx * Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

  private:
    const real_t Btheta, Bphi, Vx, Bmag;
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    // compatibility traits for the problem generator
    static constexpr auto engines { ::traits::pgen::compatible_with<SimEngine::type::SRPIC> {} };
    static constexpr auto metrics { ::traits::pgen::compatible_with<Metric::Minkowski> {} };
    static constexpr auto dimensions {
            ::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D> {}
    };

    // for easy access to variables in the child class

    const SimulationParams& params;

    Metadomain<S, M>& global_domain;

    // domain properties
    const real_t  global_xmin, global_xmax;
    // gas properties
    const real_t  drift_ux, temperature, temperature_ratio, filling_fraction;
    // injector properties
    const real_t  injector_velocity, injection_start, dt;
    const int     injection_frequency;
    // magnetic field properties
    real_t        Btheta, Bphi, Bmag;
    // properties of thermalization layer
    real_t        thermal_bath_offset, thermal_bath_width;
    InitFields<D> init_flds;
    npart_t       stride;
    real_t        x_track, x_track_offset;
    real_t        avg_coeff;
    ncells_t      i1_out_low, i1_out_up, i2_out_low, i2_out_up;
    bool          random_track;
    bool          tracked = false;
    bool          first_step = true;
    bool          interp_prtl;
    array_t<real_t***> cbuff;



    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : params { p }
      , global_domain { global_domain }
      , global_xmin { global_domain.mesh().extent(in::x1).first }
      , global_xmax { global_domain.mesh().extent(in::x1).second }
      , drift_ux { p.template get<real_t>("setup.drift_ux") }
      , temperature { p.template get<real_t>("setup.temperature") }
      , temperature_ratio { p.template get<real_t>("setup.temperature_ratio") }
      , Bmag { p.template get<real_t>("setup.Bmag", ZERO) }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Bmag, Btheta, Bphi, drift_ux }
      , filling_fraction { p.template get<real_t>("setup.filling_fraction", 1.0) }
      , injector_velocity { p.template get<real_t>("setup.injector_velocity", 1.0) }
      , injection_start { p.template get<real_t>("setup.injection_start", 0.0) }
      , injection_frequency { p.template get<int>("setup.injection_frequency", 100) }
      , thermal_bath_offset { p.template get<real_t>(
          "setup.thermal_bath_offset",
          2.0) }
      , thermal_bath_width { p.template get<real_t>("setup.thermal_bath_width",
                                                    20.0) }
      , dt { p.template get<real_t>("algorithms.timestep.dt") }
      , stride { p.template get<npart_t>("output.particles.stride", 2) }
      , x_track { p.template get<real_t>("setup.x_track",0.0) }
      , x_track_offset { p.template get<real_t>("setup.x_track_offset",0.0) }
      , tracked { p.template get<bool>("setup.tracked", false) }
      , random_track { p.template get<bool>("setup.random_track", true) }
      , avg_coeff { p.template get<real_t>("setup.avg_coeff", 0.01) }
      , interp_prtl { p.template get<bool>("setup.interp_prtl", false) } {}


    inline PGen() {}

    auto MatchFields(real_t time) const -> InitFields<D> {
      return init_flds;
    }

    auto FixFieldsConst(const bc_in&,
                        const em& comp) const -> std::pair<real_t, bool> {
      if (comp == em::ex1) {
        return { init_flds.ex1({ ZERO }), true };
      } else if (comp == em::ex2) {
        return { ZERO, true };
      } else if (comp == em::ex3) {
        return { ZERO, true };
      } else if (comp == em::bx1) {
        return { init_flds.bx1({ ZERO }), true };
      } else if (comp == em::bx2) {
        return { init_flds.bx2({ ZERO }), true };
      } else if (comp == em::bx3) {
        return { init_flds.bx3({ ZERO }), true };
      } else {
        raise::Error("Invalid component", HERE);
        return { ZERO, false };
      }
    }

    inline void InitPrtls(Domain<S, M>& domain) {

      /*
       *  Plasma setup as partially filled box
       *
       *  Plasma setup:
       *
       * global_xmin                            global_xmax
       * |                                      |
       * V                                      V
       * |:::::::::::|..........................|
       *             ^
       *             |
       *        filling_fraction
       */

      // minimum and maximum position of particles
      real_t xg_min = global_xmin;
      real_t xg_max = global_xmin + filling_fraction * (global_xmax - global_xmin);

      // define box to inject into
      boundaries_t<real_t> box;
      // loop over all dimensions
      for (auto d { 0u }; d < (unsigned int)M::Dim; ++d) {
        // compute the range for the x-direction
        if (d == static_cast<decltype(d)>(in::x1)) {
          box.push_back({ xg_min, xg_max });
        } else {
          // inject into full range in other directions
          box.push_back(Range::All);
        }
      }

      // define temperatures of species
      const auto temperatures = std::make_pair(temperature,
                                               temperature_ratio * temperature);
      // define drift speed of species
      const auto drifts       = std::make_pair(
        std::vector<real_t> { -drift_ux, ZERO, ZERO },
        std::vector<real_t> { -drift_ux, ZERO, ZERO });

      
      std::vector<npart_t> n_prtl_prev ={ 
                      domain.species[0].npart(),
                      domain.species[1].npart()
                      };
      // inject particles
      arch::InjectUniformMaxwellians<S, M>(params,
                                           domain,
                                           ONE,
                                           temperatures,
                                           { 1, 2 },
                                           drifts,
                                           false,
                                           box);
      for (auto& s : { 1u, 2u }) {
        auto& species = domain.species[s - 1];
        auto  pld_i      = species.pld_i;
        auto  tag        = species.tag;
        auto  npart_now  = species.npart();
        auto  npart_prev = n_prtl_prev[s-1];
        auto& random_pool = domain.random_pool();
        const auto stride_dev = stride;
        const auto rand_track = random_track;

        Kokkos::parallel_for(
            "Modify Prtl id",
            CreateParticleRangePolicy<Dim::_1D>({npart_prev}, {npart_now}),
            Lambda(prtlidx_t p) {
              if (tag(p) == ParticleTag::dead) {
                return; }
              npart_t plus_offset;
              
              // if track random prtls:
              // pldi % stride = 0,1,2,... stride -1, 1/stride prtls are tracked
              if (rand_track){
                  auto rand_gen = random_pool.get_state();
                  plus_offset = (npart_t)(rand_gen.urand(stride_dev));
                  random_pool.free_state(rand_gen);
              }
              else {
              // pldi % stride = 1,2,3,...stride-1, so that by default new injected prtls are not tracked
                if (s == 2){ plus_offset = 1;}
                else {
                    auto rand_gen = random_pool.get_state();
                    plus_offset = (npart_t)(rand_gen.urand(stride_dev-1)) + 1;
                    random_pool.free_state(rand_gen);
                }
              }
              pld_i(p, pldi::spcCtr) = (npart_t)(stride_dev * pld_i(p, pldi::spcCtr)+ plus_offset);
              return; 
              }
          );
        }
    }

    void CustomFieldOutput(const std::string&   name,
                           ndfield_t<M::Dim, 6> buffer,
                           idx_t              index,
                           timestep_t,
                           simtime_t,
                           const Domain<S, M>& domain) {
      // only 2d is implemented 
      if constexpr (D == Dim::_2D) {
        if (name == "avgE1") {
          Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, index),
                            Kokkos::subview(cbuff, Kokkos::ALL, Kokkos::ALL, 0));
        } else if (name == "avgE2") {
          Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, index),
                            Kokkos::subview(cbuff, Kokkos::ALL, Kokkos::ALL, 1));
        } else if (name == "avgE3") {
          Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, index),
                            Kokkos::subview(cbuff, Kokkos::ALL, Kokkos::ALL, 2));
        } else if (name == "avgB1") {
          Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, index),
                            Kokkos::subview(cbuff, Kokkos::ALL, Kokkos::ALL, 3));
        } else if (name == "avgB2") {
          Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, index),
                            Kokkos::subview(cbuff, Kokkos::ALL, Kokkos::ALL, 4));
        } else if (name == "avgB3") {
          Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, index),
                            Kokkos::subview(cbuff, Kokkos::ALL, Kokkos::ALL, 5));
        } else {
          raise::Error("Custom output not provided", HERE);
        }
      }
    }


    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {

      /*
       *  Replenish plasma in a moving injector
       *
       *  Injector setup:
       *
       * global_xmin           purge/replenish  global_xmax
       * |         x_init            |          |
       * V           v               V          V
       * |:::::::::::;::::::::::|\\\\\\\\|......|
       *                       xmin    xmax
       *                                 ^
       *                                 |
       *                           moving injector
      */
      // load raw electromagnetic field for custom output
      const auto EB = domain.fields.em;

      // calculate running average of ElectroMagnetic fields
      if (first_step) {

        // allocate the array at time = 0
        if constexpr (D == Dim::_2D) {
          cbuff          = array_t<real_t***>("cbuff",
                                              domain.mesh.n_all(in::x1),
                                              domain.mesh.n_all(in::x2),
                                              6);
          auto cbuff_loc = cbuff;
	  
          Kokkos::parallel_for(
            "FillCbuff",
            domain.mesh.rangeActiveCells(),
            Lambda(cellidx_t i1, cellidx_t i2) {
              cbuff_loc(i1, i2, 0) = EB(i1, i2, em::ex1);
              cbuff_loc(i1, i2, 1) = EB(i1, i2, em::ex2);
              cbuff_loc(i1, i2, 2) = EB(i1, i2, em::ex3);
              cbuff_loc(i1, i2, 3) = EB(i1, i2, em::bx1);
              cbuff_loc(i1, i2, 4) = EB(i1, i2, em::bx2);
              cbuff_loc(i1, i2, 5) = EB(i1, i2, em::bx3);
            });
        }

	// empty pld_r
        for (auto& s :
               { 1u }) { // only calculate the em fields interpolated for electrons
            auto& species = domain.species[s - 1];
            auto  pld_r   = species.pld_r;
            auto  tag     = species.tag;
            Kokkos::parallel_for(
              "EmptyPLDR",
              species.rangeActiveParticles(),
              Lambda(prtlidx_t p) {
                if (tag(p) == ParticleTag::dead) {
                  return;
                }
                pld_r(p, 0) = 0.0;
                pld_r(p, 1) = 0.0;
                pld_r(p, 2) = 0.0;
                pld_r(p, 3) = 0.0;
                pld_r(p, 4) = 0.0;
                pld_r(p, 5) = 0.0;
                return;
              });
          }

	// reset status
        first_step = false;
      } else {

        if constexpr (D == Dim::_2D) {
          auto       cbuff_loc     = cbuff;
          const auto avg_coeff_loc = avg_coeff;
          Kokkos::parallel_for(
            "FillCbuff_runningavg",
            domain.mesh.rangeActiveCells(),
            Lambda(cellidx_t i1, cellidx_t i2) {
	      cbuff_loc(i1, i2, 0) = std::isfinite(cbuff_loc(i1, i2, 0)) ? cbuff_loc(i1, i2, 0): 0.0;
	      cbuff_loc(i1, i2, 1) = std::isfinite(cbuff_loc(i1, i2, 1)) ? cbuff_loc(i1, i2, 1): 0.0;
	      cbuff_loc(i1, i2, 2) = std::isfinite(cbuff_loc(i1, i2, 2)) ? cbuff_loc(i1, i2, 2): 0.0;
	      cbuff_loc(i1, i2, 3) = std::isfinite(cbuff_loc(i1, i2, 3)) ? cbuff_loc(i1, i2, 3): 0.0;
	      cbuff_loc(i1, i2, 4) = std::isfinite(cbuff_loc(i1, i2, 4)) ? cbuff_loc(i1, i2, 4): 0.0;
	      cbuff_loc(i1, i2, 5) = std::isfinite(cbuff_loc(i1, i2, 5)) ? cbuff_loc(i1, i2, 5): 0.0;
              
	      cbuff_loc(i1, i2, 0) = cbuff_loc(i1, i2, 0) * (1.0 - avg_coeff_loc) +
                                     EB(i1, i2, em::ex1) * avg_coeff_loc;
              cbuff_loc(i1, i2, 1) = cbuff_loc(i1, i2, 1) * (1.0 - avg_coeff_loc) +
                                     EB(i1, i2, em::ex2) * avg_coeff_loc;
              cbuff_loc(i1, i2, 2) = cbuff_loc(i1, i2, 2) * (1.0 - avg_coeff_loc) +
                                     EB(i1, i2, em::ex3) * avg_coeff_loc;
              cbuff_loc(i1, i2, 3) = cbuff_loc(i1, i2, 3) * (1.0 - avg_coeff_loc) +
                                     EB(i1, i2, em::bx1) * avg_coeff_loc;
              cbuff_loc(i1, i2, 4) = cbuff_loc(i1, i2, 4) * (1.0 - avg_coeff_loc) +
                                     EB(i1, i2, em::bx2) * avg_coeff_loc;
              cbuff_loc(i1, i2, 5) = cbuff_loc(i1, i2, 5) * (1.0 - avg_coeff_loc) +
                                     EB(i1, i2, em::bx3) * avg_coeff_loc;
            });
        }
      }

      if constexpr (D == Dim::_2D) {
        if (interp_prtl) {
          // calculate interpolated EM field on electrons
          for (auto& s :
               { 1u }) { // only calculate the em fields interpolated for electrons
            auto& species = domain.species[s - 1];
            auto  pld_r   = species.pld_r;
            auto  tag     = species.tag;
	    auto  u1      = species.ux1;
	    auto  u2      = species.ux2;
	    auto  u3      = species.ux3;
	    auto  dt_loc  = dt;
	    auto& mesh    = domain.mesh;
	    auto  i1      = species.i1;
	    auto  i2      = species.i2;
	    auto  dx1     = species.dx1;
	    auto  dx2     = species.dx2;
	    auto  mass    = species.mass();
	    auto  charge  = species.charge();

            kernels::user::InterpolateKernel<M, D, 2u> Interpolator(species.i1,
                                                                    species.i2,
                                                                    species.i3,
                                                                    species.dx1,
                                                                    species.dx2,
                                                                    species.dx3,
                                                                    EB);
            Kokkos::parallel_for(
              "InterpField",
              species.rangeActiveParticles(),
              Lambda(prtlidx_t p) {
                if (tag(p) == ParticleTag::dead) {
                  return;
                }
                vec_t<Dim::_3D> e_interp {ZERO}, b_interp {ZERO};
		vec_t<Dim::_3D> e0 {ZERO}, b0 {ZERO};
                Interpolator.InterpolatedEMFields(p, e_interp, b_interp);
              
	        // Convert fields to physical unit	
	        coord_t<D> xp_Cd {ZERO};
		xp_Cd[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p)); 
		xp_Cd[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p)); 
		mesh.metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, e_interp, e0);
		mesh.metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, b_interp, b0);
		auto ePhys = e0;
		auto bPhys = b0;

                real_t dwxpar, dwypar, dwzpar, dwxperp, dwyperp, dwzperp;
		const real_t gm0 = math::sqrt(u1(p)*u1(p)+u2(p)*u2(p)+u3(p)*u3(p)+1);
		real_t gm_tmp;
                const real_t bnorm = math::sqrt(NORM_SQR(bPhys[0], bPhys[1],bPhys[2])); 
		// Boris
		real_t COEFF { dt_loc * HALF * (charge / mass) };
                e0[0] *= COEFF;
                e0[1] *= COEFF;
                e0[2] *= COEFF;
                vec_t<Dim::_3D> u0 { u1(p) + e0[0],
                                     u2(p) + e0[1],
                                     u3(p) + e0[2] };
                
          
                COEFF *= ONE / math::sqrt(ONE + NORM_SQR(u0[0], u0[1], u0[2]));
                gm_tmp =  math::sqrt(ONE + NORM_SQR(u0[0], u0[1], u0[2]));
	        // dimensionless velocity for first half work	
		auto beta0=  u0;
		beta0[0] = HALF*(beta0[0] / gm_tmp + u1(p) / gm0);
		beta0[1] = HALF*(beta0[1] / gm_tmp + u2(p) / gm0);
		beta0[2] = HALF*(beta0[2] / gm_tmp + u3(p) / gm0);

                b0[0] *= COEFF;
                b0[1] *= COEFF;
                b0[2] *= COEFF;
                COEFF  = TWO / (ONE + NORM_SQR(b0[0], b0[1], b0[2]));
          
                const vec_t<Dim::_3D> us {
                  (u0[0] + CROSS_x1(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF,
                  (u0[1] + CROSS_x2(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF,
                  (u0[2] + CROSS_x3(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF
                };
         
	        // u_plus	
                u0[0] += CROSS_x1(us[0], us[1], us[2], b0[0], b0[1], b0[2]);
                u0[1] += CROSS_x2(us[0], us[1], us[2], b0[0], b0[1], b0[2]);
                u0[2] += CROSS_x3(us[0], us[1], us[2], b0[0], b0[1], b0[2]);
		auto beta1=  u0;
	        gm_tmp =  math::sqrt(ONE + NORM_SQR(u0[0], u0[1], u0[2]));
		
                // u_new
		u0[0] += e0[0];
                u0[1] += e0[1];
                u0[2] += e0[2];
		const real_t gm1 = math::sqrt(ONE + NORM_SQR(u0[0], u0[1], u0[2]));	
	        
		// dimensionless velocity for first half work	
		beta1[0] = HALF*(beta1[0] / gm_tmp + u0[0] / gm1);
		beta1[1] = HALF*(beta1[1] / gm_tmp + u0[1] / gm1);
		beta1[2] = HALF*(beta1[2] / gm_tmp + u0[2] / gm1);

                // end Boris
	
		// normalized
		bPhys[0] = bPhys[0] / bnorm;
		bPhys[1] = bPhys[1] / bnorm;
		bPhys[2] = bPhys[2] / bnorm;
		
		// real_t Epar = (ePhys[0]*bPhys[0]+ePhys[1]*bPhys[1]+ePhys[2]*bPhys[2]);

		// dwxpar = HALF*(beta0[0]+beta1[0]) * Epar * bPhys[0];
		// dwypar = HALF*(beta0[1]+beta1[1])  * Epar * bPhys[1] ;
		// dwzpar = HALF*(beta0[2]+beta1[2])  * Epar * bPhys[2] ;
		// dwxperp =HALF*(beta0[0]+beta1[0])   * (ePhys[0] - Epar * bPhys[0]);
		// dwyperp =HALF*(beta0[1]+beta1[1])  * (ePhys[1] - Epar * bPhys[1]);
		// dwzperp =HALF*(beta0[2]+beta1[2]) * (ePhys[2] - Epar * bPhys[2]);
	         
                real_t Epar = (e0[0]*bPhys[0]+e0[1]*bPhys[1]+e0[2]*bPhys[2]);

		dwxpar = (beta0[0]+beta1[0]) * Epar * bPhys[0];
		dwypar = (beta0[1]+beta1[1]) * Epar * bPhys[1];
		dwzpar = (beta0[2]+beta1[2]) * Epar * bPhys[2];
		dwxperp =(beta0[0]+beta1[0]) * (e0[0] - Epar * bPhys[0]);
		dwyperp =(beta0[1]+beta1[1]) * (e0[1] - Epar * bPhys[1]);
		dwzperp =(beta0[2]+beta1[2]) * (e0[2] - Epar * bPhys[2]);

                pld_r(p, 0) += math::isfinite(dwxpar) ? dwxpar:0.0;
                pld_r(p, 1) += math::isfinite(dwypar) ? dwypar:0.0;
                pld_r(p, 2) += math::isfinite(dwzpar) ? dwzpar:0.0;
                pld_r(p, 3) += math::isfinite(dwxperp) ? dwxperp:0.0;
                pld_r(p, 4) += math::isfinite(dwyperp) ? dwyperp:0.0;
                pld_r(p, 5) += math::isfinite(dwzperp) ? dwzperp:0.0;
		
		return;
              });
          }
        }
      }

 

      // compute the maximum x position of particles in the domain
      const auto local_xmax = domain.mesh.extent(in::x1).second;

      // initial position of injector
      const auto x_init = global_xmin +
                          filling_fraction * (global_xmax - global_xmin);

      // compute the position of the injector after the current timestep
      auto xmax = x_init+injector_velocity *
              (std::max<real_t>(time - injection_start, ZERO) + dt);
      auto xmax_prev = x_init+injector_velocity *
              (std::max<real_t>(time - injection_start - dt* injection_frequency, ZERO) + dt);

      if (xmax_prev >= global_xmax) {
        xmax_prev = global_xmax;
      }

      if (xmax >= global_xmax) {
        xmax = global_xmax;
      }

      // compute the beginning of the injected region
      auto xmin = xmax_prev - injection_frequency * dt * drift_ux;
      if (xmin <= global_xmin) {
        xmin = global_xmin;
      }
      const auto& mesh = domain.mesh;
      
      // make sure we are at an injection step and that the injection has started and is within the current domain
      if ((step % injection_frequency == 0) && (step > 1) && ( local_xmax > xmin ) ) {
      
              /*
                      Inject slab of fresh plasma
              */

              // define box to inject into
              boundaries_t<real_t> inj_box;
              // loop over all dimension
              for (auto d = 0u; d < M::Dim; ++d) {
                   if (d == 0) {
                     inj_box.push_back({ xmin, xmax });
                   } else {
                     inj_box.push_back(Range::All);
                   }
              }

              // set temperature to zero to inject cold plasma, which is more numerically stable for the shock setup
              const auto temperatures = std::make_pair(ZERO, ZERO);
              const auto drifts       = std::make_pair(
                    std::vector<real_t> { -drift_ux, ZERO, ZERO },
                    std::vector<real_t> { -drift_ux, ZERO, ZERO });
              
              
              std::vector<npart_t> n_prtl_prev =  {
                      domain.species[0].npart(),
                      domain.species[1].npart()
                      };

              arch::InjectUniformMaxwellians<S, M>(params,
                                                   domain,
                                                   ONE,
                                                   temperatures,
                                                   { 1, 2 },
                                                   drifts,
                                                   false,
                                                   inj_box); 
              
                for (auto& s : { 1u, 2u }) {
                 auto& species = domain.species[s - 1];
                 auto  pld_i      = species.pld_i;
                 auto  tag        = species.tag;
                 auto  npart_now  = species.npart();
                 auto  npart_prev = n_prtl_prev[s-1];
                 auto& random_pool = domain.random_pool();
                 const auto stride_dev = stride;
                 const auto rand_track = random_track;

                 Kokkos::parallel_for(
                     "Modify Prtl id",
                     CreateParticleRangePolicy<Dim::_1D>({npart_prev}, {npart_now}),
                     Lambda(prtlidx_t p) {
                       if (tag(p) == ParticleTag::dead) {
                         return; }
                       npart_t plus_offset;
                       
                       // if track random prtls:
                       // pldi % stride = 0,1,2,... stride -1, 1/stride prtls are tracked
                       if (rand_track){
                           auto rand_gen = random_pool.get_state();
                          plus_offset = (npart_t)(rand_gen.urand(stride_dev));
                          random_pool.free_state(rand_gen);
                       }
                       else {
                       // pldi % stride = 1,2,3,...stride-1, so that by default new injected prtls are not tracked
                         if (s == 2){ plus_offset = 1;}
                         else {
                             auto rand_gen = random_pool.get_state();
                             plus_offset = (npart_t)(rand_gen.urand(stride_dev-1)) + 1;
                             random_pool.free_state(rand_gen);
                         }
                       }
                       pld_i(p, pldi::spcCtr) = (npart_t)(stride_dev * pld_i(p, pldi::spcCtr)+ plus_offset);
                       return; 
                       }
                   );
                 }    
           


            
      }

              if (!random_track && !tracked){
                  Kokkos::printf("Begin Tracking\n");
                  // check if particles are to track
                  // could define other rules to put prtl into tracking
                  for (auto& s : { 1u, 2u }) {
                    auto& species = domain.species[s - 1];
                    auto  pld_i      = species.pld_i;
                    auto  i1      = species.i1;
                    auto  dx1     = species.dx1;
                    auto& mesh = domain.mesh;
                    auto  x_track_min = x_track;
                    auto x_track_max = x_track + x_track_offset;
                    const auto stride_dev = stride;
                
                        Kokkos::parallel_for(
                            "If prtl is to track",
                            species.rangeActiveParticles(),
                            Lambda(prtlidx_t p) {
                              const auto x_Cd = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
                              const auto x_Ph = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(x_Cd);
                              if ((x_Ph < x_track_max) && (x_Ph > x_track_min)){ // prtl is in the box
                                  // track prtl
                                  pld_i(p, pldi::spcCtr) = (npart_t)((npart_t)(pld_i(p, pldi::spcCtr) / stride_dev) * stride_dev);
                              } 
                              else {
                                  // if it is tracked, untrack
                                  if (pld_i(p, pldi::spcCtr) % stride_dev == 0){
                                      pld_i(p, pldi::spcCtr) += 1;                                
                                  }
                              }
                              return; 
                            }
                   ); }
   
                   tracked = true; // to avoid execute this block again
                   Kokkos::printf("Tracking finished.\n");
                   Kokkos::printf("global xmax is %.2f\n",global_xmax);
                }




        // compute the beginning of the thermalisation region
      auto therm_xmin = xmin - thermal_bath_offset - thermal_bath_width;
      if (therm_xmin <= global_xmin) {
        therm_xmin = global_xmin;
      }

      // compute the end of the thermalisation region
      auto therm_xmax = xmin - thermal_bath_offset;
      if (therm_xmax <= global_xmin) {
        therm_xmax = global_xmin;
      }


      if ( local_xmax < therm_xmin ) {
        // if the maximum x position of particles in the domain is less than the beginning of the thermalisation region, 
        // we can skip the rest of the function since there are no particles to thermalise
        return;
      }


      // same maxwell distribution as above
      const auto mass_1   = domain.species[0].mass();
      const auto mass_2   = domain.species[1].mass();
      const auto T_e  = temperature / mass_1;
      const auto T_p  = temperature_ratio * temperature / mass_2;

      const auto maxwellian_e = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        T_e, { -drift_ux, ZERO, ZERO });
      const auto maxwellian_p = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        T_p,
        { -drift_ux, ZERO, ZERO });
      const auto maxwellian_zero = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        ZERO, { -drift_ux, ZERO, ZERO });


      for (auto& s : { 1u, 2u }) {
        auto& species = domain.species[s - 1];
        auto  i1      = species.i1;
        auto  dx1     = species.dx1;
        auto  ux1     = species.ux1;
        auto  ux2     = species.ux2;
        auto  ux3     = species.ux3;
        auto  tag     = species.tag;

        Kokkos::parallel_for(
          "ResetParticles",
          species.rangeActiveParticles(),
          Lambda(prtlidx_t p) {
            if (tag(p) == ParticleTag::dead) {
              return;
            }

                  const auto x_Cd = static_cast<real_t>(i1(p)) +
                              static_cast<real_t>(dx1(p));
            const auto x_Ph = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(
              x_Cd);

            const coord_t<M::Dim> x_dummy { ZERO };

            // sample from maxwellian if the particle is in the thermalisation region
            if (x_Ph < therm_xmax && x_Ph > therm_xmin) {
              vec_t<Dim::_3D> v_T { ZERO }, v_Cd { ZERO };
              if (s == 1u) {
                maxwellian_e(x_dummy, v_T);
              } else {
                maxwellian_p(x_dummy, v_T);
              }
              mesh.metric.template transform_xyz<Idx::T, Idx::XYZ>(x_dummy, v_T, v_Cd);
              ux1(p) = v_Cd[0];
              ux2(p) = v_Cd[1];
              ux3(p) = v_Cd[2];
            }
                 // sample from maxwellian if the particle is in the thermalisation region
            if (x_Ph > therm_xmax) {
              vec_t<Dim::_3D> v_T { ZERO }, v_Cd { ZERO };
              
                    maxwellian_zero(x_dummy, v_T);
              
                    mesh.metric.template transform_xyz<Idx::T, Idx::XYZ>(x_dummy, v_T, v_Cd);
              ux1(p) = v_Cd[0];
              ux2(p) = v_Cd[1];
              ux3(p) = v_Cd[2];
            }

          });
        }
    }
  };
} // namespace user
#endif

