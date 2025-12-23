#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/utils.h"
#include "archetypes/field_setter.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"
#include <Kokkos_Random.hpp>
#include "poststep.hpp"

#include <algorithm>
#include <utility>

namespace user {
  using namespace ntt;
  using vector_t = Kokkos::View<real_t[3]>;
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
    InitFields(real_t bmag, real_t bmag_lb, real_t btheta, real_t bphi, real_t drift_ux, real_t global_min, real_t global_max)
      : Bmag { bmag }
      , Bmag_lb {bmag_lb}
      , Btheta { btheta * static_cast<real_t>(convert::deg2rad) }
      , Bphi { bphi * static_cast<real_t>(convert::deg2rad) }
      , Vx { -drift_ux }
      , global_min {global_min}
      , global_max {global_max} {}
    // magnetic field components
    Inline auto bx1(const coord_t<D>& x) const -> real_t {
      // To ensure the left match boundary is consistent with the jump condition
      if (x[0] < global_min+ 0.01 * (global_max - global_min))
        return Bmag_lb * math::cos(Btheta);     // theta = pi/2, phi = 0, such that B only has Bz. 
      return Bmag * math::cos(Btheta);    
    }

    Inline auto bx2(const coord_t<D>& x) const -> real_t {
      if (x[0] < global_min+ 0.01 * (global_max - global_min))
      	return Bmag_lb * math::sin(Btheta) * math::sin(Bphi);
      return Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

    Inline auto bx3(const coord_t<D>& x) const -> real_t {
      if (x[0] < global_min+ 0.01 * (global_max - global_min))
      	return Bmag_lb * math::sin(Btheta) * math::cos(Bphi);
      return Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    // electric field components
    Inline auto ex1(const coord_t<D>& x) const -> real_t {
      if (x[0] < global_min+ 0.01 * (global_max - global_min))
          return ZERO;
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&  x) const -> real_t {
      if (x[0] < global_min+ 0.01 * (global_max - global_min))
      	return Vx / 4.0 * Bmag_lb * math::sin(Btheta) * math::cos(Bphi);
      return Vx * Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    Inline auto ex3(const coord_t<D>& x) const -> real_t {
      if (x[0] < global_min+ 0.01 * (global_max - global_min))
      	return -Vx / 4.0 * Bmag_lb * math::sin(Btheta) * math::sin(Bphi);
      return -Vx * Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

  private:
    const real_t Btheta, Bphi, Vx, Bmag, global_min, global_max, Bmag_lb;
  };


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




  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    // domain properties
    const real_t  global_xmin, global_xmax;
    // gas properties
    const real_t  drift_ux, temperature, temperature_ratio;  //, filling_fraction, filling_fraction_upstream;
    // injector properties
    const real_t  injection_start, dt, average_coeff;
    const int     injection_frequency;
    // magnetic field properties
    real_t        Btheta, Bphi, Bmag, Bmag_lb, shock_filling_fraction, Lsh;
    real_t        nu0, nu_coeff;
    InitFields<D> init_flds;
    array_t<real_t*> cbuff, cbuff2, cbuff3;
    const int                        random_seed;
    random_number_pool_t             random_pool;
    bool DEBUG;
    bool is_resuming=false;



    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_xmin { global_domain.mesh().extent(in::x1).first }
      , global_xmax { global_domain.mesh().extent(in::x1).second }
      , drift_ux { p.template get<real_t>("setup.drift_ux") } // the magnitude of upstream drift velocity
      , temperature { p.template get<real_t>("setup.temperature") } 
      , temperature_ratio { p.template get<real_t>("setup.temperature_ratio") }
      , Bmag { p.template get<real_t>("setup.Bmag", ZERO) }   // the upstream magnetic field
      , Bmag_lb { p.template get<real_t>("setup.Bmag_lb", ZERO) } // the magnetic field at left (downstream) boundary
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) } 
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Bmag, Bmag_lb, Btheta, Bphi, drift_ux , global_xmin, global_xmax} 
      , injection_start { p.template get<real_t>("setup.injection_start", 0.0) }
      , injection_frequency { p.template get<int>("setup.injection_frequency", 100) }
      , dt { p.template get<real_t>("algorithms.timestep.dt")}
      , average_coeff { p.template get<real_t>("setup.average_coeff", 0.01)} // running smooth, X_avg_1 = X_avg * (1-coeff) + coeff * X
      , shock_filling_fraction {p.template get<real_t>("setup.shock_filling_fraction",0.2)} // the relative location of the shock front
      , Lsh {p.template get<real_t>("setup.Lsh")} // the scale of shock transition layer
      , nu0 {p.template get<real_t>("setup.nu0")} // the scattering frequency of ions
      , nu_coeff {p.template get<real_t>("setup.nu_coeff", 1.0)}
      , random_seed { p.template get<int>("setup.seed", -1) } 
      , random_pool { init_pool(random_seed) }
      , DEBUG {p.template get<bool>("setup.DEBUG")} 
      , is_resuming {p.template get<bool>("checkpoint.is_resuming")} { // if is resuming, the smoothed quantities need to be initialized again


}

    inline PGen() {}

    auto MatchFields(real_t time) const -> InitFields<D> {
      return init_flds;
    }

    auto FixFieldsConst(const bc_in&, const em& comp) const
      -> std::pair<real_t, bool> {
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

    inline void InitPrtls(Domain<S, M>& local_domain) {
    }

    // Custom output
    void CustomFieldOutput(const std::string&    name,
                         ndfield_t<M::Dim, 6> buffer,
                         index_t              index,
                         timestep_t,
                         simtime_t,
                         const Domain<S, M>& domain) {
     // EM components smoothed by time
     if (name == "AvgEx") {
        if constexpr (D == Dim::_1D) {
              Kokkos::deep_copy(Kokkos::subview(buffer,Kokkos::ALL, index), cbuff);
        }
     } 
     if (name == "AvgEy") {
        if constexpr (D == Dim::_1D) {
              Kokkos::deep_copy(Kokkos::subview(buffer,Kokkos::ALL, index), cbuff2);
        }
     } 
     if (name == "AvgBz") {
        if constexpr (D == Dim::_1D) {
              Kokkos::deep_copy(Kokkos::subview(buffer,Kokkos::ALL, index), cbuff3);
        }
     } 
    
     // To test the normalization of the EM field
     if (name=="JptEx"){
        if constexpr (M::Dim == Dim::_1D) {
           const auto& mesh = domain.mesh;
           const auto& EM = domain.fields.em;
           Kokkos::parallel_for(
           "MyField",
           mesh.rangeActiveCells(),
           Lambda(index_t i1) {
              buffer(i1, index) = EM(i1, em::ex1);
           });
     }
    } 
    } // CustonFieldOutput
    

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      const auto& mesh = domain.mesh;
      // loop over prtl species
      // check if the injector should be active
      bool PRINT=false;
      if (DEBUG){
          if (step % injection_frequency == 0) {
            PRINT=true;
          }
      }
      
      // Apply stochastic scattering 
      for (auto& species : domain.species) {
          Kokkos::parallel_for(
             "Scatter and Inertial term",
             species.rangeActiveParticles(),
             kernel::mcp::UpdateVelKernel<M, D>(
                 domain.mesh.metric,
                 species.i1, species.dx1, species.ux1, species.ux2, species.ux3,
                 species.tag, species.mass(), domain.species[1].mass(), time, dt,  //domain.mesh.metric,
                 shock_filling_fraction,
                 global_xmin,
                 global_xmax,
                 drift_ux,
                 Lsh,
                 Bmag,
                 nu0,
                 nu_coeff,
                 domain.random_pool, 
                 PRINT
                 ));
         }


      // compute the mean electric field Ex
      if ((step == 0) || (is_resuming)){ // initialize cbuff value
        if constexpr (D == Dim::_1D) {
            cbuff = array_t<real_t*>("cbuff", domain.mesh.n_all(in::x1));
            cbuff2 = array_t<real_t*>("cbuff2", domain.mesh.n_all(in::x1));
            cbuff3 = array_t<real_t*>("cbuff3", domain.mesh.n_all(in::x1));
            auto cbuff_loc = cbuff;
            auto cbuff_loc2 = cbuff2; 
      	    Kokkos::parallel_for(
                 "FillCbuff",
                 mesh.rangeActiveCells(),
                 KOKKOS_LAMBDA(index_t i1) {
                       cbuff_loc(i1) = 0.0;
                       cbuff_loc2(i1)= 0.0; 
                  });    
        }
        is_resuming = false;
      }
       // To avoid warnings
       auto cbuff_loc = cbuff; 
       auto cbuff_loc2 = cbuff2; 
       auto cbuff_loc3 = cbuff3; 
       const auto EB = domain.fields.em;

       // updating average
        if constexpr (D == Dim::_1D) {
             const auto coeff = average_coeff;
             if (step < 100){      // Not smoothing for the initial few steps
                 Kokkos::parallel_for(
                     "average Ex",
                     mesh.rangeActiveCells(),
                     KOKKOS_LAMBDA(index_t i1){
                         cbuff_loc(i1) = EB(i1, em::ex1);
                         cbuff_loc2(i1) = EB(i1, em::ex2);
                         cbuff_loc3(i1) = EB(i1, em::bx3);
                     });
             }
             else {
                 Kokkos::parallel_for(
                     "average Ex",
                     mesh.rangeActiveCells(),
                     KOKKOS_LAMBDA(index_t i1){
                           cbuff_loc(i1) = cbuff_loc(i1) * (1.0-coeff)+ EB(i1, em::ex1) * coeff;
                           cbuff_loc2(i1) = cbuff_loc2(i1) * (1.0-coeff)+ EB(i1, em::ex2) * coeff;
                           cbuff_loc3(i1) = cbuff_loc3(i1) * (1.0-coeff)+ EB(i1, em::bx3) * coeff;

                     }
                  );
        }}
        // 2D and 3D not implemented
        if constexpr (D == Dim::_2D) {
        }
        if constexpr (D == Dim::_3D) {
        }      
      

      // check if the injector should be active
      if (step % injection_frequency != 0) {
        return;
      }


      auto xmax = global_xmax - (global_xmax - global_xmin) * 0.01;
      // compute the beginning of the injected region
      auto xmin = xmax - injection_frequency * dt * drift_ux / math::sqrt(drift_ux * drift_ux + ONE);//* (injector_velocity + drift_ux); 
      if (xmin <= global_xmin) {
          xmin = global_xmin;
      }
                                                                                                    

      // define indice range to reset fields
      boundaries_t<bool> incl_ghosts;
      for (auto d = 0; d < M::Dim; ++d) {
        incl_ghosts.push_back({ false, false });
      }

      // define box to reset fields
      boundaries_t<real_t> purge_box;
      // loop over all dimension
      for (auto d = 0u; d < M::Dim; ++d) {
        if (d == 0) {
          purge_box.push_back({ xmin, global_xmax });
        } else {
          purge_box.push_back(Range::All);
        }
      }

      const auto extent = domain.mesh.ExtentToRange(purge_box, incl_ghosts);
      tuple_t<std::size_t, M::Dim> x_min { 0 }, x_max { 0 };
      for (auto d = 0; d < M::Dim; ++d) {
        x_min[d] = extent[d].first;
        x_max[d] = extent[d].second;
      }

      /*Kokkos::parallel_for("ResetFields",
                           CreateRangePolicy<M::Dim>(x_min, x_max),
                           arch::SetEMFields_kernel<decltype(init_flds), S, M> {
                             domain.fields.em,
                             init_flds,
                             domain.mesh.metric });
      */

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

      // same maxwell distribution as above
      const auto temperatures = std::make_pair(temperature,
                                               temperature_ratio * temperature);
      const auto drifts       = std::make_pair(
        std::vector<real_t> { -drift_ux, ZERO, ZERO },
        std::vector<real_t> { -drift_ux, ZERO, ZERO });
      arch::InjectUniformMaxwellians<S, M>(params,
                                           domain,
                                           ONE,
                                           temperatures,
                                           { 1, 2 },
                                           drifts,
                                           false,
                                           inj_box);

    }
  };
} // namespace user
#endif
