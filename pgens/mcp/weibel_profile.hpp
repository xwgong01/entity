/**
 * @file kernels/weibel_profile.h
 * @brief Load the velocity of Weibel frame from file, or with analytical form.
 * @implements
 *   - kernel::weibel::WeibelKernel<>
 * @namespaces:
 *   - kernel::weibel::
 * @macros:
 *   - MPI_ENABLED
 * @note
 * At the end of the boundary condition call, if MPI is enabled particles
 * are additionally tagged depending on which direction they are leaving
 */

#ifndef KERNELS_WEIBEL_PROFILE_HPP
#define KERNELS_WEIBEL_PROFILE_HPP

#include <cmath>
#include <Kokkos_Core.hpp>
#include "enums.h"
#include "global.h"
#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"
#include "archetypes/problem_generator.h"


namespace kernel::weibel{
    
    struct WeibelKernel{
            
        private:
            real_t drift_ux;
            real_t shock_filling_fraction; // the relative location of the shock front with respect to the 
            real_t Lsh, Xsh;  // the width of shock transition layer, unit in omg/c (physical unit)
            real_t global_min;
            real_t global_max;
        
        public:
            WeibelKernel() = default;
            WeibelKernel(
                real_t shock_filling_fraction,
                real_t global_min,
                real_t global_max,
                real_t drift_ux,
                real_t Lsh
                ): shock_filling_fraction {shock_filling_fraction}
                 , global_min {global_min}
                 , global_max {global_max}
                 , drift_ux {drift_ux}
                 , Lsh { Lsh }
                 , Xsh {global_min + shock_filling_fraction * (global_max - global_min)}
                 {}
	
	Inline auto tanh(real_t x) const -> real_t{
	    if (x < -6.0)
		return -1.0;
	    if (x > 6.0)
                return 1.0;
            return math::tanh(x);
	}

        Inline auto getux(real_t x) const -> real_t {  // compute ux of weibel frame
            return (5./8. + 3./8. * tanh(2 * (x - Xsh)/Lsh)) * ( - drift_ux);
        }

        Inline auto getdudx(real_t x) const -> real_t { // compute dudx
            real_t dudx = -3. * drift_ux / ( 4. * Lsh *math::pow( math::cosh(2. * (x-Xsh) / Lsh),2));
	    if (dudx * dudx < 1e-10)
		return 0.0;
            return dudx;
        }
    };
};

#endif
