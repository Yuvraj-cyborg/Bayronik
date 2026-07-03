//! Background cosmology, linear growth, and the linear matter power spectrum.
//!
//! Conventions:
//! - Flat LCDM (Omega_k = 0, Omega_L = 1 - Omega_m).
//! - h-units throughout: lengths in Mpc/h, masses in Msun/h, wavenumbers in h/Mpc.
//! - Code time unit is 1/H0, i.e. H0 = 1. Velocities are then in units of
//!   100 h km/s (one Mpc/h per 1/H0).
//! - The linear power spectrum uses the Eisenstein & Hu (1998) zero-baryon
//!   ("no-wiggle") transfer function, normalized to sigma8 at z = 0.

/// Critical density today in (Msun/h) / (Mpc/h)^3. In h-units this carries no
/// stray factors of h: rho_crit = 3 H0^2 / (8 pi G) = 2.7754e11 h^2 Msun/Mpc^3.
pub const RHO_CRIT: f64 = 2.775_366_27e11;

/// CMB temperature ratio Theta = T_cmb / 2.7 K (Fixsen 2009).
const THETA_CMB: f64 = 2.7255 / 2.7;

/// Flat LCDM cosmology. Defaults match the CAMELS fiducial values
/// (IllustrisTNG suite: h = 0.6711, n_s = 0.9624, Omega_b = 0.049).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cosmology {
    pub omega_m: f64,
    pub omega_b: f64,
    pub h: f64,
    pub sigma8: f64,
    pub n_s: f64,
}

impl Default for Cosmology {
    fn default() -> Self {
        Self {
            omega_m: 0.3,
            omega_b: 0.049,
            h: 0.6711,
            sigma8: 0.8,
            n_s: 0.9624,
        }
    }
}

impl Cosmology {
    pub fn omega_l(&self) -> f64 {
        1.0 - self.omega_m
    }

    /// Dimensionless Hubble rate E(a) = H(a)/H0.
    pub fn e(&self, a: f64) -> f64 {
        (self.omega_m / (a * a * a) + self.omega_l()).sqrt()
    }

    /// Comoving mean matter density in (Msun/h) / (Mpc/h)^3 (constant in a).
    pub fn mean_matter_density(&self) -> f64 {
        self.omega_m * RHO_CRIT
    }

    /// CDM mass fraction of total matter, (Omega_m - Omega_b) / Omega_m.
    pub fn cdm_fraction(&self) -> f64 {
        (self.omega_m - self.omega_b) / self.omega_m
    }

    /// Unnormalized growth integral I(a) = int_0^a da' / (a' E(a'))^3.
    ///
    /// The integrand behaves like a'^{3/2} near zero, so a plain Simpson rule
    /// starting at zero is well behaved.
    fn growth_integral(&self, a: f64) -> f64 {
        let f = |x: f64| {
            if x <= 0.0 {
                0.0
            } else {
                let ae = x * self.e(x);
                1.0 / (ae * ae * ae)
            }
        };
        simpson(f, 0.0, a, 512)
    }

    /// Linear growth factor D(a), normalized so D(1) = 1.
    ///
    /// D(a) is proportional to E(a) * int_0^a da' / (a' E(a'))^3
    /// (Heath 1977 growing-mode solution for LCDM).
    pub fn growth_d(&self, a: f64) -> f64 {
        let unnorm = |x: f64| self.e(x) * self.growth_integral(x);
        unnorm(a) / unnorm(1.0)
    }

    /// Logarithmic growth rate f(a) = dlnD/dlna.
    ///
    /// From D = E * I: f = dlnE/dlna + 1 / (a^2 E^3 I).
    pub fn growth_f(&self, a: f64) -> f64 {
        let e = self.e(a);
        let dln_e_dln_a = -1.5 * self.omega_m / (a * a * a) / (e * e);
        let i = self.growth_integral(a);
        dln_e_dln_a + 1.0 / (a * a * e * e * e * i)
    }

    /// Leapfrog kick factor int_{a0}^{a1} da / (a E(a)) for the canonical
    /// momentum p = a^2 dx/dt with dp/dt = -grad phi (H0 = 1 units).
    pub fn kick_factor(&self, a0: f64, a1: f64) -> f64 {
        simpson(|a| 1.0 / (a * self.e(a)), a0, a1, 64)
    }

    /// Leapfrog drift factor int_{a0}^{a1} da / (a^3 E(a)) for
    /// dx/da = p / (a^3 E(a)).
    pub fn drift_factor(&self, a0: f64, a1: f64) -> f64 {
        simpson(|a| 1.0 / (a * a * a * self.e(a)), a0, a1, 64)
    }

    /// Poisson equation prefactor at scale factor a:
    /// grad^2 phi = (3/2) Omega_m delta / a  (comoving, H0 = 1).
    pub fn poisson_prefactor(&self, a: f64) -> f64 {
        1.5 * self.omega_m / a
    }

    /// Eisenstein & Hu (1998) zero-baryon transfer function T(k).
    /// k in h/Mpc. Includes the baryon-induced shape suppression via the
    /// effective shape parameter Gamma_eff, but no acoustic oscillations.
    pub fn transfer(&self, k: f64) -> f64 {
        if k <= 0.0 {
            return 1.0;
        }
        let om_h2 = self.omega_m * self.h * self.h;
        let ob_h2 = self.omega_b * self.h * self.h;
        let fb = self.omega_b / self.omega_m;

        // Sound horizon in Mpc (EH98 eq. 26).
        let s = 44.5 * (9.83 / om_h2).ln() / (1.0 + 10.0 * ob_h2.powf(0.75)).sqrt();

        // Effective shape parameter (EH98 eqs. 30-31).
        let alpha_gamma =
            1.0 - 0.328 * (431.0 * om_h2).ln() * fb + 0.38 * (22.3 * om_h2).ln() * fb * fb;
        let k_mpc = k * self.h; // physical Mpc^-1 for the (0.43 k s) term
        let gamma_eff = self.omega_m
            * self.h
            * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k_mpc * s).powi(4)));

        // EH98 eqs. 28-29.
        let q = k * THETA_CMB * THETA_CMB / gamma_eff;
        let l0 = (2.0 * std::f64::consts::E + 1.8 * q).ln();
        let c0 = 14.2 + 731.0 / (1.0 + 62.5 * q);
        l0 / (l0 + c0 * q * q)
    }

    /// Unnormalized P(k) shape: k^{n_s} T^2(k).
    fn power_shape(&self, k: f64) -> f64 {
        let t = self.transfer(k);
        k.powf(self.n_s) * t * t
    }

    /// RMS linear fluctuation in a top-hat sphere of radius r (Mpc/h),
    /// for a spectrum `amp * power_shape(k)`.
    fn sigma_r(&self, r: f64, amp: f64) -> f64 {
        // sigma^2 = int dlnk Delta^2(k) W^2(kR),  Delta^2 = k^3 P / (2 pi^2)
        let two_pi2 = 2.0 * std::f64::consts::PI * std::f64::consts::PI;
        let ln_k_min = (1e-4f64).ln();
        let ln_k_max = (1e2f64).ln();
        let n = 800;
        let dlnk = (ln_k_max - ln_k_min) / n as f64;
        let mut sum = 0.0;
        for i in 0..=n {
            let ln_k = ln_k_min + i as f64 * dlnk;
            let k = ln_k.exp();
            let x = k * r;
            let w = if x < 1e-3 {
                1.0 - x * x / 10.0
            } else {
                3.0 * (x.sin() - x * x.cos()) / (x * x * x)
            };
            let delta2 = k * k * k * amp * self.power_shape(k) / two_pi2;
            let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
            sum += weight * delta2 * w * w * dlnk;
        }
        sum.sqrt()
    }
}

/// Linear matter power spectrum at z = 0, normalized to sigma8.
/// k in h/Mpc, P(k) in (Mpc/h)^3.
pub struct LinearPower {
    cosmo: Cosmology,
    amplitude: f64,
}

impl LinearPower {
    pub fn new(cosmo: &Cosmology) -> Self {
        let sigma_unnorm = cosmo.sigma_r(8.0, 1.0);
        let amplitude = (cosmo.sigma8 / sigma_unnorm).powi(2);
        Self {
            cosmo: *cosmo,
            amplitude,
        }
    }

    pub fn p(&self, k: f64) -> f64 {
        if k <= 0.0 {
            0.0
        } else {
            self.amplitude * self.cosmo.power_shape(k)
        }
    }
}

/// Composite Simpson rule with n subintervals (n rounded up to even).
fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    if b <= a {
        return 0.0;
    }
    let n = if n % 2 == 0 { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f(x);
    }
    sum * h / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e_of_a_today_is_one() {
        let c = Cosmology::default();
        assert!((c.e(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn growth_normalized_today() {
        let c = Cosmology::default();
        assert!((c.growth_d(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn growth_matches_eds_limit() {
        // For Omega_m = 1 the growing mode is exactly D = a.
        let c = Cosmology {
            omega_m: 1.0,
            omega_b: 0.049,
            ..Default::default()
        };
        for &a in &[0.1, 0.25, 0.5, 0.9] {
            let d = c.growth_d(a);
            assert!(
                (d - a).abs() / a < 1e-3,
                "EdS growth should be D=a: D({a}) = {d}"
            );
        }
    }

    #[test]
    fn growth_rate_bounds() {
        let c = Cosmology::default();
        // f = Omega_m(a)^0.55 approximation: f(1) ~ 0.51 for Omega_m = 0.3.
        let f1 = c.growth_f(1.0);
        let approx = (0.3f64).powf(0.55);
        assert!(
            (f1 - approx).abs() < 0.02,
            "f(1) = {f1}, expected ~{approx}"
        );
        // At early times matter dominates: f -> 1.
        let f_early = c.growth_f(0.02);
        assert!((f_early - 1.0).abs() < 0.01, "f(0.02) = {f_early}");
    }

    #[test]
    fn transfer_limits() {
        let c = Cosmology::default();
        assert!((c.transfer(1e-5) - 1.0).abs() < 0.01, "T(k->0) -> 1");
        // Transfer function decreases monotonically.
        let mut prev = c.transfer(1e-4);
        for i in 1..50 {
            let k = 1e-4 * (10f64).powf(i as f64 * 0.14);
            let t = c.transfer(k);
            assert!(t <= prev + 1e-12, "T(k) should be non-increasing");
            assert!(t > 0.0);
            prev = t;
        }
    }

    #[test]
    fn sigma8_normalization_roundtrip() {
        let c = Cosmology::default();
        let pk = LinearPower::new(&c);
        let sigma = c.sigma_r(8.0, pk.amplitude);
        assert!(
            (sigma - c.sigma8).abs() < 1e-6,
            "sigma(8) after normalization = {sigma}, want {}",
            c.sigma8
        );
    }

    #[test]
    fn power_spectrum_sane_shape() {
        let c = Cosmology::default();
        let pk = LinearPower::new(&c);
        // LCDM P(k) peaks near k ~ 0.02 h/Mpc at around 2-3e4 (Mpc/h)^3.
        let p_peak = pk.p(0.02);
        assert!(
            p_peak > 5e3 && p_peak < 1e5,
            "P(0.02) = {p_peak}, expected O(1e4)"
        );
        // Small scales are suppressed.
        assert!(pk.p(10.0) < pk.p(0.1));
    }

    #[test]
    fn kick_drift_factors_positive_and_ordered() {
        let c = Cosmology::default();
        let k = c.kick_factor(0.5, 0.6);
        let d = c.drift_factor(0.5, 0.6);
        assert!(k > 0.0 && d > 0.0);
        // 1/(a^3 E) > 1/(a E) for a < 1, so drift factor is larger.
        assert!(d > k);
    }
}
