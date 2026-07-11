//! Statistics helpers for benchmark reports (port of `benchmarks/metrics.py`).

/// Wilson score interval for a binomial proportion (z = 1.96).
pub fn wilson_ci(hits: u64, n: u64) -> (f64, f64) {
    let z = 1.96f64;
    if n == 0 {
        return (0.0, 0.0);
    }
    let n = n as f64;
    let p = hits as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let margin = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    ((centre - margin).max(0.0), (centre + margin).min(1.0))
}

/// Mean and sample standard deviation, with std 0 for n < 2.
pub fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, 0.0);
    }
    let variance = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (mean, variance.sqrt())
}

/// Linear-interpolated percentile, or 0.0 for empty input.
pub fn percentile(xs: &[f64], p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut ordered = xs.to_vec();
    ordered.sort_by(|a, b| a.total_cmp(b));
    if ordered.len() == 1 {
        return ordered[0];
    }
    let k = (ordered.len() - 1) as f64 * (p / 100.0);
    let lo = k.floor() as usize;
    let hi = k.ceil() as usize;
    if lo == hi {
        return ordered[k as usize];
    }
    ordered[lo] * (hi as f64 - k) + ordered[hi] * (k - lo as f64)
}

/// The 50th percentile.
pub fn median(xs: &[f64]) -> f64 {
    percentile(xs, 50.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn wilson_ci_matches_reference() {
        // Reference values computed with the Python implementation.
        let (lo, hi) = wilson_ci(8, 10);
        assert!(close(lo, 0.4901, 1e-3), "lo {lo}");
        assert!(close(hi, 0.9433, 1e-3), "hi {hi}");
        assert_eq!(wilson_ci(0, 0), (0.0, 0.0));
        let (lo, hi) = wilson_ci(0, 5);
        assert_eq!(lo, 0.0);
        assert!(hi > 0.0 && hi < 1.0);
        let (lo, hi) = wilson_ci(5, 5);
        assert!(lo > 0.0 && lo < 1.0);
        assert_eq!(hi, 1.0);
    }

    #[test]
    fn percentile_linear_interpolation() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        assert!(close(percentile(&xs, 95.0), 3.85, 1e-12));
        assert_eq!(percentile(&xs, 0.0), 1.0);
        assert_eq!(percentile(&xs, 100.0), 4.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
        assert_eq!(percentile(&[7.0], 95.0), 7.0);
    }

    #[test]
    fn median_odd_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn mean_std_basics() {
        assert_eq!(mean_std(&[]), (0.0, 0.0));
        assert_eq!(mean_std(&[5.0]), (5.0, 0.0));
        let (mean, std) = mean_std(&[1.0, 2.0, 3.0]);
        assert_eq!(mean, 2.0);
        assert_eq!(std, 1.0);
    }
}
