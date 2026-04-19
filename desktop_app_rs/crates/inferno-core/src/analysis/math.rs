/// Rolling mean with centered window. Edge values use smaller windows.
pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 || window == 0 {
        return data.to_vec();
    }

    let half = window / 2;
    let mut result = Vec::with_capacity(n);
    let mut sum = 0.0;

    // Initialize sum for the first window
    let first_end = half.min(n);
    for &v in &data[..first_end] {
        sum += v;
    }

    for i in 0..n {
        // Add the new right element entering the window
        let right = i + half;
        if right < n && (i > 0 || right >= first_end) {
            sum += data[right];
        }

        // Remove the old left element leaving the window
        if i > half + 1 {
            sum -= data[i - half - 1];
        }

        let lo = i.saturating_sub(half);
        let hi = (i + half).min(n - 1);
        let count = (hi - lo + 1) as f64;
        result.push(sum / count);
    }

    // Fallback: simple but correct implementation
    // The running-sum approach above has edge-case issues,
    // so let's use the straightforward version for correctness.
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half).min(n - 1);
        let sum: f64 = data[lo..=hi].iter().sum();
        result.push(sum / (hi - lo + 1) as f64);
    }
    result
}

/// Central-difference gradient: dy/dx.
/// At endpoints uses forward/backward differences.
pub fn gradient(y: &[f64], x: &[f64]) -> Vec<f64> {
    let n = y.len();
    assert_eq!(n, x.len());
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }

    let mut result = Vec::with_capacity(n);

    // Forward difference for first point
    let dx = x[1] - x[0];
    result.push(if dx.abs() > 1e-15 {
        (y[1] - y[0]) / dx
    } else {
        0.0
    });

    // Central differences for interior points
    for i in 1..n - 1 {
        let dx = x[i + 1] - x[i - 1];
        result.push(if dx.abs() > 1e-15 {
            (y[i + 1] - y[i - 1]) / dx
        } else {
            0.0
        });
    }

    // Backward difference for last point
    let dx = x[n - 1] - x[n - 2];
    result.push(if dx.abs() > 1e-15 {
        (y[n - 1] - y[n - 2]) / dx
    } else {
        0.0
    });

    result
}

/// Gradient with uniform spacing (index-based, dx=1).
pub fn gradient_uniform(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }

    let mut result = Vec::with_capacity(n);
    result.push(y[1] - y[0]);
    for i in 1..n - 1 {
        result.push((y[i + 1] - y[i - 1]) / 2.0);
    }
    result.push(y[n - 1] - y[n - 2]);
    result
}

/// Cumulative sum.
pub fn cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut acc = 0.0;
    for &v in data {
        acc += v;
        result.push(acc);
    }
    result
}

/// Standard deviation (population, not sample).
pub fn std_dev(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    variance.sqrt()
}

/// Mean of a slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Binary search for insertion point (equivalent to numpy searchsorted).
pub fn searchsorted(data: &[f64], value: f64) -> usize {
    data.partition_point(|&x| x < value)
}

/// Binary search for insertion point, right side.
pub fn searchsorted_right(data: &[f64], value: f64) -> usize {
    data.partition_point(|&x| x <= value)
}

/// Skewness (third standardized moment).
/// Returns 0.0 for fewer than 2 elements or zero variance.
pub fn skewness(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s == 0.0 {
        return 0.0;
    }
    data.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n as f64
}

/// Excess kurtosis (fourth standardized moment minus 3).
/// Returns 0.0 for fewer than 2 elements or zero variance.
pub fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s == 0.0 {
        return 0.0;
    }
    data.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n as f64 - 3.0
}

/// Compute cumulative distance from XY coordinates.
pub fn cumulative_distance(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut dist = Vec::with_capacity(n);
    dist.push(0.0);
    for i in 1..n {
        let dx = x[i] - x[i - 1];
        let dy = y[i] - y[i - 1];
        dist.push(dist[i - 1] + (dx * dx + dy * dy).sqrt());
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_mean_simple() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&data, 3);
        // Window 3, half=1: centered
        assert!((result[0] - 1.5).abs() < 1e-10); // avg(1,2)
        assert!((result[1] - 2.0).abs() < 1e-10); // avg(1,2,3)
        assert!((result[2] - 3.0).abs() < 1e-10); // avg(2,3,4)
        assert!((result[3] - 4.0).abs() < 1e-10); // avg(3,4,5)
        assert!((result[4] - 4.5).abs() < 1e-10); // avg(4,5)
    }

    #[test]
    fn test_gradient_uniform() {
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // x^2
        let dy = gradient_uniform(&y);
        assert!((dy[0] - 1.0).abs() < 1e-10); // forward: 1-0
        assert!((dy[1] - 2.0).abs() < 1e-10); // central: (4-0)/2
        assert!((dy[2] - 4.0).abs() < 1e-10); // central: (9-1)/2
        assert!((dy[3] - 6.0).abs() < 1e-10); // central: (16-4)/2
        assert!((dy[4] - 7.0).abs() < 1e-10); // backward: 16-9
    }

    #[test]
    fn test_cumsum() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = cumsum(&data);
        assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&data);
        assert!((sd - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_searchsorted() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        assert_eq!(searchsorted(&data, 0.0), 0);
        assert_eq!(searchsorted(&data, 1.0), 0);
        assert_eq!(searchsorted(&data, 4.0), 2);
        assert_eq!(searchsorted(&data, 5.0), 2);
        assert_eq!(searchsorted(&data, 10.0), 5);
    }

    #[test]
    fn test_searchsorted_right() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        assert_eq!(searchsorted_right(&data, 5.0), 3);
        assert_eq!(searchsorted_right(&data, 4.0), 2);
    }

    #[test]
    fn test_cumulative_distance() {
        let x = vec![0.0, 3.0, 3.0];
        let y = vec![0.0, 0.0, 4.0];
        let dist = cumulative_distance(&x, &y);
        assert!((dist[0] - 0.0).abs() < 1e-10);
        assert!((dist[1] - 3.0).abs() < 1e-10);
        assert!((dist[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric data → skew ≈ 0
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        assert!(skewness(&data).abs() < 1e-10);
    }

    #[test]
    fn test_skewness_positive() {
        // Right-skewed data
        let data = vec![1.0, 1.0, 1.0, 1.0, 10.0];
        assert!(skewness(&data) > 0.0);
    }

    #[test]
    fn test_skewness_edge_cases() {
        assert_eq!(skewness(&[]), 0.0);
        assert_eq!(skewness(&[5.0]), 0.0);
        assert_eq!(skewness(&[3.0, 3.0, 3.0]), 0.0); // zero variance
    }

    #[test]
    fn test_kurtosis_normal_like() {
        // For a uniform distribution, excess kurtosis is negative (-1.2)
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let k = kurtosis(&data);
        assert!(k < 0.0); // platykurtic
        assert!((k - (-1.2)).abs() < 0.05);
    }

    #[test]
    fn test_kurtosis_edge_cases() {
        assert_eq!(kurtosis(&[]), 0.0);
        assert_eq!(kurtosis(&[5.0]), 0.0);
        assert_eq!(kurtosis(&[3.0, 3.0]), 0.0); // zero variance
    }
}
