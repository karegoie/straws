use std::f64::consts::PI;

use rayon::prelude::*;
use ndarray::{Array, Array1, Array2, Axis, s};
use rustfft::{FftPlanner, num_complex::Complex};
use sys_info::mem_info;
use log::debug;
use std::iter::Iterator;

// Define constants
const OMEGA_0: f64 = 6.0;

// Params struct with periods
#[derive(Clone, Debug)]
pub struct Params {
    pub num: usize,
    pub periods: Vec<f64>, // List of periods
}

// Function to convert period to scale
fn period_to_scale(t: f64) -> f64 {
    (OMEGA_0 * t) / (2.0 * PI)
}

fn pairwise_multiply(a: &[f64], b: &[Complex<f64>]) -> Vec<Complex<f64>> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

// Morlet wavelet function
fn psi(s: f64) -> Array1<Complex<f64>> {
    let t_len = (2.0 * OMEGA_0 * s) as usize;
    let t = Array::linspace(-OMEGA_0 * s, OMEGA_0 * s, t_len);

    let coeff = (PI.powf(-0.25)) / s.sqrt();
    let exp_term = t.mapv(|x| (-0.5 * (x / s).powi(2)).exp());
    let cos_term = t.mapv(|x| Complex::new(0.0, OMEGA_0 * x / s).exp());

    let p = Array1::from_vec(pairwise_multiply(&exp_term.to_vec(), &cos_term.to_vec()));
    let psi = Complex::new(coeff, 0.0) * p;
    psi
}

// Wavelet convolution using FFT
fn wavelet_convolution(f: &Vec<Complex<f64>>, scale: f64, fft_planner: &mut FftPlanner<f64>) -> Array1<f64> {
    let h = psi(scale);
    let f_len = f.len();
    let h_len = h.len();
    let fft_len = f_len + h_len;

    // Zero-padding
    let mut f_padded = vec![Complex::new(0.0, 0.0); fft_len];
    for (i, &val) in f.iter().enumerate() {
        f_padded[i] = val;
    }

    let mut h_padded = vec![Complex::new(0.0, 0.0); fft_len];
    for (i, &val) in h.iter().enumerate() {
        h_padded[i] = val;
    }

    // Perform FFT
    let fft = fft_planner.plan_fft_forward(fft_len);
    fft.process(&mut f_padded);
    fft.process(&mut h_padded);

    // Point-wise multiplication
    let mut result_complex: Vec<Complex<f64>> = f_padded.iter()
        .zip(h_padded.iter())
        .map(|(&a, &b)| a * b)
        .collect();

    // Perform IFFT
    let ifft = fft_planner.plan_fft_inverse(fft_len);
    ifft.process(&mut result_complex);

    // Normalize and extract the relevant part
    let result_norm: Vec<f64> = result_complex.iter().map(|&val| val.norm()).collect();
    let result_view = Array1::from_shape_vec(fft_len, result_norm).unwrap();
    let start = h_len / 2;
    let end = start + f_len;
    result_view.slice(s![start..end]).to_owned()
}

fn cwt_perform(f: &Vec<Complex<f64>>, opt: &Params) -> Array2<f64> {
    let f_len = f.len();
    let periods = &opt.periods;
    let periods_len = periods.len();

    // Parallel computation of each period's wavelet convolution
    let rows: Vec<Array1<f64>> = periods.par_iter().map(|&t| {
        let scale = period_to_scale(t);
        let mut fft_planner = FftPlanner::new();
        wavelet_convolution(f, scale, &mut fft_planner)
    }).collect();

    // Assemble the results into a 2D array
    let mut result_2d = Array2::zeros((f_len, periods_len));
    for (i, row) in rows.into_iter().enumerate() {
        result_2d.column_mut(i).assign(&row);
    }

    result_2d
}

// Normalize function (optional)
pub fn _normalize(matrix: &mut Array2<f64>) {
    matrix.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let min = row.fold(f64::MAX, |a, &b| a.min(b));
            let max = row.fold(f64::MIN, |a, &b| a.max(b));
            let range = max - min;

            if range != 0.0 {
                row.mapv_inplace(|x| (x - min) / range);
            } else {
                row.fill(0.5);
            }
        });
}

// Standardize function (optional)
pub fn _standardize(matrix: &mut Array2<f64>) {
    matrix.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let mean = row.mean().unwrap_or(0.0);
            let std_dev = row.std(0.0);

            if std_dev != 0.0 {
                row.mapv_inplace(|x| (x - mean) / std_dev);
            } else {
                row.mapv_inplace(|x| x - mean);
            }
        });
}

// CwtIterator struct with FFTPlanner
#[derive(Clone)]
pub struct CwtIterator {
    signal: Vec<Complex<f64>>,
    opt: Params,
    current_batch: usize,
    batch_size: usize,
}

impl CwtIterator {
    pub fn new(seq: &mut Vec<u8>, opt: &Params) -> Self {
        let signal = super::seq::convert_to_signal(seq);
        let opt_clone = opt.clone();

        let mem = mem_info().expect("Failed to get memory info");

        let available_ram = mem.free as f64 * 1024.0; // KB to Bytes
        let ram_for_batches = available_ram * 0.5; // 50% of available RAM

        let size_of_complex = std::mem::size_of::<Complex<f64>>() as f64; // size of Complex<f64> in bytes

        let inner_vec_size = opt_clone.periods.len(); // Number of periods
        let size_of_inner_vec = inner_vec_size as f64 * size_of_complex; // Byte

        let max_inner_vecs_in_batch = (ram_for_batches / size_of_inner_vec).floor() as usize;

        let fbatch_size = if max_inner_vecs_in_batch < signal.len() {
            max_inner_vecs_in_batch
        } else {
            signal.len()
        };

        CwtIterator {
            signal,
            opt: opt_clone,
            current_batch: 0,
            batch_size: fbatch_size,
        }
    }

    pub fn iter(&self) -> CwtIterator {
        self.clone()
    }
}

impl Iterator for CwtIterator {
    type Item = Array2<f64>; // Return CWT result and period list

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_batch >= (self.signal.len() + self.batch_size - 1) / self.batch_size {
            return None;
        }
        debug!("Current batch: {}", self.current_batch + 1);

        let start = self.current_batch * self.batch_size;
        let end = std::cmp::min(start + self.batch_size, self.signal.len());
        let f = self.signal[start..end].to_vec();

        let batch_cwt = cwt_perform(&f, &self.opt);

        // _normalize(&mut batch_cwt); // Optional
        self.current_batch += 1;
        debug!("{:?}", batch_cwt);
        Some(batch_cwt) // Return period list
    }
}
