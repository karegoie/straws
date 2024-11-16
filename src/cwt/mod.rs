use std::f64::consts::PI;

use rayon::prelude::*;
use ndarray::{Array, Array1, Array2, Axis};
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

// Add new struct for reusable buffers
#[derive(Clone)]
struct WaveletBuffers {
    f_padded: Vec<Complex<f64>>,
    h_padded: Vec<Complex<f64>>,
    result_complex: Vec<Complex<f64>>,
}

impl WaveletBuffers {
    fn new(fft_len: usize) -> Self {
        WaveletBuffers {
            f_padded: vec![Complex::new(0.0, 0.0); fft_len],
            h_padded: vec![Complex::new(0.0, 0.0); fft_len],
            result_complex: vec![Complex::new(0.0, 0.0); fft_len],
        }
    }

    fn reset(&mut self) {
        self.f_padded.fill(Complex::new(0.0, 0.0));
        self.h_padded.fill(Complex::new(0.0, 0.0));
        self.result_complex.fill(Complex::new(0.0, 0.0));
    }
}

// Optimized wavelet convolution function
fn wavelet_convolution(
    f: &[Complex<f64>], 
    scale: f64, 
    fft_planner: &mut FftPlanner<f64>,
    buffers: &mut WaveletBuffers
) -> Array1<f64> {
    let h = psi(scale);
    let f_len = f.len();
    let h_len = h.len();
    
    buffers.reset();
    
    // Copy input data to padded buffers
    buffers.f_padded[..f_len].copy_from_slice(f);
    buffers.h_padded[..h_len].copy_from_slice(h.as_slice().unwrap());

    // Reuse FFT plans
    let fft = fft_planner.plan_fft_forward(buffers.f_padded.len());
    fft.process(&mut buffers.f_padded);
    fft.process(&mut buffers.h_padded);

    // In-place multiplication
    for i in 0..buffers.f_padded.len() {
        buffers.result_complex[i] = buffers.f_padded[i] * buffers.h_padded[i];
    }

    // In-place IFFT
    let ifft = fft_planner.plan_fft_inverse(buffers.result_complex.len());
    ifft.process(&mut buffers.result_complex);

    // Extract relevant part and convert to real values
    let start = h_len / 2;
    let end = start + f_len;
    Array1::from_iter(buffers.result_complex[start..end].iter().map(|&x| x.norm()))
}

// Modify cwt_perform to use the new buffers
fn cwt_perform(f: &[Complex<f64>], opt: &Params) -> Array2<f64> {
    let f_len = f.len();
    let periods = &opt.periods;
    let periods_len = periods.len();
    
    let fft_len = f_len + psi(period_to_scale(periods[0])).len();
    let buffers = WaveletBuffers::new(fft_len);

    // Parallel computation of each period's wavelet convolution
    let rows: Vec<Array1<f64>> = periods.par_iter().map(|&t| {
        let scale = period_to_scale(t);
        let mut fft_planner = FftPlanner::new();
        let mut local_buffers = buffers.clone();
        wavelet_convolution(f, scale, &mut fft_planner, &mut local_buffers)
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
