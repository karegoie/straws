use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use ndarray::{Array, Array1, s, Array2, Axis};
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
fn wavelet_convolution(tup: (&Vec<Complex<f64>>, f64)) -> Array1<f64> {
    let f = Array1::from_vec(tup.0.clone()); // f is the signal
    let wavelet_size = tup.1;
    let f_len = f.len();
    let h = psi(wavelet_size);

    let fft_len = f_len + h.len();

    debug!("Signal length: {}, Wavelet size: {}", f_len, wavelet_size);
    let mut f_hat = Array1::zeros(fft_len);
    f_hat.slice_mut(s![..f_len]).assign(&f);
    let mut h_hat = Array1::zeros(fft_len);
    h_hat.slice_mut(s![..h.len()]).assign(&h);

    let mut planner = FftPlanner::new();

    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    let mut f_hat_complex: Vec<Complex<f64>> = f_hat.to_vec();
    let mut h_hat_complex: Vec<Complex<f64>> = h_hat.to_vec();

    fft.process(&mut f_hat_complex);
    fft.process(&mut h_hat_complex);

    let mut result_complex: Vec<Complex<f64>> = f_hat_complex.iter().zip(h_hat_complex.iter()).map(|(&a, &b)| a * b).collect();

    ifft.process(&mut result_complex);

    let result_norm: Vec<f64> = result_complex.iter().map(|&val| val.norm()).collect();
    let result_view = Array1::from_shape_vec(fft_len, result_norm).unwrap();
    let start = h.len() / 2;
    let end = start + f_len;
    result_view.slice(s![start..end]).to_owned()
}

// Perform CWT
fn cwt_perform(f: &Vec<Complex<f64>>, opt: &Params) -> Array2<f64> {
    let f_len = f.len();
    let periods = &opt.periods;

    let result_2d = Arc::new(Mutex::new(Array2::zeros((f_len, periods.len()))));

    periods.par_iter().enumerate().for_each(|(i, &t)| {
        let scale = period_to_scale(t);
        let row = wavelet_convolution((&f, scale));
        result_2d.lock().unwrap().slice_mut(s![.., i]).assign(&row);
    });

    let result_cwt_perform = result_2d.lock().unwrap();
    result_cwt_perform.to_owned()
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

// CwtIterator struct
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

        let available_ram = mem.free; // KB
        let ram_for_batches = available_ram as f64 * 0.5; // 50% of available RAM

        let size_of_complex = std::mem::size_of::<Complex<f64>>() as f64; // size of Complex<f64> in bytes

        let inner_vec_size = opt_clone.periods.len(); // Number of periods
        let size_of_inner_vec = inner_vec_size as f64 * size_of_complex; // Byte

        let max_inner_vecs_in_batch = (ram_for_batches * 1024.0) / size_of_inner_vec;

        let fbatch_size: usize;
        if max_inner_vecs_in_batch < signal.len() as f64 {
            fbatch_size = max_inner_vecs_in_batch.floor() as usize;
        }
        else {
            fbatch_size = signal.len();
        }

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

        self.current_batch += 1;
        //_standardize(&mut batch_cwt);
        //_normalize(&mut batch_cwt);
        Some(batch_cwt) // Return period list
    }
}
