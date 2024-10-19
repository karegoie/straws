use std::f64::consts::PI;

use rayon::prelude::*;
use ndarray::{Array, Array1, s, Array2, Axis};
use rustfft::{FftPlanner, num_complex::Complex};
use sys_info::mem_info;
use std::sync::{Arc, Mutex};
use std::iter::Iterator;

#[derive(Clone, Debug)]
pub struct Params {
    pub num: usize,
    pub tradeoff: f64,
    pub t_values: Vec<f64>,
}

pub fn linspace(start: f64, stop: f64, num: usize) -> Array1<f64> {
    let delta = (stop - start) / ((num - 1) as f64);
    Array::from_shape_fn(num, |i| stop - (i as f64) * delta) // TODO: optimize it to capacity vector
}

fn psi(wavelet_length: &f64, s: f64) -> Array1<Complex<f64>> {
    let kappa_sigma = (-0.5 * s * s).exp();
    let c_sigma = (1.0 + (-s*s).exp() - 2.0*((-0.75*s*s).exp())).powf(-0.5);
    let i_s = Complex::new(0.0, s as f64);

    let t = linspace(-5.0,5.0, (*wavelet_length) as usize);
    // HERE: -5.0: 5.0 implies bandwith 10 for mother wavelet; multiply 10 with input

    // Psi = c_sigma * (pi^(-1/4))* e^(-1/2 * t^2) * (e^(i*s*t) - kappa_sigma)
    let coeff = Complex::new(c_sigma * (PI.powf(-0.25)), 0.0);
    let part1 = t.mapv(|x| ((-0.5 * x * x) + i_s*x).exp()).mapv(|x| coeff * x);
    let part2 = c_sigma * (PI.powf(-0.25)) * kappa_sigma * t.mapv(|x| (-0.5 * x * x).exp());
    part1 - part2
}

fn wavelet_convolution(tup: (&Array1<Complex<f64>>, f64), s: f64) -> Array1<f64> {
    let f = tup.0; // f is the signal
    let wavelet_length = tup.1; // s is the scale
    let f_len = f.len();

    let mut f_hat = Array1::zeros(f_len + wavelet_length as usize);
    f_hat.slice_mut(s![..f_len]).assign(f);
    let h = psi(&wavelet_length, s as f64);
    //let h = psi(&wavelet_length, s as f64) * kaiser_window(&wavelet_length, 14.0);
    let mut h_hat = Array1::zeros(f_len + wavelet_length as usize);
    h_hat.slice_mut(s![..h.len()]).assign(&h);

    let mut planner = FftPlanner::new();
    let fft_len = f_len + wavelet_length as usize;
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    let mut f_hat_complex: Vec<Complex<f64>> = f_hat.to_vec();
    let mut h_hat_complex: Vec<Complex<f64>> = h_hat.to_vec();

    fft.process(&mut f_hat_complex);
    fft.process(&mut h_hat_complex);

    let mut result_complex: Vec<Complex<f64>> = f_hat_complex.iter().zip(h_hat_complex.iter()).map(|(&a, &b)| a * b).collect();

    ifft.process(&mut result_complex);

    let result_real: Vec<f64> = result_complex.iter().map(|&val| (val.re*val.re + val.im*val.im).sqrt()).collect();
    let result_view = Array1::from_shape_vec(fft_len, result_real).unwrap();
    let start = wavelet_length as usize / 2;
    let end = start + f_len;
    result_view.slice(s![start..end]).to_owned()

}

fn cwt_perform(f: &Array1<Complex<f64>>, opt: &Params) -> Array2<f64> {
    let f_len = f.len();
    let t_values = opt.t_values.clone();

    // Initialize result_2d array
    let result_2d = Arc::new(Mutex::new(Array2::zeros((t_values.len(), f_len))));

    // Perform wavelet convolution and assign results directly to result_2d
    t_values.par_iter().enumerate().for_each(|(i, &t)| {
        let row = wavelet_convolution((&f, t), opt.tradeoff);
        result_2d.lock().unwrap().slice_mut(s![i, ..]).assign(&row);
    });

    let result_cwt_perform = result_2d.lock().unwrap();
    result_cwt_perform.to_owned()
}

pub fn _normalize(matrix: &mut Array2<f64>) {
  for mut row in matrix.axis_iter_mut(Axis(0)) {
        let min = row.fold(f64::MAX, |a, &b| a.min(b));
        let max = row.fold(f64::MIN, |a, &b| a.max(b));
        let range = max - min;
        
        if range != 0.0 {
            row.mapv_inplace(|x| (x - min) / range);
        } else {
            row.fill(0.5);
        }
    }
}

pub fn _standardize(matrix: &mut Array2<f64>) {
    for mut col in matrix.axis_iter_mut(Axis(0)) {
        let mean = col.mean().unwrap_or(0.0);
        let std_dev = col.std(0.0);

        if std_dev != 0.0 {
            col.mapv_inplace(|x| (x - mean) / std_dev);
        } else {
            col.mapv_inplace(|x| x - mean);
        }
    }
}

#[derive(Clone)]
pub struct CwtIterator {
    sig_seqs: Array2<Complex<f64>>,
    opt: Params,
    current_batch: usize,
    batch_size: usize,
}

impl CwtIterator {
    pub fn new(seq: &mut Vec<u8>, opt: &Params) -> Self {
        let mut sig_seqs = super::seq::convert_to_signal(seq);
        //println!("sig_seqs: {:?}", sig_seqs.len());
        //let (overlap, n) = find_size(sig_seqs.len(), opt.batch_size);
        let opt_clone = opt.clone();
        // Convert Vec<Vec<u8>> to Array2<bool>
        let sig_seqs: Array2<Complex<f64>> = Array::from_shape_vec((sig_seqs.len(), sig_seqs[0].len()), sig_seqs.par_iter_mut().flatten().map(|x| *x).collect()).unwrap();

        let mem = mem_info().expect("Failed to get memory info");


        let available_ram = mem.free; // KB
        let ram_for_batches = available_ram as f64 * 0.3; // 30% of available RAM


        let size_of_complex = std::mem::size_of::<Complex<f64>>() as f64; // size of Complex<f64> in bytes

        let inner_vec_size = opt.num;
        let size_of_inner_vec = inner_vec_size as f64 * size_of_complex; // 바이트 단위

        let max_inner_vecs_in_batch = (ram_for_batches * 1024.0) / size_of_inner_vec;

        let fbatch_size: usize;
        if max_inner_vecs_in_batch < sig_seqs.dim().0 as f64 {
            fbatch_size = max_inner_vecs_in_batch.floor() as usize;
        }
        else {
            fbatch_size = sig_seqs.dim().0;
        }

        CwtIterator {
            sig_seqs,
            opt: opt_clone,
            current_batch: 0,
            batch_size: fbatch_size,
        }
    }

    pub fn iter(&self) -> CwtIterator {
        //just like ordinary iter
        self.clone()
    }
}

impl Iterator for CwtIterator {
    type Item = Array2<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        
        if self.current_batch >= (self.sig_seqs.dim().0 + self.batch_size - 1) / self.batch_size {
            return None;
        }
        
        let start = self.current_batch * self.batch_size;
        let end = std::cmp::min(start + self.batch_size, self.sig_seqs.dim().0);
        let batch = self.sig_seqs.slice(s![start..end, ..]).to_owned();

        let mut batch_cwt = Array2::<f64>::zeros((self.opt.num, batch.dim().0));
        batch.axis_iter(Axis(1)).for_each(|f| {
            let one_base_result = cwt_perform(&f.to_owned(), &self.opt);
            batch_cwt = &batch_cwt + &one_base_result;
        });

        self.current_batch += 1;
        //_normalize(&mut batch_cwt);
        _standardize(&mut batch_cwt);
        Some(batch_cwt)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // Test _normalize function
    #[test]
    fn test_normalize_simple() {
        let mut matrix = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        _normalize(&mut matrix);
        
        let expected = arr2(&[[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]);
        assert_eq!(matrix, expected);
    }
}
