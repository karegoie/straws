use num_complex::Complex;
use rayon::prelude::*;

struct Converter;
trait Convert {
    fn convert(&self, value: u8) -> Vec<Complex<f64>>; // Vec<bool>
}

impl Convert for Converter {
    fn convert(&self, value: u8) -> Vec<Complex<f64>> {
        match value {
            b'A' | b'a' => vec![Complex::new(1.0, 1.0)],  // 1+i
            b'C' | b'c' => vec![Complex::new(1.0, -1.0)], // 1-i
            b'G' | b'g' => vec![Complex::new(-1.0, 1.0)], // -1+i
            b'T' | b't' => vec![Complex::new(-1.0, -1.0)],// -1-i
            _ => vec![Complex::new(0.0, 0.0)],            // 0+0i
        }
    }
}

pub fn convert_to_signal(sequence: &mut Vec<u8>) -> Vec<Vec<Complex<f64>>> { // Vec<Vec<bool>>
    let converter = Converter;
    let mut converted_sequence = Vec::with_capacity(sequence.len());
    converted_sequence.par_extend(sequence.par_iter_mut().map(|x| converter.convert(*x)));
    converted_sequence
}

pub fn transpose(matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return vec![]; 
    }

    let nrows = matrix.len();
    let ncols = matrix[0].len();


    let mut transposed = vec![vec![0.0; nrows]; ncols];

    for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            transposed[j][i] = value;
        }
    }

    transposed
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64], mean: f64) -> f64 {
    let variance = data.iter()
        .map(|&value| {
            let diff = mean - value;
            diff * diff
        })
        .sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

pub fn calculate_shannon_diversity_for_vector(sequence: &[f64]) -> f64 {
    let mean = mean(sequence);
    let std_dev = std_dev(sequence, mean);
    
    let entropy: f64 = sequence.iter().map(|&x| {
        let p = (x - mean) / std_dev;
        let p_density = (-0.5 * p * p).exp() / (std_dev * (2.0 * std::f64::consts::PI).sqrt());
        -p_density * p_density.ln()
    }).sum::<f64>().abs() / (sequence.len() as f64);
    
    entropy
}