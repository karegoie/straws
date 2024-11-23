use num_complex::Complex;
use rayon::prelude::*;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub const PAD: usize = 1000;

struct Converter;
trait Convert {
    fn convert(&self, value: u8) -> Complex<f64>; // Vec<bool>
}

impl Convert for Converter {
    fn convert(&self, value: u8) -> Complex<f64> {
        match value {
            b'A' | b'a' => Complex::new(1.0, 1.0),  // 1+i
            b'C' | b'c' => Complex::new(1.0, -1.0), // 1-i
            b'G' | b'g' => Complex::new(-1.0, 1.0), // -1+i
            b'T' | b't' => Complex::new(-1.0, -1.0),// -1-i
            _ => Complex::new(0.0, 0.0),            // 0+0i
        }
    }
}

pub fn convert_to_signal(sequence: &Vec<u8>) -> Vec<Complex<f64>> {
    static CONVERTER: Converter = Converter;
    let chunk_size = 1024;
    use std::sync::Mutex;

    let signal = Mutex::new(vec![Complex::new(0.0, 0.0); sequence.len()+PAD]);
    
    sequence.chunks(chunk_size)
        .enumerate()
        .par_bridge()
        .for_each(|(i, chunk)| {
            let start = i * chunk_size;
            chunk.iter()
                .enumerate()
                .for_each(|(j, &x)| {
                    let mut signal = signal.lock().unwrap();
                    signal[start + j] = CONVERTER.convert(x);
                });
        });

    let signal = Mutex::into_inner(signal).unwrap();

    signal
}

pub fn read_sequence<P: AsRef<Path>>(file_path: P, target_seqid: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut sequence = Vec::new();
    let mut is_target_sequence = false;

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('>') {
            if is_target_sequence {
                break;
            }
            let seqid = line[1..].split_whitespace().next().unwrap_or("");
            is_target_sequence = seqid == target_seqid;
        } else if is_target_sequence {
            sequence.extend(line.trim().bytes().filter(|&b| b != b' ' && b != b'\t'));
        }
    }

    if sequence.is_empty() {
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Sequence with ID '{}' not found", target_seqid),
        )))
    } else {
        Ok(sequence)
    }
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
