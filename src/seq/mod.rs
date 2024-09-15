use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use num_complex::Complex;
use rayon::prelude::*;
use tempfile::NamedTempFile;

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

pub fn read_fasta_to_vec(file: &NamedTempFile) -> Result<Vec<u8>, io::Error>
{
    let file = file.reopen()?;
    let reader = BufReader::new(file);
    
    let mut sequence = Vec::new();

    for line_result in reader.lines() {
        let line = line_result.expect("Reading individual fasta file error");
        sequence.extend_from_slice(line.as_bytes());
    }

    Ok(sequence)
}

pub fn read_fastq_to_vec(file: &str) -> Result<Vec<u8>, io::Error> {
    let file = File::open(file).expect("file opening error");
    let reader = BufReader::new(file);
    let mut sequences = Vec::new();
    let mut lines = reader.lines().enumerate();

    while let Some((i, line_result)) = lines.next() {
        let line = line_result?;
        
        // Every 4th line (0-indexed) is the start of a new record
        if i % 4 == 0 {
            if !line.starts_with('@') {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Malformed FASTQ: ID line doesn't start with @"));
            }
            // Skip to the next line, which contains the sequence
            if let Some((_, seq_result)) = lines.next() {
                let seq = seq_result?;
                sequences.extend(seq.bytes());
            } else {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Malformed FASTQ: unexpected end of file"));
            }
            // Skip the '+' line and quality score line
            lines.next();
            lines.next();
        }
    }

    Ok(sequences)
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

pub fn split_fasta(filename: &str) -> Result<(Vec<NamedTempFile>, Vec<String>),io::Error> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let mut sequences = Vec::new();
    let mut is_sequence_line = false;

    let mut seqnames: Vec<String> = Vec::new();
    let mut tempfile_list: Vec<NamedTempFile> = Vec::new();

    for line_result in reader.lines() {
        let line = line_result?;
        if line.starts_with('>') {
            if is_sequence_line {
                let mut tempfile = NamedTempFile::new()?;
                tempfile.write_all(&sequences)?;
                tempfile_list.push(tempfile);
                sequences.clear();
            }
            let desc: Vec<&str> = line.split_whitespace().collect();
            seqnames.push(desc[0][1..].to_string());
            is_sequence_line = true;
        } else if is_sequence_line {
            sequences.extend_from_slice(line.as_bytes());
            sequences.push(b'\n');
        }
    }

    if !sequences.is_empty() {
        let mut tempfile = NamedTempFile::new()?;
        tempfile.write_all(&sequences)?;
        tempfile_list.push(tempfile);
    }

    Ok((tempfile_list, seqnames))
}

pub fn mean(data: &[f64]) -> f64 {
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