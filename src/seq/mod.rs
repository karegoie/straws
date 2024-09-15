use std::fs::File;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{self, Read, BufRead, BufReader, Write};
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

pub fn read_endian(file_path: String, ncols: usize) -> Result<Vec<Vec<f64>>, io::Error> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = vec![];

    // Read the entire file into the buffer
    reader.read_to_end(&mut buffer)?;

    let nrows = buffer.len() / (ncols * 8);
    let mut matrix = Vec::with_capacity(nrows);

    let mut offset = 0;
    while offset + 8 * ncols <= buffer.len() {
        let mut row = Vec::with_capacity(ncols);
        for _ in 0..ncols {
            if offset + 8 > buffer.len() {
                break;
            }
            // Read f64 directly from buffer in little-endian format
            let value = (&buffer[offset..offset+8]).read_f64::<LittleEndian>()?;
            row.push(value);
            offset += 8;
        }
        matrix.push(row);
    }

    Ok(transpose(matrix))
}


pub fn parse_and_encode_gff(file_path: &String, seqname: String, length: usize) -> Vec<f64> {
    let chromosome_length = length;
    let mut genome_array = vec![false; chromosome_length];

    if let Ok(file) = File::open(file_path) {
        for line in io::BufReader::new(file).lines() {
            if let Ok(record) = line {
                if record.starts_with('#') {
                    continue;
                }
                let columns: Vec<&str> = record.split('\t').collect();
                if columns.len() != 9 || columns[0] != seqname {
                    continue;
                }

                let start: usize = columns[3].parse().unwrap_or(0);
                let end: usize = columns[4].parse().unwrap_or(0);
                let feature = columns[2];

                match feature {
                    "exon" => {
                        for i in start - 1..end {
                            genome_array[i] = true;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    genome_array.iter().map(|&n| if n { 1.0 } else {0.0}).collect()
}

pub fn calculate_histograms(cwt_matrix: Vec<Vec<f64>>, bin_count: usize) -> Vec<usize> {
    let bin_edges: Vec<f64> = (0..bin_count).map(|x| x as f64 / (bin_count as f64 - 1.0)).collect();

    let histograms: Vec<usize> = cwt_matrix.par_iter()
        .flat_map(|row| {
            let mut histogram = vec![0; bin_count - 1];
            
            for &value in row.iter() {
                for j in 0..bin_edges.len() - 1 {
                    if value >= bin_edges[j] && value < bin_edges[j + 1] {
                        histogram[j] += 1;
                        break;
                    }
                }
            }

            histogram
        })
        .collect();

    histograms
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

