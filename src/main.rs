use std::fs::File;
use std::io::Write;

use strauws::cwt;
use strauws::seq;

use structopt::StructOpt;
use ndarray::prelude::*;
use byteorder::{LittleEndian, WriteBytesExt};
use rayon::prelude::*;
use tempfile::NamedTempFile;
use std::sync::{Arc, Mutex};
use sys_info::mem_info;

const GB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, StructOpt)]
#[structopt(name = "straws", about = "Size-free Tandem Repeat analysis using Continuous Wavelet Transform and Signaling")]
struct Opt {
    /// Input sequence file
    #[structopt(short, long, default_value = "input.fasta")]
    input: String,

    /// Start position (integer)
    #[structopt(short, long, default_value = "100")]
    start: usize,

    /// End position (integer)  
    #[structopt(short, long, default_value = "300")]
    end: usize,

    /// Number (integer)
    #[structopt(short, long, default_value = "30")]  
    number: usize,

    #[structopt(short, long, default_value = "7.0")]
    tradeoff: f64,

    #[structopt(short, long)]  
    filter: bool,

    #[structopt(short = "d", long = "threshold", required_if("filter", "true"))]
    threshold: Option<f64>,
}

fn cwt_and_process(sequence: &mut Vec<u8>, params: &cwt::Params, processed_seqnames: &Arc<Mutex<Vec<String>>>, seqname: String, opt: &Opt)
                   -> Result<Option<Vec<f64>>, std::io::Error>
{                   
    let cwt_iterator = cwt::CwtIterator::new(sequence, &params);

    let mut file = File::create(format!("{}", seqname)).unwrap();  
    let mut length = 0;
    let mut shannon_diversity = Vec::new();
    for batch in cwt_iterator.iter() {
        for row in batch.axis_iter(Axis(1)) {
            for val in row.iter() {
                file.write_f64::<LittleEndian>(*val).unwrap();      
            }
            shannon_diversity.push(seq::calculate_shannon_diversity_for_vector(&row.to_vec()));
            length += 1;
        }
    }

    // write shannon diversity to .ent file  
    let ent_file = format!("{}.ent", seqname);
    let mut ent = File::create(ent_file).unwrap();
    for val in shannon_diversity.iter() {
        ent.write_f64::<LittleEndian>(*val).unwrap();
    }
    
    let conf_file = format!("{}.conf", seqname);  
    let mut conf = File::create(conf_file).unwrap();    
    conf.write_all(format!("{},{}", length, opt.number).as_bytes()).unwrap();

    processed_seqnames.lock().unwrap().push(seqname.clone());

    if opt.filter {
        Ok(Some(shannon_diversity))  
    } else {
        Ok(None)
    }
}

fn main() {
    let opt = Opt::from_args();
    let start = opt.start as f64;
    let end = opt.end as f64;  
    let num = opt.number; 
    let t_values = cwt::linspace(start * 10.0, end * 10.0, num).to_vec(); // wavelet is range (-5, 5, wavelet_length) , so multiply by 10

    let params = cwt::Params {
        num,  
        tradeoff: opt.tradeoff,
        t_values, 
    };

    let mem = mem_info().unwrap(); 
    let available_mem_gb = mem.avail / GB;
    let chunk_size = if available_mem_gb >= 500 {
        1_000_000_000 // 1GB chunk for systems with >= 500GB RAM
    } else if available_mem_gb >= 300 {
        750_000_000 // 750MB chunk for systems with 300-499GB RAM
    } else if available_mem_gb >= 200 {
        500_000_000 // 500MB chunk for systems with 200-299GB RAM  
    } else if available_mem_gb >= 100 {
        250_000_000 // 250MB chunk for systems with 100-199GB RAM
    } else if available_mem_gb > 16 { 
        100_000_000 // 100MB chunk for systems with > 16GB RAM
    } else if available_mem_gb > 8 {
        50_000_000 // 50MB chunk for systems with 8-16GB RAM
    } else {  
        10_000_000 // 10MB chunk for systems with <= 8GB RAM  
    };

    let processed_seqnames = Arc::new(Mutex::new(Vec::new()));

    if opt.filter {
        let filtered = Arc::new(Mutex::new(File::create("filtered.fasta").expect("Error creating filtered file")));

        if opt.input.ends_with(".fasta") || opt.input.ends_with(".fa") {
            match seq::split_fasta(opt.input.as_str()) {
                Ok((tempfile_list, seqnames)) => {
                    tempfile_list.par_iter().zip(seqnames.par_iter())
                        .filter_map(|(file, seqname)| {
                            println!("{} processing...", seqname);
                            let mut initial_seq = seq::read_fasta_to_vec(file).expect("Error reading tempfile");
                            
                            match cwt_and_process(&mut initial_seq, &params, &processed_seqnames, seqname.to_string(), &opt) {
                                Ok(Some(shannon_diversity)) => {
                                    let thres = opt.threshold.unwrap();
                                    if (shannon_diversity.iter().sum::<f64>() / shannon_diversity.len() as f64) < thres {
                                        Some(initial_seq)  
                                    } else {
                                        None
                                    }
                                },
                                Ok(None) => None,
                                Err(e) => { 
                                    eprintln!("Processing Error: {}", e);
                                    None
                                }
                            }
                        })
                        .for_each(|seq| {
                            filtered.lock().unwrap().write_all(&seq).expect("Error writing to filtered file"); 
                        });
                },
                Err(e) => eprintln!("Error splitting FASTA file: {}", e),
            }
        } else if opt.input.ends_with(".fastq") || opt.input.ends_with(".fq") {
            match seq::read_fastq_to_vec(opt.input.as_str(), chunk_size) {
                Ok(chunks) => {
                    chunks.par_iter()
                        .filter_map(|chunk| {
                            let temp_cwt = NamedTempFile::new().expect("Failed to create temporary file");
    
                            let mut chunk_copy = chunk.clone();
                            match cwt_and_process(&mut chunk_copy, &params, &processed_seqnames, temp_cwt.path().to_str().unwrap().to_string(), &opt) {
                                Ok(Some(shannon_diversity)) => {
                                    let thres = opt.threshold.unwrap();
                                    if (shannon_diversity.iter().sum::<f64>() / shannon_diversity.len() as f64)< thres {
                                        Some(chunk.clone())  
                                    } else {
                                        None
                                    }  
                                },
                                Ok(None) => None,
                                Err(e) => {
                                    eprintln!("Processing Error: {}", e);
                                    None
                                }
                            }
                        })  
                        .for_each(|chunk| {
                            filtered.lock().unwrap().write_all(&chunk).expect("Error writing to filtered file");
                        });
                }
                Err(e) => eprintln!("Error reading FASTQ file: {}", e),
            }
        }
    } else {
        if opt.input.ends_with(".fasta") {
            match seq::split_fasta(opt.input.as_str()) {
                Ok((tempfile_list, seqnames)) => {
                    tempfile_list.par_iter().zip(seqnames.par_iter())
                        .for_each(|(file, seqname)| {
                            println!("{} processing...", seqname);
                            let mut initial_seq = seq::read_fasta_to_vec(file).expect("Error reading tempfile");
                            cwt_and_process(&mut initial_seq, &params, &processed_seqnames, seqname.to_string(), &opt).expect("Processing Error");
                        });
                },
                Err(e) => eprintln!("Error splitting FASTA file: {}", e),
            }
        } else if opt.input.ends_with(".fastq") {
            match seq::read_fastq_to_vec(opt.input.as_str(), chunk_size) {
                Ok(chunks) => {
                    chunks.par_iter()
                        .for_each(|chunk| {
                            let temp_cwt = NamedTempFile::new().expect("Failed to create temporary file");
                            let mut chunk_copy = chunk.clone();
                            cwt_and_process(&mut chunk_copy, &params, &processed_seqnames, temp_cwt.path().to_str().unwrap().to_string(), &opt).expect("Processing Error");
                        });
                }
                Err(e) => eprintln!("Error reading FASTQ file: {}", e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]  
    fn test_main() {
        main();
    }
}