use std::fs::File;
use std::io::Write;

use strauws::cwt;
use strauws::seq;

use structopt::StructOpt;
use ndarray::prelude::*;
use byteorder::{LittleEndian, WriteBytesExt};
use rayon::prelude::*;
use tempfile::NamedTempFile;


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
    threshold: f64,
}

fn cwt_and_process(sequence: &mut Vec<u8>, params: &cwt::Params, processed_seqnames: &mut Vec<String>, seqname: String, opt: &Opt)
                   -> Result<Vec<f64>, std::io::Error>
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

    processed_seqnames.push(seqname.clone());
    Ok(shannon_diversity)
}

fn main() {
    let opt = Opt::from_args();
    let start = opt.start as f64;
    let end = opt.end as f64;
    let num = opt.number; // FIX THIS
    let t_values = cwt::linspace(start * 10.0, end * 10.0, num).to_vec(); // wavelet is range (-5, 5, wavelet_length) , so multiply by 10

    let params = cwt::Params {
        num,
        tradeoff: opt.tradeoff,
        t_values, // (1/wavelet length)
    };


    if opt.filter {
        let mut filtered = File::create("filtered.fasta").expect("Error creating filtered file");
    
        match seq::read_fastq_to_vec(opt.input.as_str(), 10000) {
            Ok(chunks) => {
                let filtered_chunks: Vec<_> = chunks.par_iter()
                    .filter_map(|chunk| {
                        let temp_cwt = NamedTempFile::new().expect("Failed to create temporary file");
    
                        let mut chunk_copy = chunk.clone();
                        match cwt_and_process(&mut chunk_copy, &params, &mut Vec::new(), temp_cwt.path().to_str().unwrap().to_string(), &opt) {
                            Ok(shannon_diversity) => {
                                if seq::mean(&shannon_diversity) < opt.threshold {
                                    Some(chunk.clone())
                                } else {
                                    None
                                }
                            }
                            Err(e) => {
                                eprintln!("Processing Error: {}", e);
                                None
                            }
                        }
                    })
                    .collect();
    
                for chunk in filtered_chunks {
                    filtered.write_all(&chunk).expect("Error writing to filtered file");
                }
            }
            Err(e) => eprintln!("Error reading input file: {}", e),
        }
        return;
    }

    // let (mut initial_seq, seqname) : (Vec<u8>, String) = seq::read_fasta_to_vec(opt.input.as_str()).unwrap();
    match seq::split_fasta(opt.input.as_str()) {
        Ok((tempfile_list, seqnames)) => {
            let mut processed_seqnames = Vec::new();
            for (i, file) in tempfile_list.iter().enumerate() {
                let seqname = seqnames[i].clone();
                println!("{} processing...", seqname);
                match seq::read_fasta_to_vec(file) {
                    Ok(mut initial_seq) => {
                        cwt_and_process(&mut initial_seq, &params, &mut processed_seqnames, seqname, &opt).expect("Processing Error");
                    }
                    Err(e) => eprintln!("Error reading tempfile {}: {}", i, e),
                }
            }
        }
        Err(e) => eprintln!("Error splitting FASTA file: {}", e),
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