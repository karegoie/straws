use std::fs::File;
use std::io::Write;

use strauws::cwt;
use strauws::seq;

use structopt::StructOpt;
use ndarray::prelude::*;
use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};

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
                   -> Result<(), std::io::Error>
{
    let cwt_iterator = cwt::CwtIterator::new(sequence, &params);


    let mut file = File::create(format!("{}.cwt", seqname.clone())).unwrap();
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
    let mut ent = File::create(format!("{}.ent", seqname.clone())).unwrap();
    for val in shannon_diversity.iter() {
        ent.write_f64::<LittleEndian>(*val).unwrap();
    }
    
    let mut conf = File::create(format!("{}.conf", seqname.clone())).unwrap();
    conf.write_all(format!("{},{}", length, opt.number).as_bytes()).unwrap();


    processed_seqnames.push(seqname.clone());
    return Ok(())

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
        // filter the single given fastq file (not using split fasta function)
        // read fastq file with `read_fastq_to_vec` function
        // cwt each of the fastq reads if the read is longer than wavelet size
        // wavelet sizes are given by option -s and -e
        // if the mean shannon diversity of cwt is smaller than threshold, redirect it to output file (filtered.fasta)
        // with the id as sequential integars
        // TODO: implement this
        let mut filtered = File::create("filtered.fasta").unwrap();
        match seq::read_fastq_to_vec(opt.input.as_str()) {
            Ok(mut initial_seq) => {
                cwt_and_process(&mut initial_seq, &params, &mut Vec::new(), "filtered".to_string(), &opt).expect("Processing Error");
                // open the .ent file and check the mean shannon diversity
                // if the mean shannon diversity is smaller than threshold, write the sequence to the output file
                let mut ent = File::open("filtered.ent").unwrap();
                let mut shannon_diversity = Vec::new();
                {
                    let mut val = ent.read_f64::<LittleEndian>().unwrap();
                    while val != 0.0 {
                        shannon_diversity.push(val);
                        val = ent.read_f64::<LittleEndian>().unwrap();
                    }
                }
                if seq::mean(&shannon_diversity) < opt.threshold {
                    filtered.write_all(&initial_seq).unwrap();
                }
            }
            Err(e) => eprintln!("Error reading input file: {}", e),
        }
        // remove the temporary files
        std::fs::remove_file("filtered.cwt").unwrap();
        std::fs::remove_file("filtered.ent").unwrap();
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