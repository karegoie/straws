use std::fs::File;
use std::io::{Write, Read};

use strauws::cwt;
use strauws::seq;

use structopt::StructOpt;
use ndarray::prelude::*;
use byteorder::{LittleEndian, WriteBytesExt};

#[derive(Debug, StructOpt)]
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
}

fn cwt_and_process(sequence: &mut Vec<u8>, params: &cwt::Params, processed_seqnames: &mut Vec<String>, seqname: String, opt: &Opt, infer_model: String)
                   -> Result<(), std::io::Error>
{
    let cwt_iterator = cwt::CwtIterator::new(sequence, &params);


    let mut file = File::create(format!("{}.cwt", seqname.clone())).unwrap();
    let mut length = 0;
    for batch in cwt_iterator.iter() {
        for row in batch.axis_iter(Axis(1)) {
            for val in row.iter() {
                file.write_f64::<LittleEndian>(*val).unwrap();
            }
            length += 1;
        }
    }

    let mut number_of_cwt = opt.number;
    if infer_model == "None" {
        let mut conf = File::create(format!("{}.conf", seqname.clone())).unwrap();
        conf.write_all(format!("{},{}", length, opt.number).as_bytes()).unwrap();
    } else {
        let mut conf = File::open(format!("{}.conf", infer_model)).unwrap();
        let mut buffer = String::new();
        conf.read_to_string(&mut buffer).unwrap();
        let iter: Vec<_> = buffer.split(",").collect();
        number_of_cwt = iter[1].parse::<usize>().unwrap();
    }


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
        only: opt.only,
    };

    // let (mut initial_seq, seqname) : (Vec<u8>, String) = seq::read_fasta_to_vec(opt.input.as_str()).unwrap();
    match seq::split_fasta(opt.input.as_str()) {
        Ok((tempfile_list, seqnames)) => {
            let mut processed_seqnames = Vec::new();
            for (i, file) in tempfile_list.iter().enumerate() {
                let seqname = seqnames[i].clone();
                println!("{} processing...", seqname);
                match seq::read_fasta_to_vec(file) {
                    Ok(mut initial_seq) => {
                        cwt_and_process(&mut initial_seq, &params, &mut processed_seqnames, seqname, &opt, opt.model.clone()).expect("Processing Error");
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