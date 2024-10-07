use std::fs::File;
use std::io::{Write, BufReader, BufRead, BufWriter};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use straws::cwt;
use straws::seq;

use structopt::StructOpt;
use ndarray::prelude::*;
use rayon::prelude::*;
use tempfile::NamedTempFile;
use memmap2::MmapOptions;

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

fn process_fasta<P: AsRef<Path>>(
    path: P,
    params: &cwt::Params,
    processed_seqnames: &Arc<Mutex<Vec<String>>>,
) -> Result<(), std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut current_seq = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('>') {
            if !current_seq.is_empty() {
                process_sequence_fasta(&current_seq, params, processed_seqnames)?;
                current_seq.clear();
            }
        } else {
            current_seq.extend(line.bytes());
        }
    }

    if !current_seq.is_empty() {
        process_sequence_fasta(&current_seq,  params, processed_seqnames)?;
    }

    Ok(())
}

fn process_sequence_fastq(
    seq: &[u8], 
    id: &str,
    params: &cwt::Params,
    processed_seqnames: &Arc<Mutex<Vec<String>>>,
    opt: &Opt,
    filtered: &Arc<Mutex<BufWriter<File>>>,
) -> Result<Option<f64>, std::io::Error> {
    let mut seq_copy = seq.to_vec();
    let temp_cwt = NamedTempFile::new()?;
    let seqname = temp_cwt.path().to_str().unwrap().to_string();
    
    let cwt_iterator = cwt::CwtIterator::new(&mut seq_copy, params);

    let mut shannon_diversity = Vec::new();
    
    for batch in cwt_iterator.iter() {
        for row in batch.axis_iter(Axis(1)) {
            shannon_diversity.push(seq::calculate_shannon_diversity_for_vector(&row.to_vec()));
        }
    }
    
    
    processed_seqnames.lock().unwrap().push(seqname);
    
    let shannon_diversity_avg = shannon_diversity.iter().sum::<f64>() / shannon_diversity.len() as f64;
    let threshold = opt.threshold.unwrap();
    if shannon_diversity_avg < threshold {
        let mut filtered_lock = filtered.lock().unwrap();
        filtered_lock.write_all(b">")?; 
        filtered_lock.write_all(id.as_bytes())?;
        filtered_lock.write_all(b"\n")?;
        filtered_lock.write_all(seq)?;
        filtered_lock.write_all(b"\n")?;
    }
    Ok(Some(shannon_diversity_avg))

}

fn process_sequence_fasta(
    seq: &[u8],
    params: &cwt::Params,
    processed_seqnames: &Arc<Mutex<Vec<String>>>,
) -> Result<(), std::io::Error> {
    let mut seq_copy = seq.to_vec();
    let temp_cwt = NamedTempFile::new()?;
    let seqname = temp_cwt.path().to_str().unwrap().to_string();

    let cwt_iterator = cwt::CwtIterator::new(&mut seq_copy, params);

    let mut shannon_diversity = Vec::new();

    for batch in cwt_iterator.iter() {
        for row in batch.axis_iter(Axis(1)) {
            shannon_diversity.push(seq::calculate_shannon_diversity_for_vector(&row.to_vec()));
        }
    }

    processed_seqnames.lock().unwrap().push(seqname);

    Ok(())
}

fn report_progress(total_reads: &AtomicUsize, start_time: &Instant) {
    let reads = total_reads.load(Ordering::Relaxed);
    let elapsed = start_time.elapsed();
    let reads_in_hundred_thousands = reads / 100_000;
    println!("Processed {} hundred thousand reads in {:?}", reads_in_hundred_thousands, elapsed);
}

fn process_fastq<P: AsRef<Path> + Sync>(
    path: P,
    params: &cwt::Params,
    processed_seqnames: &Arc<Mutex<Vec<String>>>,
    opt: &Opt,
    filtered: &Arc<Mutex<BufWriter<File>>>,
) -> Result<(), std::io::Error> {
    let file = File::open(&path)?;
    let file_size = file.metadata()?.len();

    let mmap = unsafe { MmapOptions::new().map(&file)? };

    let num_cpus = sys_info::cpu_num().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))? as usize;
    println!("Detected {} CPUs", num_cpus);

    let chunk_size = file_size / num_cpus as u64;
    let mut start_positions = vec![0u64];

    // Find chunk boundaries
    for i in 1..num_cpus {
        let mut pos = i as u64 * chunk_size;
        while pos < file_size && mmap[pos as usize] != b'\n' {
            pos += 1;
        }
        pos += 1; // Move to the start of the next line
        while pos < file_size && mmap[pos as usize] != b'@' {
            while pos < file_size && mmap[pos as usize] != b'\n' {
                pos += 1;
            }
            pos += 1;
        }
        if pos < file_size {
            start_positions.push(pos);
        }
    }
    start_positions.push(file_size);

    let total_reads = Arc::new(AtomicUsize::new(0));
    let start_time = Instant::now();

    println!("Processing FASTQ file with {} chunks", start_positions.len() - 1);
    start_positions.par_windows(2).for_each(|window| {
        let start = window[0] as usize;
        let end = window[1] as usize;
        let chunk = &mmap[start..end];
        
        let mut current_pos = 0;
        while current_pos < chunk.len() {
            // Read the ID line
            if chunk[current_pos] != b'@' {
                // Skip invalid records
                while current_pos < chunk.len() && chunk[current_pos] != b'\n' {
                    current_pos += 1;
                }
                current_pos += 1;
                continue;
            }
            let mut id_end = current_pos + 1;
            while id_end < chunk.len() && chunk[id_end] != b'\n' {
                id_end += 1;
            }
            if id_end == chunk.len() {
                break;
            }
            let id = &chunk[current_pos + 1..id_end]; // Skip the '@' character
            current_pos = id_end + 1;

            // Read the sequence line
            let mut seq_end = current_pos;
            while seq_end < chunk.len() && chunk[seq_end] != b'\n' {
                seq_end += 1;
            }
            if seq_end == chunk.len() {
                break;
            }
            let seq = &chunk[current_pos..seq_end];
            current_pos = seq_end + 1;

            // Skip the '+' line
            if current_pos < chunk.len() && chunk[current_pos] == b'+' {
                while current_pos < chunk.len() && chunk[current_pos] != b'\n' {
                    current_pos += 1;
                }
                current_pos += 1;
            } else {
                // Invalid record, skip to next '@'
                while current_pos < chunk.len() && chunk[current_pos] != b'@' {
                    current_pos += 1;
                }
                continue;
            }

            // Skip the quality score line
            let mut qual_end = current_pos;
            while qual_end < chunk.len() && chunk[qual_end] != b'\n' {
                qual_end += 1;
            }
            if qual_end == chunk.len() {
                break;
            }
            current_pos = qual_end + 1;
            
            if let Err(e) = process_sequence_fastq_with_id(id, seq, params, processed_seqnames, opt, filtered) {
                eprintln!("Error processing sequence: {}", e);
            }
            
            total_reads.fetch_add(1, Ordering::Relaxed);
            if total_reads.load(Ordering::Relaxed) % 100_000 == 0 {
                report_progress(&total_reads, &start_time);
            }
        }
    });

    // Final report
    report_progress(&total_reads, &start_time);

    Ok(())
}

fn process_sequence_fastq_with_id(
    id: &[u8], 
    seq: &[u8], 
    params: &cwt::Params, 
    processed_seqnames: &Arc<Mutex<Vec<String>>>, 
    opt: &Opt, 
    filtered: &Arc<Mutex<BufWriter<File>>>
) -> Result<(), std::io::Error> {
    //let seqname = temp_cwt.path().to_str().unwrap().to_string();
    
    match process_sequence_fastq(seq, std::str::from_utf8(id).unwrap(), params, processed_seqnames, opt, filtered) {
        Ok(Some(shannon_diversity_avg)) => {
            let threshold = opt.threshold.unwrap();
            if shannon_diversity_avg < threshold {
                let mut filtered_lock = filtered.lock().unwrap();
                filtered_lock.write_all(b">")?;
                filtered_lock.write_all(id)?;
                filtered_lock.write_all(b"\n")?;
                filtered_lock.write_all(seq)?;
                filtered_lock.write_all(b"\n")?;
            }
        },
        Ok(None) => {},
        Err(e) => {
            eprintln!("Processing Error: {}", e);
            return Err(e);
        },
    }
    
    Ok(())
}

fn main() -> Result<(), std::io::Error> {
    let opt = Opt::from_args();
    let start = opt.start as f64;
    let end = opt.end as f64;  
    let num = opt.number; 
    let t_values = cwt::linspace(start * 10.0, end * 10.0, num).to_vec();

    let params = cwt::Params {
        num,  
        tradeoff: opt.tradeoff,
        t_values, 
    };

    let processed_seqnames = Arc::new(Mutex::new(Vec::new()));

    if (opt.input.ends_with(".fasta") || opt.input.ends_with(".fa")) && !opt.filter {
        process_fasta(&opt.input, &params, &processed_seqnames)?;
    } else if (opt.input.ends_with(".fastq") || opt.input.ends_with(".fq")) && opt.filter {
        let filtered = Arc::new(Mutex::new(BufWriter::new(File::create("filtered.fasta")?)));
        process_fastq(&opt.input, &params, &processed_seqnames, &opt, &filtered)?;
    } else {
        eprintln!("Unsupported file format. Please use .fasta, .fa, .fastq, or .fq files.");
        return Ok(());
    }

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]  
    fn test_main() {
        main().unwrap();
    }
}