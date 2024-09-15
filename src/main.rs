use std::fs::File;
use std::io::{Write, BufReader, BufRead, SeekFrom, Seek};
use std::path::Path;
use strauws::cwt;
use strauws::seq;
use structopt::StructOpt;
use ndarray::prelude::*;
use byteorder::{LittleEndian, WriteBytesExt};
use rayon::prelude::*;
use tempfile::NamedTempFile;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

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
                   -> Result<Option<f64>, std::io::Error>
{                   
    let cwt_iterator = cwt::CwtIterator::new(sequence, &params);

    let mut file = File::create(format!("{}", seqname))?;  
    let mut length = 0;
    let mut shannon_diversity = Vec::new();
    for batch in cwt_iterator.iter() {
        for row in batch.axis_iter(Axis(1)) {
            for val in row.iter() {
                file.write_f64::<LittleEndian>(*val)?;      
            }
            shannon_diversity.push(seq::calculate_shannon_diversity_for_vector(&row.to_vec()));
            length += 1;
        }
    }

    // write shannon diversity to .ent file  
    let ent_file = format!("{}.ent", seqname);
    let mut ent = File::create(ent_file)?;
    for val in shannon_diversity.iter() {
        ent.write_f64::<LittleEndian>(*val)?;
    }
    
    let conf_file = format!("{}.conf", seqname);  
    let mut conf = File::create(conf_file)?;    
    conf.write_all(format!("{},{}", length, opt.number).as_bytes())?;

    processed_seqnames.lock().unwrap().push(seqname.clone());

    if opt.filter {
        Ok(Some(shannon_diversity.iter().sum::<f64>() / shannon_diversity.len() as f64))
    } else {
        Ok(None)
    }
}

fn process_sequence(seq: Vec<u8>, params: &cwt::Params, processed_seqnames: &Arc<Mutex<Vec<String>>>, opt: &Opt, filtered: &Arc<Mutex<File>>, record_counter: &Arc<AtomicUsize>) -> Result<(), std::io::Error> {
    let temp_cwt = NamedTempFile::new()?;
    let mut seq_copy = seq.clone();
    
    match cwt_and_process(&mut seq_copy, params, processed_seqnames, temp_cwt.path().to_str().unwrap().to_string(), opt) {
        Ok(Some(shannon_diversity_avg)) => {
            let threshold = opt.threshold.unwrap();
            if shannon_diversity_avg < threshold {
                let id = record_counter.fetch_add(1, Ordering::SeqCst);
                let mut filtered_lock = filtered.lock().unwrap();
                writeln!(filtered_lock, ">{}", id)?;
                filtered_lock.write_all(&seq)?;
                writeln!(filtered_lock)?;
            }
        },
        Ok(None) => {},
        Err(e) => eprintln!("Processing Error: {}", e),
    }
    
    Ok(())
}

fn process_fasta<P: AsRef<Path>>(path: P, params: &cwt::Params, processed_seqnames: &Arc<Mutex<Vec<String>>>, opt: &Opt, filtered: &Arc<Mutex<File>>, record_counter: &Arc<AtomicUsize>) -> Result<(), std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut current_seq = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('>') {
            if !current_seq.is_empty() {
                process_sequence(current_seq, params, processed_seqnames, opt, filtered, record_counter)?;
                current_seq = Vec::new();
            }
        } else {
            current_seq.extend(line.bytes());
        }
    }

    if !current_seq.is_empty() {
        process_sequence(current_seq, params, processed_seqnames, opt, filtered, record_counter)?;
    }

    Ok(())
}
fn process_fastq_chunk(chunk: Vec<String>, params: &cwt::Params, processed_seqnames: &Arc<Mutex<Vec<String>>>, opt: &Opt, filtered: &Arc<Mutex<File>>, record_counter: &Arc<AtomicUsize>) -> Result<(), std::io::Error> {
    for seq in chunk.into_iter() {
        process_sequence(seq.into_bytes(), params, processed_seqnames, opt, filtered, record_counter)?;
    }
    Ok(())
}
fn report_progress(total_reads: &AtomicUsize, start_time: &Instant) {
    let reads = total_reads.load(Ordering::Relaxed);
    let elapsed = start_time.elapsed();
    let reads_in_hundred_thousands = reads / 100_000;
    println!("Processed {} hundred thousand reads in {:?}", reads_in_hundred_thousands, elapsed);
}

fn process_fastq<P: AsRef<Path> + Sync>(path: P, params: &cwt::Params, processed_seqnames: &Arc<Mutex<Vec<String>>>, opt: &Opt, filtered: &Arc<Mutex<File>>, record_counter: &Arc<AtomicUsize>) -> Result<(), std::io::Error> {
    let file = File::open(&path)?;
    let mut reader = BufReader::new(file);
    
    let num_cpus = sys_info::cpu_num().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))? as usize;
    println!("Detected {} CPUs", num_cpus);
    
    let file_size = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;
    
    let chunk_size = file_size / num_cpus as u64;
    
    let mut start_positions = vec![0u64];
    for i in 1..num_cpus {
        let mut pos = i as u64 * chunk_size;
        reader.seek(SeekFrom::Start(pos))?;
        let mut line = String::new();
        reader.read_line(&mut line)?;
        pos = reader.seek(SeekFrom::Current(0))?;
        while !line.starts_with('@') {
            reader.read_line(&mut line)?;
            pos = reader.seek(SeekFrom::Current(0))?;
        }
        start_positions.push(pos);
    }
    start_positions.push(file_size);
    
    let total_reads = Arc::new(AtomicUsize::new(0));
    let start_time = Instant::now();

    println!("Processing FASTQ file with {} chunks", start_positions.len() - 1);
    start_positions.par_windows(2).map(|window| {
        let start = window[0];
        let end = window[1];
        let mut chunk_reader = BufReader::new(File::open(&path)?);
        chunk_reader.seek(SeekFrom::Start(start))?;
        let mut chunk = Vec::new();
        let mut current_pos = start;
        
        while current_pos < end {
            let mut seq = String::new();
            chunk_reader.read_line(&mut seq)?;  // Read sequence
            if seq.is_empty() {
                break;
            }
            chunk_reader.read_line(&mut String::new())?;  // Skip '+' line
            chunk_reader.read_line(&mut String::new())?;  // Skip quality line
            chunk_reader.read_line(&mut String::new())?;  // Skip name line of next record
            
            chunk.push(seq.trim().to_string());
            current_pos = chunk_reader.seek(SeekFrom::Current(0))?;
        }
        
        let reads_in_chunk = chunk.len();
        total_reads.fetch_add(reads_in_chunk, Ordering::Relaxed);

        if total_reads.load(Ordering::Relaxed) % 100_000 == 0 {
            report_progress(&total_reads, &start_time);
        }

        process_fastq_chunk(chunk, params, processed_seqnames, opt, filtered, record_counter)
    }).collect::<Result<(), std::io::Error>>()?;

    // Final report
    report_progress(&total_reads, &start_time);

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
    let record_counter = Arc::new(AtomicUsize::new(1));
    let filtered = Arc::new(Mutex::new(File::create("filtered.fasta")?));

    if opt.input.ends_with(".fasta") || opt.input.ends_with(".fa") {
        process_fasta(&opt.input, &params, &processed_seqnames, &opt, &filtered, &record_counter)?;
    } else if opt.input.ends_with(".fastq") || opt.input.ends_with(".fq") {
        process_fastq(&opt.input, &params, &processed_seqnames, &opt, &filtered, &record_counter)?;
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