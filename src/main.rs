use std::fs::File;
use std::io::{Write, BufReader, BufRead, BufWriter};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use straws::cwt;
use straws::seq;
use straws::logger;

use structopt::StructOpt;
use ndarray::prelude::*;
use rayon::prelude::*;
use tempfile::NamedTempFile;
use memmap2::MmapOptions;
use byteorder::{WriteBytesExt, LittleEndian};
use log::{info, warn, error, debug, LevelFilter};

/// Command-line options structure.
#[derive(Debug, StructOpt)]
#[structopt(name = "straws", about = "Size-free Tandem Repeat analysis using Continuous Wavelet Transform and Signaling")]
struct Opt {
    /// Input sequence file
    #[structopt(short, long)]
    input: String,

    /// Wavelet sizes (comma-separated list)
    #[structopt(short = "w", long = "wavelet-sizes", required = true)]
    wavelet_sizes: String,

    /// Enable filtering
    #[structopt(short, long)]  
    filter: bool,

    /// No .cwt file for visualize
    #[structopt(short)]
    cwt: bool,

    /// Threshold for filtering (required if --filter is set)
    #[structopt(short, long = "threshold", required_if("filter", "true"))]
    threshold: Option<f64>,

    /// Extract sequences below threshold into FASTA (requires --filter)
    #[structopt(long = "extract", requires("filter"))]
    extract: bool,
    
    #[structopt(short, long, default_value = "output")]
    output: String,
}

/// Structure to hold sequence processing results.
struct SequenceResult {
    id: String,
    diversity: f64,
    sequence: Vec<u8>,
}

/// Processes a FASTA file by reading each sequence and processing it.
fn process_fasta<P: AsRef<Path>>(
    path: P,
    params: &cwt::Params,
    opt: &Opt,
    processed_seqnames: &Arc<Mutex<Vec<String>>>,
    bed_writer: &Arc<Mutex<BufWriter<File>>>, // Added bed_writer as a shared resource
) -> Result<(), std::io::Error> {
    info!("Opening FASTA file: {:?}", path.as_ref());
    let file = File::open(&path)?;
    let reader = BufReader::new(file);
    let mut current_seq = Vec::new();
    let mut current_id = String::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('>') {
            if !current_seq.is_empty() {
                debug!("Processing sequence ID: {}", current_id);
                debug!("Sequence length: {}", current_seq.len());
                process_sequence_fasta(&current_seq, &current_id, params, opt, processed_seqnames, bed_writer)?;
                current_seq.clear();
            }
            current_id = line[1..].trim().split_ascii_whitespace().next().unwrap().to_string();
            info!("Found sequence ID: {}", current_id);
        } else {
            current_seq.extend(line.bytes());
        }
    }

    if !current_seq.is_empty() {
        debug!("Processing last sequence ID: {}", current_id);
        process_sequence_fasta(&current_seq, &current_id, params, opt, processed_seqnames, bed_writer)?;
    }

    info!("Completed processing FASTA file.");
    Ok(())
}

/// Processes a single FASTA sequence.
fn process_sequence_fasta(
    seq: &[u8],
    id: &str,
    params: &cwt::Params,
    opt: &Opt,
    processed_seqnames: &Arc<Mutex<Vec<String>>>,
    bed_writer: &Arc<Mutex<BufWriter<File>>>, // Added bed_writer as a shared resource
) -> Result<(), std::io::Error> {
    debug!("Creating temporary file for sequence ID: {}", id);
    let mut seq_copy = seq.to_vec();
    let temp_cwt = NamedTempFile::new()?;
    let seqname = temp_cwt.path().to_str().unwrap().to_string();

    debug!("Initializing CWT iterator for sequence ID: {}", id);
    let cwt_iterator = cwt::CwtIterator::new(&mut seq_copy, params);

    let mut shannon_diversity = Vec::new();
    let cwt_file = File::create(format!("{}.cwt", id))?;
    let mut writer = BufWriter::with_capacity(1024 * 1024 * 1024, cwt_file);
    let mut length = 0;

    if opt.cwt {
        for batch in cwt_iterator.iter() {
            for row in batch.axis_iter(Axis(0)) {
                for val in row.iter() {
                    writer.write_f64::<LittleEndian>(*val)?;
                }
                let diversity = seq::calculate_shannon_diversity_for_vector(&row.to_vec());
                shannon_diversity.push(diversity);
                debug!("Calculated Shannon diversity: {}", diversity);
                length += 1;
            }
        }
    } else {
        for batch in cwt_iterator.iter() {
            for row in batch.axis_iter(Axis(0)) {
                let diversity = seq::calculate_shannon_diversity_for_vector(&row.to_vec());
                shannon_diversity.push(diversity);
                debug!("Calculated Shannon diversity: {}", diversity);
                length += 1;
            }
        }
    }
    
    {
        let mut seqnames = processed_seqnames.lock().unwrap();
        seqnames.push(seqname);
        debug!("Added sequence name to processed_seqnames.");
    }
    let mut conf_file = File::create(format!("{}.conf", id))?;
    conf_file.write_all(format!("{},{},{}", id, length, params.num).as_bytes())?;

    // Process Shannon diversity for BED output
    // Set threshold if it's not provided
    let threshold = match opt.threshold {
        Some(t) => t,
        None => {
            let mut sorted_diversity = shannon_diversity.clone();
            sorted_diversity.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let index = (sorted_diversity.len() as f64 * 0.01).ceil() as usize;
            let threshold = sorted_diversity[index.min(sorted_diversity.len() - 1)];
            info!("Calculated threshold: {}", threshold);
            threshold
        },
    };
    let mut in_low_diversity_region = false;
    let mut start_pos = 0;
    let mut sum_diversity = 0.0;
    let mut region_length = 0;

    for (i, &diversity) in shannon_diversity.iter().enumerate() {
        if diversity < threshold {
            if !in_low_diversity_region {
                in_low_diversity_region = true;
                start_pos = i;
                sum_diversity = diversity;
                region_length = 1;
            } else {
                sum_diversity += diversity;
                region_length += 1;
            }
        } else {
            if in_low_diversity_region {
                let end_pos = i;
                let mean_diversity = sum_diversity / region_length as f64;
                let repeat_length = end_pos - start_pos;
                // Write BED entry
                if repeat_length as f64 > params.periods.iter().cloned().fold(0./0., f64::max) * 1.5 {
                    let mut bed_writer = bed_writer.lock().unwrap();
                    writeln!(
                        bed_writer,
                        "{}\t{}\t{}\tl={};s={:.4e};w={}",
                        id,
                        start_pos,
                        end_pos,
                        repeat_length,
                        mean_diversity,
                        params.periods.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(",")
                    )?;
                }
                in_low_diversity_region = false;
            }
        }
    }
    if in_low_diversity_region {
        let end_pos = shannon_diversity.len();
        let mean_diversity = sum_diversity / region_length as f64;
        let repeat_length = end_pos - start_pos;
        // Write BED entry
        if repeat_length as f64 > params.periods.iter().cloned().fold(0./0., f64::max) * 1.5 {
            let mut bed_writer = bed_writer.lock().unwrap();
            writeln!(
                bed_writer,
                "{}\t{}\t{}\tl={};s={:.4e};w={}",
                id,
                start_pos,
                end_pos,
                repeat_length,
                mean_diversity,
                params.periods.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(",")
            )?;
        }
    }

    info!("Processed sequence ID: {}", id);
    Ok(())
}

/// Reports the progress of sequence processing.
fn report_progress(total_reads: &AtomicUsize, start_time: &Instant) {
    let reads = total_reads.load(Ordering::Relaxed);
    let elapsed = start_time.elapsed();
    let reads_in_hundred_thousands = reads / 100_000;
    info!("Processed {} hundred thousand reads in {:?}", reads_in_hundred_thousands, elapsed);
}

/// Processes a FASTQ file by reading each sequence and processing it with filtering.
fn process_fastq<P: AsRef<Path> + Sync>(
    path: P,
    params: &cwt::Params,
    processed_seqnames: &Arc<Mutex<Vec<String>>>,
    results: &Arc<Mutex<Vec<SequenceResult>>>,
) -> Result<(), std::io::Error> {
    info!("Opening FASTQ file: {:?}", path.as_ref());
    let file = File::open(&path)?;
    let file_size = file.metadata()?.len();

    debug!("Memory-mapping the FASTQ file.");
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    let num_cpus = sys_info::cpu_num().unwrap_or(4);
    info!("Detected {} CPUs.", num_cpus);

    let chunk_size = file_size / num_cpus as u64;
    let mut start_positions = vec![0u64];

    // Determine chunk boundaries to ensure each chunk starts at a record boundary
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
            debug!("Chunk {} starts at byte position {}.", i, pos);
        }
    }
    start_positions.push(file_size);

    let total_reads = Arc::new(AtomicUsize::new(0));
    let start_time = Instant::now();

    info!("Processing FASTQ file with {} chunks.", start_positions.len() - 1);
    start_positions.par_windows(2).for_each(|window| {
        let start = window[0] as usize;
        let end = window[1] as usize;
        let chunk = &mmap[start..end];

        let mut current_pos = 0;
        while current_pos < chunk.len() {
            // Read the ID line
            if chunk[current_pos] != b'@' {
                warn!("Invalid record start at position {}. Skipping to next line.", current_pos);
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
                warn!("Missing '+' line after sequence ID: {}. Skipping to next record.", String::from_utf8_lossy(id));
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
            
            if let Err(e) = process_sequence_fastq_with_id(id, seq, params, processed_seqnames, results) {
                error!("Error processing sequence: {}", e);
            }

            let reads = total_reads.fetch_add(1, Ordering::Relaxed) + 1;
            if reads % 100_000 == 0 {
                report_progress(&total_reads, &start_time);
            }
        }
    });

    // Final report
    report_progress(&total_reads, &start_time);
    info!("Completed processing FASTQ file.");

    Ok(())
}

/// Processes a single FASTQ sequence with its ID.
fn process_sequence_fastq_with_id(
    id: &[u8], 
    seq: &[u8], 
    params: &cwt::Params, 
    processed_seqnames: &Arc<Mutex<Vec<String>>>, 
    results: &Arc<Mutex<Vec<SequenceResult>>>
) -> Result<(), std::io::Error> {
    let id_str = match std::str::from_utf8(id) {
        Ok(s) => s.to_string(),
        Err(_) => {
            warn!("Invalid UTF-8 sequence ID. Skipping.");
            return Ok(());
        }
    };
    debug!("Processing sequence ID: {}", id_str);

    match process_sequence_fastq(seq, &id_str, params, processed_seqnames) {
        Ok(Some(shannon_diversity_avg)) => {
            // Push the result to the shared vector
            let sequence = seq.to_vec();
            let result = SequenceResult {
                id: id_str,
                diversity: shannon_diversity_avg,
                sequence,
            };
            {
                let mut res = results.lock().unwrap();
                res.push(result);
            }
        },
        Ok(None) => {
            warn!("No Shannon diversity calculated for sequence ID: {}", id_str);
        },
        Err(e) => {
            error!("Processing Error for sequence ID {}: {}", id_str, e);
            return Err(e);
        },
    }

    Ok(())
}

/// Processes a single FASTQ sequence.
fn process_sequence_fastq(
    seq: &[u8], 
    id: &str,
    params: &cwt::Params,
    processed_seqnames: &Arc<Mutex<Vec<String>>>,
) -> Result<Option<f64>, std::io::Error> {
    debug!("Creating temporary file for sequence ID: {}", id);
    let mut seq_copy = seq.to_vec();
    let temp_cwt = NamedTempFile::new()?;
    let seqname = temp_cwt.path().to_str().unwrap().to_string();

    debug!("Initializing CWT iterator for sequence ID: {}", id);
    let cwt_iterator = cwt::CwtIterator::new(&mut seq_copy, params);

    let mut shannon_diversity = Vec::new();

    for batch in cwt_iterator.iter() {
        for row in batch.axis_iter(Axis(0)) {
            let diversity = seq::calculate_shannon_diversity_for_vector(&row.to_vec());
            shannon_diversity.push(diversity);
            debug!("Calculated Shannon diversity: {}", diversity);
        }
    }

    {
        let mut seqnames = processed_seqnames.lock().unwrap();
        seqnames.push(seqname);
        debug!("Added sequence name to processed_seqnames.");
    }

    let average_diversity = if !shannon_diversity.is_empty() {
        Some(shannon_diversity.iter().copied().sum::<f64>() / shannon_diversity.len() as f64)
    } else {
        None
    };

    if let Some(avg) = average_diversity {
        debug!("Calculated average Shannon diversity for ID {}: {}", id, avg);
        Ok(Some(avg))
    } else {
        Ok(None)
    }
}

/// Entry point of the application.
fn main() -> Result<(), std::io::Error> {
    // Initialize the logger
    logger::init_logger(LevelFilter::Info).expect("Failed to initialize logger");

    info!("Starting the STRAWS application.");
    let opt = Opt::from_args();
    debug!("Parsed command-line arguments: {:?}", opt);

    let periods: Vec<f64> = if opt.wavelet_sizes.contains('-') {
        let parts: Vec<&str> = opt.wavelet_sizes.split('-').collect();
        if parts.len() != 2 {
            panic!("Invalid wavelet size range. Expected format start-end");
        }
        let start: f64 = parts[0].trim().parse().expect("Invalid start of wavelet size range");
        let end: f64 = parts[1].trim().parse().expect("Invalid end of wavelet size range");
        let num = 8;
        (0..num).map(|i| start + (end - start) * i as f64 / (num - 1) as f64).collect()
    } else {
        opt.wavelet_sizes
            .split(',')
            .map(|s| s.trim().parse::<f64>().expect("Invalid wavelet size"))
            .collect()
    };

    let num = periods.len();

    let params = cwt::Params {
        num,
        periods,
    };
    debug!("CWT parameters set: {:?}", params);

    let processed_seqnames = Arc::new(Mutex::new(Vec::new()));

    if opt.input.ends_with(".fasta") || opt.input.ends_with(".fa") {
        // Open BED file for writing if filtering is enabled
        let bed_file = File::create(format!("{}.bed", &opt.output))?;
        let bed_writer = Arc::new(Mutex::new(BufWriter::new(bed_file)));
        info!("Processing FASTA file.");
        process_fasta(&opt.input, &params, &opt, &processed_seqnames, &bed_writer)?;
    } else if (opt.input.ends_with(".fastq") || opt.input.ends_with(".fq")) && opt.filter {
        info!("Processing FASTQ file with filtering.");
        // Shared vector to collect processing results
        let results = Arc::new(Mutex::new(Vec::new()));
        process_fastq(&opt.input, &params, &processed_seqnames, &results)?;

        // Collect the results
        let mut results_locked = results.lock().unwrap();
        info!("Total sequences processed: {}", results_locked.len());

        // Sort the results by Shannon diversity (ascending)
        results_locked.sort_by(|a, b| {
            a.diversity.partial_cmp(&b.diversity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        

        // Write the sorted results to a txt file
        let mut txt_file = BufWriter::new(File::create(format!("{}.txt", &opt.output))?);
        for result in results_locked.iter() {
            writeln!(txt_file, "{},{}", result.id, result.diversity)?;
        }
        info!("Written sorted Shannon diversity to output.txt.");

        // If extract is enabled, write sequences below threshold to FASTA
        if opt.extract {
            let threshold = match opt.threshold {
                Some(t) => t,
                None => {
                    let mut sorted_diversity = results_locked.iter().map(|d| d.diversity).collect::<Vec<_>>();
                    sorted_diversity.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let index = (sorted_diversity.len() as f64 * 0.01).ceil() as usize;
                    let threshold = sorted_diversity[index.min(sorted_diversity.len() - 1)];
                    info!("Calculated threshold: {}", threshold);
                    threshold
                },
            };
            let mut fasta_file = BufWriter::new(File::create(format!("{}.fasta", &opt.output))?);
            for result in results_locked.iter().filter(|r| r.diversity < threshold) {
                writeln!(fasta_file, ">{}", result.id)?;
                writeln!(fasta_file, "{}", String::from_utf8_lossy(&result.sequence))?;
            }
            info!("Extracted sequences below threshold {} to file.", threshold);
        }
    } else {
        error!("Unsupported file format or incorrect filtering options.");
        eprintln!("Unsupported file format or incorrect options. Please use .fasta, .fa with appropriate options, or .fastq, .fq with --filter.");
        return Ok(());
    }

    info!("STRAWS application completed successfully.");
    Ok(())
}
