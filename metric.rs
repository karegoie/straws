//! ```cargo
//! [dependencies]
//! bio = "*"
//! ```

use bio::io::bed;
use std::collections::{HashMap, HashSet};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

type Interval = (i64, i64);

fn read_bed_to_intervals(file_path: &str) -> Result<HashMap<String, Vec<Interval>>, Box<dyn Error>> {
    let mut reader = bed::Reader::from_file(file_path)?;
    let mut intervals: HashMap<String, Vec<Interval>> = HashMap::new();

    for result in reader.records() {
        let record = result?;
        let chrom = record.chrom().to_string();
        let start = record.start() as i64;
        let end = record.end() as i64;

        intervals.entry(chrom).or_insert_with(Vec::new).push((start, end));
    }

    Ok(intervals)
}

fn read_chrom_sizes(file_path: &str) -> Result<HashMap<String, i64>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut chrom_sizes = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let mut parts = line.split_whitespace();
        if let (Some(chrom), Some(size_str)) = (parts.next(), parts.next()) {
            let size: i64 = size_str.parse()?;
            chrom_sizes.insert(chrom.to_string(), size);
        }
    }

    Ok(chrom_sizes)
}

fn merge_intervals(intervals: &Vec<Interval>) -> Vec<Interval> {
    let mut intervals = intervals.clone();
    intervals.sort_by_key(|k| k.0);
    let mut merged: Vec<Interval> = Vec::new();

    for interval in intervals {
        if let Some(last) = merged.last_mut() {
            if interval.0 < last.1 {
                last.1 = last.1.max(interval.1);
            } else {
                merged.push(interval);
            }
        } else {
            merged.push(interval);
        }
    }

    merged
}

fn total_nucleotides(intervals: &Vec<Interval>) -> i64 {
    intervals.iter().map(|&(start, end)| end - start).sum()
}

fn overlapping_nucleotides(intervals1: &Vec<Interval>, intervals2: &Vec<Interval>) -> i64 {
    let mut i = 0;
    let mut j = 0;
    let mut total_overlap = 0;

    let intervals1 = merge_intervals(intervals1);
    let intervals2 = merge_intervals(intervals2);

    while i < intervals1.len() && j < intervals2.len() {
        let (start1, end1) = intervals1[i];
        let (start2, end2) = intervals2[j];

        if end1 <= start2 {
            i += 1;
        } else if end2 <= start1 {
            j += 1;
        } else {
            let overlap_start = start1.max(start2);
            let overlap_end = end1.min(end2);
            total_overlap += overlap_end - overlap_start;

            if end1 <= end2 {
                i += 1;
            } else {
                j += 1;
            }
        }
    }

    total_overlap
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <reference_bed> <target_bed> <chrom_sizes>", args[0]);
        std::process::exit(1);
    }

    let reference_bed = &args[1];
    let target_bed = &args[2];
    let chrom_sizes_file = &args[3];

    let reference_intervals = read_bed_to_intervals(reference_bed)?;
    let target_intervals = read_bed_to_intervals(target_bed)?;
    let chrom_sizes = read_chrom_sizes(chrom_sizes_file)?;

    let mut total_tp = 0i64;
    let mut total_fp = 0i64;
    let mut total_fn = 0i64;
    let mut total_tn = 0i64;

    let mut chromosomes: HashSet<String> = HashSet::new();
    chromosomes.extend(reference_intervals.keys().cloned());
    chromosomes.extend(target_intervals.keys().cloned());
    chromosomes.extend(chrom_sizes.keys().cloned());

    // Define empty vectors outside the loop
    let empty_intervals = Vec::new();

    for chrom in chromosomes {
        let ref_intervals = reference_intervals.get(&chrom).unwrap_or(&empty_intervals);
        let tar_intervals = target_intervals.get(&chrom).unwrap_or(&empty_intervals);

        let merged_ref = merge_intervals(ref_intervals);
        let merged_tar = merge_intervals(tar_intervals);

        let tp = overlapping_nucleotides(&merged_ref, &merged_tar);
        let ref_total = total_nucleotides(&merged_ref);
        let tar_total = total_nucleotides(&merged_tar);

        let fn_count = ref_total - tp;
        let fp = tar_total - tp;

        let chrom_length = *chrom_sizes.get(&chrom).unwrap_or(&0);

        let tn = chrom_length - (tp + fp + fn_count);

        total_tp += tp;
        total_fp += fp;
        total_fn += fn_count;
        total_tn += tn;
    }

    let sensitivity = if (total_tp + total_fn) > 0 {
        total_tp as f64 / (total_tp + total_fn) as f64
    } else {
        0.0
    };

    let specificity = if (total_tn + total_fp) > 0 {
        total_tn as f64 / (total_tn + total_fp) as f64
    } else {
        0.0
    };

    let precision = if (total_tp + total_fp) > 0 {
        total_tp as f64 / (total_tp + total_fp) as f64
    } else {
        0.0
    };

    println!("Sensitivity (Sn): {:.2}%", sensitivity * 100.0);
    println!("Specificity (Sp): {:.2}%", specificity * 100.0);
    println!("Precision (Pr): {:.2}%", precision * 100.0);

    Ok(())
}