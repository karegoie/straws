//! ```cargo
//! [dependencies]
//! rand = "*"
//! ```

use rand::Rng;

fn main() {
    let seq_length = 80000;
    let mut seq: Vec<char> = Vec::with_capacity(seq_length);
    let nucleotides = ['A', 'C', 'G', 'T'];
    let mut rng = rand::thread_rng();

    for _ in 0..seq_length {
        seq.push(nucleotides[rng.gen_range(0..4)]);
    }

    let repeats = [
        (10000, 60),
        (15000, 600),
        (20000, 3000),
        (45000, 60),
        (50000, 600),
        (55000, 3000),
        (70000, 2000)
    ];

    for &(position, unit_length) in &repeats {
        let unit_seq: Vec<char> = (0..unit_length).map(|_| nucleotides[rng.gen_range(0..4)]).collect();
        let repeat_length = unit_length * 3;
        let repeat_seq: Vec<char> = unit_seq.iter().cycle().take(repeat_length).cloned().collect();

        let start_idx = position - 1;
        let end_idx = std::cmp::min(start_idx + repeat_length, seq_length);
        let available_length = end_idx - start_idx;
        let adjusted_repeat_seq: Vec<char> = repeat_seq.into_iter().take(available_length).collect();

        seq.splice(start_idx..end_idx, adjusted_repeat_seq.into_iter());
    }

    let seq_string: String = seq.into_iter().collect();
    println!(">set");
    println!("{}", seq_string);
}