use log::{Record, Level, Metadata, LevelFilter, SetLoggerError};
use std::time::{SystemTime, UNIX_EPOCH};
use std::fmt::Write as FmtWrite;


pub struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let (color_code, reset_code) = get_color_codes(record.level());
            let timestamp = get_formatted_time();
            println!("{}{} - {} - {}{}",
                     color_code,
                     timestamp,
                     record.level(),
                     record.args(),
                     reset_code);
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

pub fn init_logger(level: LevelFilter) -> Result<(), SetLoggerError> {
    log::set_logger(&LOGGER)
        .map(|()| log::set_max_level(level))
}

fn get_formatted_time() -> String {
    let now = SystemTime::now();
    let duration = now.duration_since(UNIX_EPOCH).unwrap();
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();

    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;

    let mut output = String::new();
    write!(output, "{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis).unwrap();
    output
}

fn get_color_codes(level: Level) -> (&'static str, &'static str) {
    match level {
        Level::Error => ("\x1b[31m", "\x1b[0m"),   // Red
        Level::Warn => ("\x1b[33m", "\x1b[0m"),    // Yellow
        Level::Info => ("\x1b[32m", "\x1b[0m"),    // Green
        Level::Debug => ("\x1b[36m", "\x1b[0m"),   // Cyan
        Level::Trace => ("\x1b[35m", "\x1b[0m"),   // Magenta
    }
}