use clap;
use clap::Parser;
use std::str::FromStr;
mod openapi_example;

#[derive(Debug, PartialEq, Clone)]
enum Example {
    OAPI,
}

impl FromStr for Example {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "oapi" => Ok(Example::OAPI),
            _ => Err(format!("unknown example: {}", s)),
        }
    }
}

#[derive(Parser, Debug)]
struct MainOpt {
    /// set the listen addr
    #[clap(short = 'n', long = "name", default_value = "oapi")]
    example_name: Example,
}

fn main() {
    let opt = MainOpt::parse();
    match opt.example_name {
        Example::OAPI => openapi_example::main(),
    }
}
