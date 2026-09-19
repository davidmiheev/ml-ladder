//! Run the ported block over an exported input and write the result.
//!
//! Usage: `block-forward <weights.npz> <input.npy> <output.npy> <heads> <gelu>`
//! where `gelu` is `exact` or `tanh`. Deliberately a plain positional CLI: the
//! Python driver is the only caller, and an argument parser would be more
//! dependency than the rung is worth.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use ignite_port::transformer::{load_npz, Block, Gelu};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        bail!("usage: {} <weights.npz> <input.npy> <output.npy> <heads> <exact|tanh>", args[0]);
    }
    let device = Device::Cpu;
    let weights = load_npz(&args[1], &device)?;
    let input = Tensor::read_npy(&args[2])?.to_dtype(DType::F32)?.to_device(&device)?;
    let heads: usize = args[4].parse()?;
    let gelu = match args[5].as_str() {
        "exact" => Gelu::Exact,
        "tanh" => Gelu::Tanh,
        other => bail!("unknown gelu {other:?}; expected exact or tanh"),
    };

    let block = Block::from_weights(&weights, heads, gelu)?;
    let output = block.forward(&input)?;
    output.write_npy(&args[3])?;
    println!("{:?}", output.dims());
    Ok(())
}
