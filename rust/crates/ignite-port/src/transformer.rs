//! One pre-LN transformer block, ported from the Python in rung C3.
//!
//! The point of the port is not that Rust is faster — at these sizes it is not,
//! and nothing here is optimized. The point is that a forward pass you can only
//! describe is a forward pass you do not know. Writing it against Candle forces
//! every implicit choice PyTorch makes for you to become an explicit line:
//! which tensor gets transposed, where the mask is applied, what `nn.GELU()`
//! actually computes.
//!
//! Three of those choices bite, and all three are measured by the driver rather
//! than asserted here:
//!
//! 1. **GELU has two definitions.** PyTorch's `nn.GELU()` defaults to the exact
//!    form, $x\Phi(x)$ with $\Phi$ the Gaussian CDF; Candle's `gelu()` is the
//!    tanh approximation from the original paper, and `gelu_erf()` is the exact
//!    one. Picking the wrong one is not a bug that crashes — it is a quiet
//!    1e-3 disagreement that looks like a precision issue.
//! 2. **`matmul` wants contiguous inputs.** After `transpose`, a Candle tensor
//!    is a view with permuted strides, and `matmul` on it either errors or
//!    silently costs a copy. PyTorch hides this; here it is a `.contiguous()`
//!    you have to write, and forgetting it on a batched path is a bug that does
//!    not appear at batch 1.
//! 3. **`Linear` is $xW^\top$.** Candle's `Linear`, like PyTorch's, stores the
//!    weight as `[out, in]` and transposes on use. Loading a weight in the
//!    wrong orientation gives a shape error if the dimensions differ and
//!    silently wrong numbers if they do not.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{LayerNorm, Linear, Module};

/// Which GELU to use in the MLP.
///
/// Kept as a knob rather than a constant because the difference between the two
/// is the rung's most useful number: it is the size of the error you get from a
/// porting decision that no type system catches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gelu {
    /// $x\Phi(x)$ — what `torch.nn.GELU()` computes by default.
    Exact,
    /// The $\tanh$ approximation — what `Tensor::gelu` computes in Candle.
    Tanh,
}

impl Gelu {
    pub fn apply(self, xs: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Gelu::Exact => xs.gelu_erf()?,
            Gelu::Tanh => xs.gelu()?,
        })
    }
}

/// The weights of one block, in the names the Python side exports.
pub struct Block {
    norm1: LayerNorm,
    qkv: Linear,
    projection: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    heads: usize,
    head_width: usize,
    gelu: Gelu,
}

fn take(weights: &[(String, Tensor)], name: &str) -> Result<Tensor> {
    weights
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, tensor)| tensor.clone())
        .with_context(|| format!("weight {name:?} missing from the export"))
}

impl Block {
    /// Build from the flat `name -> tensor` map the Python driver writes.
    ///
    /// LayerNorm's epsilon is passed explicitly and matches PyTorch's 1e-5
    /// default. It is a small number that changes the answer in the fifth
    /// decimal, which is precisely the range this port is trying to certify,
    /// so it cannot be left to whichever default each library happens to hold.
    pub fn from_weights(weights: &[(String, Tensor)], heads: usize, gelu: Gelu) -> Result<Self> {
        let norm1 = LayerNorm::new(take(weights, "norm1.weight")?, take(weights, "norm1.bias")?, 1e-5);
        let norm2 = LayerNorm::new(take(weights, "norm2.weight")?, take(weights, "norm2.bias")?, 1e-5);
        let qkv = Linear::new(take(weights, "attention.qkv.weight")?, Some(take(weights, "attention.qkv.bias")?));
        let projection = Linear::new(
            take(weights, "attention.projection.weight")?,
            Some(take(weights, "attention.projection.bias")?),
        );
        let fc1 = Linear::new(take(weights, "mlp.0.weight")?, Some(take(weights, "mlp.0.bias")?));
        let fc2 = Linear::new(take(weights, "mlp.2.weight")?, Some(take(weights, "mlp.2.bias")?));
        let width = norm1.weight().dims1()?;
        anyhow::ensure!(width % heads == 0, "width {width} does not divide into {heads} heads");
        Ok(Self {
            norm1,
            qkv,
            projection,
            norm2,
            fc1,
            fc2,
            heads,
            head_width: width / heads,
            gelu,
        })
    }

    /// Causal mask as a `u8` tensor, 1 where attention must not look.
    ///
    /// Built by hand rather than with a helper so the convention is visible:
    /// strictly upper triangular, so position $t$ sees $0..=t$. Filled with a
    /// large negative number rather than `-inf`, because `-inf` in a row that is
    /// entirely masked produces `NaN` out of softmax rather than a uniform row —
    /// not reachable here, where every row has at least its own position, but
    /// the habit is what keeps it unreachable.
    fn causal_mask(length: usize, device: &Device) -> Result<Tensor> {
        let mut data = vec![0u8; length * length];
        for row in 0..length {
            for column in (row + 1)..length {
                data[row * length + column] = 1;
            }
        }
        Ok(Tensor::from_vec(data, (length, length), device)?)
    }

    pub fn attention(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, length, width) = xs.dims3()?;
        let qkv = self.qkv.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, width)?;
        let k = qkv.narrow(D::Minus1, width, width)?;
        let v = qkv.narrow(D::Minus1, 2 * width, width)?;

        // (batch, heads, length, head_width). The `contiguous` is load-bearing:
        // `transpose` only permutes strides, and the `matmul` below needs a
        // real layout.
        let reshape = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((batch, length, self.heads, self.head_width))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let (q, k, v) = (reshape(q)?, reshape(k)?, reshape(v)?);

        let scale = 1.0 / (self.head_width as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale)?;

        let mask = Self::causal_mask(length, xs.device())?
            .broadcast_as(scores.shape())?
            .contiguous()?;
        let blocked = Tensor::full(-1e30f32, scores.shape(), xs.device())?.to_dtype(scores.dtype())?;
        let scores = mask.where_cond(&blocked, &scores)?;

        let weights = softmax_last_dim(&scores)?;
        let attended = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, length, width))?;
        Ok(self.projection.forward(&attended)?)
    }

    /// Pre-LN: normalize *inside* the residual branch, leaving the skip clean.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let attended = self.attention(&self.norm1.forward(xs)?)?;
        let xs = (xs + attended)?;
        let hidden = self.gelu.apply(&self.fc1.forward(&self.norm2.forward(&xs)?)?)?;
        Ok((&xs + self.fc2.forward(&hidden)?)?)
    }
}

/// Load every array from a `.npz` written by numpy, as f32 on the CPU.
pub fn load_npz(path: &str, device: &Device) -> Result<Vec<(String, Tensor)>> {
    let entries = Tensor::read_npz(path).with_context(|| format!("reading {path}"))?;
    entries
        .into_iter()
        .map(|(name, tensor)| Ok((name, tensor.to_dtype(DType::F32)?.to_device(device)?)))
        .collect()
}
