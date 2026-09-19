//! Rust ports of ladder rungs, kept next to the Python so the two can be diffed.
//!
//! Rung G3 lands here: a transformer-block forward pass written against Candle
//! ([`transformer`]), checked numerically against the Python implementation
//! from rung C3. This module keeps the shape conventions the two sides agree
//! on.

pub mod transformer;

/// Shape of one transformer block's activations, in the order the Python side
/// uses: `(batch, seq, d_model)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockShape {
    pub batch: usize,
    pub seq: usize,
    pub d_model: usize,
}

impl BlockShape {
    pub fn new(batch: usize, seq: usize, d_model: usize) -> Self {
        Self {
            batch,
            seq,
            d_model,
        }
    }

    /// Number of elements in one activation tensor of this shape.
    pub fn numel(&self) -> usize {
        self.batch * self.seq * self.d_model
    }

    /// Per-head width, or `None` when `d_model` does not divide evenly.
    pub fn head_dim(&self, heads: usize) -> Option<usize> {
        (heads > 0 && self.d_model % heads == 0).then(|| self.d_model / heads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_multiplies_dimensions() {
        assert_eq!(BlockShape::new(2, 8, 64).numel(), 1024);
    }

    #[test]
    fn head_dim_requires_even_split() {
        let shape = BlockShape::new(1, 4, 64);
        assert_eq!(shape.head_dim(8), Some(8));
        assert_eq!(shape.head_dim(7), None);
        assert_eq!(shape.head_dim(0), None);
    }
}
