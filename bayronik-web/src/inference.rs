use ndarray::{Array, Array4, IxDyn};
use tract_onnx::prelude::*;

pub struct Emulator {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    conditional: bool,
    resolution: usize,
}

impl Emulator {
    pub fn from_bytes(model_bytes: &[u8], conditional: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(model_bytes))?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            model,
            conditional,
            resolution: 256,
        })
    }

    pub fn run(&self, input: &[f32], conditions: Option<&[f32]>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n = self.resolution;
        
        let input_log: Vec<f32> = input.iter().map(|&x| (x + 1.0).ln()).collect();
        
        let input_tensor: Tensor = Array4::from_shape_vec(
            (1, 1, n, n),
            input_log,
        )?.into();

        let output = if self.conditional {
            if let Some(conds) = conditions {
                let cond_tensor: Tensor = Array::from_shape_vec(
                    IxDyn(&[1, conds.len()]),
                    conds.to_vec(),
                )?.into();
                
                self.model.run(tvec!(input_tensor.into(), cond_tensor.into()))?
            } else {
                return Err("Conditional model requires conditions".into());
            }
        } else {
            self.model.run(tvec!(input_tensor.into()))?
        };

        let output_tensor = output[0].to_array_view::<f32>()?;
        let output_vec: Vec<f32> = output_tensor.iter().copied().collect();

        Ok(output_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emulator_creation() {
        let dummy_input = vec![1.0f32; 256 * 256];
        let _result = dummy_input.iter().map(|&x| (x + 1.0).ln()).collect::<Vec<_>>();
    }
}
