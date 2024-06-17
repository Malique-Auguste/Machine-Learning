use super::{act_func::ActFunc, NetShape, NeuralNet};
use ndarray::Array2;

pub struct CNN {
    atc_func: ActFunc,
    shape: Vec<[usize;2]>,
    weights: Vec<Array2<f64>>,
}

impl ActFunc {
    fn new(act_func: ActFunc, shape: NetShape, rand_seed: u64) -> Result<CNN, String> {
        


        unimplemented!()
    }

    pub fn forward_propogate(&mut self) {
        
        
    }
}