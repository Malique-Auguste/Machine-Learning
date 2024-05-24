use std::cmp;

use nalgebra::DMatrix;

pub struct NeuralNet {
    act_func: ActFunc,
    layer_output_cache: Vec<DMatrix<f64>>,
    shape: Vec<usize>,
    weights: Vec<DMatrix<f64>>,
}

impl NeuralNet {
    pub fn new(act_func: ActFunc, shape: Vec<usize>) -> Result<NeuralNet, String> {
        if shape.len() < 2 {
            return Err("The size var must have at least 2 numbers which signify the number of inputs and outputs.".into());
        }

        let mut weights: Vec<DMatrix<f64>> = Vec::new();

        for i in 1..shape.len() {
            //Weights are matricies with a number of rows equal to the number of inputs, and number of columns equal to the number of outputs.
            weights.push(DMatrix::from_element(shape[i-1], shape[i], 1.0))
        }

        Ok(NeuralNet {
            act_func,
            layer_output_cache: Vec::new(),
            shape,
            weights
        })
    }

    pub fn forward_propogate(&mut self, input: DMatrix<f64>) -> Result<(), String> {
        if input.shape() != (1, self.shape[0]) {
            return Err(format!("Shape of input doesn't match expected. {:?} != (1, {})", input.shape(), self.shape[0]))
        }

        self.layer_output_cache = Vec::new();
        self.layer_output_cache.push(input);

        for layer in self.weights.iter() {
            self.layer_output_cache.push(
                self.act_func.apply(self.layer_output_cache.last().unwrap() * layer)
            )
        }

        Ok(())
    }

    pub fn backward_propogate(&mut self, network_error: f64) -> Result<(), String> {
        //network_error / delta = average absolute valoe of calculated - expected output
        let alpha = 0.1;

        if self.weights.len() != self.layer_output_cache.len() - 1 {
            return Err("Run forward_propogate before backward.".into())
        }

        //position of second to last output and of last weights
        let position = self.layer_output_cache.len() - 2;

        //dE/dw, change in weights to reduce error = input value sof weights * error of output layer
        let mut weight_deltas: Vec<DMatrix<f64>> = Vec::new();
        weight_deltas.push(
            //calculates delta for the last set of weights
            self.layer_output_cache[position].clone() * network_error
        );

        //finds the error of the outputs of the weights
        //finds the error of the output layer/nodes right after the weights
        let mut layer_error: DMatrix<f64> = self.weights[position + 1].clone() * network_error;

        //starts off with 2nd to last set of weights
        for i in (self.weights.len() - 2)..0 {
            weight_deltas.push(
                //calculates delta for the weights
                self.layer_output_cache[i].clone() * layer_error.clone()
            );

            //calculates delta for the nodes inputing into the current weights
            //In the next iteration of the loop, this will be the error of the output nodes relatvie to the weights 
            layer_error = self.weights[i].clone() * layer_error
        }

        for i in 0..self.weights.len() {
            self.weights[i] -= weight_deltas.pop().unwrap()
        }
        
        Ok(())
    }

    pub fn cached_output(&self) -> &DMatrix<f64> {
        self.layer_output_cache.last().unwrap()
    }

    pub fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn get_weights(&self) -> &Vec<DMatrix<f64>> {
        &self.weights
    }
}


//Activation Funciton
pub enum ActFunc {
    ReLU
}

impl ActFunc {
    pub fn apply(&self, mut input: DMatrix<f64>) -> DMatrix<f64> {
        match self {
            ActFunc::ReLU => {
                for elem in input.iter_mut() {
                    if *elem < 0.0 {
                        *elem = 0.0
                    }
                }
            }
        }

        input
    }
}