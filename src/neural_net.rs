use std::{cmp, fmt::Debug};

use nalgebra::DMatrix;
use rand::{distributions::{Distribution, Uniform}, rngs::StdRng, SeedableRng};


pub struct NeuralNet {
    act_func: ActFunc,
    layer_output_cache: Vec<DMatrix<f64>>,
    shape: Vec<usize>,
    weights: Vec<DMatrix<f64>>,
}

impl NeuralNet {
    pub fn new(act_func: ActFunc, shape: Vec<usize>, rand: bool) -> Result<NeuralNet, String> {
        if shape.len() < 2 {
            return Err("The size var must have at least 2 numbers which signify the number of inputs and outputs.".into());
        }

        let mut weights: Vec<DMatrix<f64>> = Vec::new();

        if rand {
            let mut rng = StdRng::seed_from_u64(0);
            let range = Uniform::new(0.0, 2.0);

            for i in 1..shape.len() {
                //Weights are matricies with a number of rows equal to the number of inputs + bias node, and number of columns equal to the number of outputs.
                weights.push(
                    DMatrix::from_fn(shape[i-1] + 1, shape[i], |i, j| range.sample(&mut rng))
                );
            }
        }
        else {
            for i in 1..shape.len() {
                //Weights are matricies with a number of rows equal to the number of inputs + bias node, and number of columns equal to the number of outputs.
                weights.push(
                    DMatrix::from_fn(shape[i-1] + 1, shape[i], |i, j| ((i+j + 1) as f64))
                );
            }
        }

        Ok(NeuralNet {
            act_func,
            layer_output_cache: vec![DMatrix::from_element(1, 1, 1.0)],
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
            //extending input to include a bias node

            let mut input: DMatrix<f64> = self.layer_output_cache.last().unwrap().clone();
            let last_column_index = input.shape().1;
            input = input.insert_column(last_column_index, 1.0);

            self.layer_output_cache.push(
                self.act_func.apply( input * layer)
            )
        }

        Ok(())
    }

    pub fn backward_propogate(&mut self, network_error: f64, alpha: f64) -> Result<(), String> {
        //network_error / delta = average absolute valoe of calculated - expected output
        let alpha = 0.1;

        if self.weights.len() != self.layer_output_cache.len() - 1 {
            return Err("Run forward_propogate before backward.".into())
        }
        else if alpha > 1.0 || alpha < 0.0 {
            return Err("Alpha must be between zero and 1, as it is used to slow down the chnage in the weights as to not overshoot".into());
        }

        //println!("Weights: {:?}\nLayer Output: {:?}", self.weights, self.layer_output_cache);

        //position of second to last output and of last weights
        let position = self.layer_output_cache.len() - 2;

        //dE/dw, change in weights to reduce error = input value sof weights * error of output layer
        let mut weight_deltas: Vec<DMatrix<f64>> = Vec::new();

        //acconting for bias node
        let mut weight_input = self.layer_output_cache[position].clone();
        let mut last_column_index = weight_input.shape().1;
        weight_input = weight_input.insert_column(last_column_index, 1.0);

        weight_deltas.push(
            //calculates delta for the last set of weights
            (weight_input * network_error).transpose()
        );


        //finds the error of the outputs of the weights
        //finds the error of the output layer/nodes right before the last weights
        let mut layer_error: DMatrix<f64> = self.weights[position].clone() * network_error;
        //removes layer error for bias node as bias node
        let mut last_row_index = layer_error.shape().0 - 1;
        layer_error = layer_error.remove_row(last_row_index);

        //starts off with 2nd to last set of weights
        for i in (0..(self.weights.len() - 1)).rev() {
            //println!("\n{})Layer Err: {:?}\n", i, layer_error);

            //accointing for bias node
            weight_input = self.layer_output_cache[i].clone();
            last_column_index = weight_input.shape().1;
            weight_input = weight_input.insert_column(last_column_index, 1.0);

            weight_deltas.push(
                //calculates delta for the weights
                (layer_error.clone() * weight_input).transpose()
            );

            //calculates delta for the nodes inputing into the current weights
            //In the next iteration of the loop, this will be the error of the output nodes relatvie to the weights 
            layer_error = self.weights[i].clone() * layer_error;

            last_row_index = layer_error.shape().1 - 1;
            layer_error = layer_error.remove_row(last_row_index);

        }

        //println!("\n\nWeight Deltas: {:?}", weight_deltas);

        for i in 0..self.weights.len() {
            self.weights[i] -= weight_deltas.pop().unwrap() * alpha
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

impl Debug for NeuralNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Neural Net: \nActivation Function: {:?}\nShape: {:?}\nOutput Cache: {:#?}\nWeights {:?}", self.act_func, self.shape, self.layer_output_cache.last().unwrap()[0], self.weights)
    }
}


//Activation Funciton
#[derive(Debug)]
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