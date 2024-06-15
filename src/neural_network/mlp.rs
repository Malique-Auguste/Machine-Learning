use super::act_func::ActFunc;
use super::{TData, TSettings};

use std::time::{Duration, Instant};
use std::fmt::Debug;
use nalgebra::{iter, DMatrix};
use rand::{distributions::{Distribution, Uniform, Bernoulli}, rngs::StdRng, SeedableRng};

//Multilayer Perceptron
pub struct MLP {
    act_func: ActFunc,
    layer_output_cache: Vec<DMatrix<f64>>,
    shape: Vec<usize>,
    weights: Vec<DMatrix<f64>>,
}

impl MLP {
    pub fn new(act_func: ActFunc, shape: Vec<usize>, rand_seed: u64) -> Result<MLP, String> {
        if shape.len() < 2 {
            return Err("The size var must have at least 2 numbers which signify the number of inputs and outputs.".into());
        }

        let mut weights: Vec<DMatrix<f64>> = Vec::new();
        
        //Based on the provided shape of the network (# of neurons per layer), weights are generated at random with a uniform distribution betwee +-2
        let mut rng = StdRng::seed_from_u64(rand_seed);
        let range = Uniform::new(-1.0, 1.0);

        for i in 1..shape.len() {
            //Weights are matricies with a number of rows equal to the number of inputs + bias node, and number of columns equal to the number of outputs.
            weights.push(
                DMatrix::from_fn(shape[i-1] + 1, shape[i], |_, _| range.sample(&mut rng))
            );
        }
        
        Ok(MLP {
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

    pub fn backward_propogate(&mut self, mut layer_error: DMatrix<f64>,dropout: bool) -> Result<Vec<DMatrix<f64>>, String> {
        //layer_error = calculated output - expected output (after forwrd propogation)
        //dropout randomly turns off half the nodes on back propogation as to not overfit network to data

        //ensures that forward propogation was done nd thus that there are layer outputs
        if self.weights.len() != self.layer_output_cache.len() - 1 {
            return Err("Run forward_propogate before backward.".into())
        }


        //Vector representing the error in the weights between layers
        //This vector is reversed. That is, the last set of weights in the network (which connects to the output layer) is first in this list
        let mut weight_deltas: Vec<DMatrix<f64>> = Vec::new();

        //shows the input into the weights, or rather the output of the lyaer immediately before the weights
        let mut weight_input: DMatrix<f64>;

        //used for dropout
        let mut rng = StdRng::seed_from_u64(0);
        let range = Bernoulli::new(0.5).unwrap();

        //backpropogates from the last set of weights
        for i in (0..(self.weights.len())).rev() {

            //accointing for bias node
            weight_input = self.layer_output_cache[i].clone();
            let last_column_index = weight_input.shape().1;
            weight_input = weight_input.insert_column(last_column_index, 1.0);

            //dropout vector essentially randomly turns off half of the nodes to intorduce noise and avoid overfitting. thus multiply by 2
            if dropout {
                let dropout_vector = DMatrix::from_fn(weight_input.shape().0, weight_input.shape().1, |_, _| range.sample(&mut rng) as i8 as f64);
                weight_input = weight_input.component_mul(&dropout_vector) * 2.0;
            }

            //calculates the error for each layer depending on the function
            let (new_weight_delta, new_layer_error) = self.act_func.delta(layer_error, weight_input, self.weights[i].clone());
            
            weight_deltas.push(new_weight_delta);

            //saves this iteration of the layer error, to be used in the following caluclation
            layer_error = new_layer_error;

            //removes error associated with bias node as the bias node doesn't backpropogate, that is: it has nothing connecitng to it
            let last_row_index = layer_error.shape().0 - 1;
            layer_error = layer_error.remove_row(last_row_index);

        }

        
        Ok(weight_deltas)

    }

    fn train_epoch(&mut self, data: &TData, settings: &TSettings) -> Result<f64, String>  {
        let mut current_error = 0.0;
        for i in 0..data.input.len() {

            self.forward_propogate(data.input[i].clone()).unwrap();
            let layer_delta = (self.cached_output() - data.output[i].clone()).transpose();

            current_error += (layer_delta.clone().component_mul(&layer_delta)).sum();

            //println!("\n{:?}\n", nn);
            let mut weight_deltas = self.backward_propogate(layer_delta, settings.dropout).unwrap();
            for i in 0..self.weights.len() {
                self.weights[i] -= weight_deltas.pop().unwrap() * settings.alpha
            }
        } 

        Ok(current_error / data.input.len() as f64)
    }

    pub fn train(&mut self, training_data: TData, testing_data: Option<TData>, settings: &TSettings) -> Result<(), String> {
        
        //ensures that the matricies for the training_data.inputs and outputs are correct
        if training_data.input[0].shape() != (1, self.shape[0]) {
            return Err(format!("1st training_data.input matrix doesn't have the expected shape. {:?} != {:?}", training_data.input[0].shape(), (1, self.shape[0])))
        }

        for i in 0..training_data.output.len() {
            if training_data.output[i].shape() != (1, *self.shape.last().unwrap()) {
                return Err(format!("{}th output matrix doesn't have the expected shape. {:?} != {:?}", i, training_data.output[i].shape(), (1, *self.shape.last().unwrap())))
            }
        }
        
        let mut start = Instant::now();

        for iter_num in 0..(settings.iterations + 1) {
            let training_error = match self.train_epoch(&training_data, &settings) {
                Ok(n) => n,
                Err(e) => return Err(e)
            };

            if settings.print_frequency != 0 {
                if iter_num % (settings.iterations / settings.print_frequency) == 0 || iter_num == settings.iterations {
                    match testing_data {
                        Some(_) => {
                            let mut testing_error = 0.0;
                            let data = testing_data.as_ref().unwrap();
                            for i in 0..data.input.len() {
                                let test_output = self.test(data.input[i].clone()).unwrap();
                                let test_error = data.output[i].clone() - test_output;
                                let test_error = test_error.component_mul(&test_error).sum();

                                testing_error += test_error;
                            }
                            
                            testing_error = testing_error / data.input.len() as f64;

                            println!("{}) Training Error: {:.5}\tTesting Error: {:.5}\tTime Taken: {:?}", iter_num, training_error, testing_error, start.elapsed());
                            start = Instant::now();
                        },
                        None => {
                            println!("{}) Training Error: {:.5}\tTime Taken: {:?}", iter_num, training_error, start.elapsed());
                            start = Instant::now();
                        }
                    }
                    
                }
            }
        }

        Ok(())
    }

    pub fn test(&mut self, input: DMatrix<f64>) -> Result<&DMatrix<f64>, String> {
        match self.forward_propogate(input) {
            Ok(_) => (),
            Err(e) => return Err(e)
        }

        Ok(self.cached_output())
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

impl Debug for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Neural Net: \nActivation Function: {:?}\nShape: {:?}\nOutput Cache: {:#?}\nWeights {:?}", self.act_func, self.shape, self.layer_output_cache.last().unwrap()[0], self.weights)
    }
}
