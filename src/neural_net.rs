pub mod act_func;
pub mod net_layer;
pub mod net_shape;
pub mod training_helpers;

use act_func::ActFunc;
use net_layer::{NetLayer, NetLayerType};
use net_shape::NetShape;
use training_helpers::{TData, TSettings};

use std::time::Instant;
use std::fmt::Debug;
use ndarray::{ Array2, ArrayView, s};
use rand::{distributions::{Distribution, Uniform, Bernoulli}, rngs::StdRng, SeedableRng};


pub struct NeuralNet {
    layers: Vec<NetLayer>,
}

impl NeuralNet {
    pub fn new(shape: NetShape, rand_seed: u64) -> Result<NeuralNet, String> {
        let mut layers: Vec<NetLayer> = Vec::new();
        
        //Based on the provided shape of the network (# of neurons per layer), weights are generated at random with a uniform distribution betwee +-2
        let mut rng = StdRng::seed_from_u64(rand_seed);
        let range = Uniform::new(-1.0, 1.0);

        //creates weights between hidden layers
        for l_type in shape.layer_types().drain(..) {
            layers.push(
                match NetLayer::new(l_type, &mut rng, &range) {
                    Ok(l) => l,
                    Err(e) => return Err(format!("Error in converting from layer type to layer: {}", e))
                }
            );
        }
        
        Ok(NeuralNet {
            layers
        })
    }

    pub fn train(&mut self, training_data: TData, testing_data: Option<TData>, settings: &TSettings) -> Result<(), String> {
        
        //ensures that the matricies for the training_data.input()s and outputs are correct
        if training_data.input()[0].shape()[1] != self.input_node_num() {
            return Err(format!("1st training_data.input() matrix doesn't have the expected shape. {:?} != {:?}", training_data.input()[0].shape()[1], self.input_node_num()))
        }

        if training_data.output()[0].shape()[1] != self.output_node_num() {
            return Err(format!("1st output matrix doesn't have the expected shape. {:?} != {:?}", training_data.output()[0].shape()[1], self.output_node_num()))
        }
        
        let mut start = Instant::now();

        for iter_num in 0..(settings.iterations() + 1) {
            let training_error = {
                let mut current_error = 0.0;
                for i in 0..training_data.input().len() {
                    self.forward_propogate(training_data.input()[i].clone());
                    
                    let layer_delta = (self.cached_output() - &training_data.output()[i]).reversed_axes();

                    current_error += (&layer_delta * &layer_delta).sum();

                    self.backward_propogate(training_data.input()[i].clone(), layer_delta, settings)
                } 

                current_error / training_data.input().len() as f64
            };

            if settings.print_frequency() != 0 {
                if iter_num % (settings.iterations() / settings.print_frequency()) == 0 || iter_num == settings.iterations() {
                    match testing_data {
                        Some(_) => {
                            let mut testing_error = 0.0;
                            let data = testing_data.as_ref().unwrap();
                            
                            for i in 0..data.input().len() {
                                self.forward_propogate(data.input()[i].clone());
                                let test_error = &data.output()[i] - self.cached_output();
                                let test_error = (&test_error * &test_error).sum();

                                testing_error += test_error;
                            }
                            
                            testing_error = testing_error / data.input().len() as f64;

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

    pub fn forward_propogate(&mut self, mut input: Array2<f64>) {

        for l_index in 0..self.layers.len() {
            self.layers[l_index].forward_propogate(input);
            input = self.layers[l_index].output().clone()
        }
    }

    pub fn backward_propogate(&mut self, input: Array2<f64>, mut layer_delta: Array2<f64>, settings: &TSettings) {

        for l_index in (1..self.layers.len()).rev() {
            //output of th elayer right before this one, which makes it the input
            let temp_input = self.layers[l_index - 1].output().clone();

            layer_delta = self.layers[l_index].back_propogate(settings.alpha(), settings.dropout(), temp_input, layer_delta);
        }

        self.layers[0].back_propogate(settings.alpha(), settings.dropout(), input, layer_delta);
    }

    pub fn input_node_num(&self) -> usize {
        self.layers[0].layer_type().input_node_num()
    }

    pub fn output_node_num(&self) -> usize {
        self.layers.last().unwrap().layer_type().output_node_num()
    }

    pub fn cached_output(&self) -> &Array2<f64> {
        self.layers.last().unwrap().output()
    }
}

impl Debug for NeuralNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut layers_string = String::new();
        for layer in self.layers.iter() {
            layers_string = format!("{}\n{:?}\n", layers_string, layer)
        }
        write!(f, "Neural Net: \nLayers:\n {}", layers_string)
    }
}