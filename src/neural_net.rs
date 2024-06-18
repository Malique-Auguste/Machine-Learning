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
    act_func: ActFunc,
    layer_output_cache: Vec<Array2<f64>>,
    shape: NetShape,
    layers: Vec<NetLayer>,
}

impl NeuralNet {
    pub fn new(act_func: ActFunc, shape: NetShape, rand_seed: u64) -> Result<NeuralNet, String> {
        let mut layers: Vec<NetLayer> = Vec::new();
        
        //Based on the provided shape of the network (# of neurons per layer), weights are generated at random with a uniform distribution betwee +-2
        let mut rng = StdRng::seed_from_u64(rand_seed);
        let range = Uniform::new(-1.0, 1.0);

        if shape.hidden_layers().len() == 0 {
            //Weights are matricies with a number of rows equal to the number of inputs + bias node, and number of columns equal to the number of outputs.
            layers.push(
                match NetLayer::new(NetLayerType::new_dense_layer_type(shape.input_node_num(), shape.output_node_num()), rand_seed) {
                    Ok(l) => l,
                    Err(e) => return Err(format!("Error in creating layer from input nodes to output: {}", e))
                }
            )
        }

        else {

            //creates weights between hidden layers
            for l in shape.layer_types().iter() {
                layers.push(
                    NetLayer::new(l, rand_seed)
                );

            }

            //creates weights between last hidden layer and output layer
            weights.push(
                Array2::from_shape_fn((shape.hidden_node_num().last().unwrap() + 1, *shape.output_node_num()), |(_, _)| range.sample(&mut rng))
            );
        }

        
        
        Ok(NeuralNet {
            act_func,
            layer_output_cache: vec![],
            shape,
            weights
        })
    }

    pub fn train(&mut self, training_data: TData, testing_data: Option<TData>, settings: &TSettings) -> Result<(), String> {
        
        //ensures that the matricies for the training_data.input()s and outputs are correct
        if training_data.input()[0].shape()[1] != *self.shape.input_node_num() {
            return Err(format!("1st training_data.input() matrix doesn't have the expected shape. {:?} != {:?}", training_data.input()[0].shape()[1], self.shape.input_node_num()))
        }

        if training_data.output()[0].shape()[1] != *self.shape.output_node_num() {
            return Err(format!("1st output matrix doesn't have the expected shape. {:?} != {:?}", training_data.output()[0].shape()[1], self.shape.output_node_num()))
        }
        
        let mut start = Instant::now();

        for iter_num in 0..(settings.iterations() + 1) {
            let training_error = match self.train_epoch(&training_data, &settings) {
                Ok(n) => n,
                Err(e) => return Err(e)
            };

            if settings.print_frequency() != 0 {
                if iter_num % (settings.iterations() / settings.print_frequency()) == 0 || iter_num == settings.iterations() {
                    match testing_data {
                        Some(_) => {
                            let mut testing_error = 0.0;
                            let data = testing_data.as_ref().unwrap();
                            for i in 0..data.input().len() {
                                let test_output = self.test(data.input()[i].clone()).unwrap();
                                let test_error = data.output()[i].clone() - test_output;
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

    pub fn test(&mut self, input: Array2<f64>) -> Result<&Array2<f64>, String> {
        match self.forward_propogate(input) {
            Ok(_) => (),
            Err(e) => return Err(e)
        }

        Ok(self.cached_output())
    }

    fn forward_propogate(&mut self, input: Array2<f64>) -> Result<(), String> {

        self.layer_output_cache = Vec::new();
        self.layer_output_cache.push(input);

        for layer in self.weights.iter() {

            //extending input to include a bias node
            let mut input: Array2<f64> = self.layer_output_cache.last().unwrap().clone();
            input.push_column(ArrayView::from(&[1.0])).unwrap();

            self.layer_output_cache.push(
                self.act_func.apply( input.dot(layer))
            )
        }

        Ok(())
    }

    fn backward_propogate(&mut self, mut layer_error: Array2<f64>,dropout: bool) -> Result<Vec<Array2<f64>>, String> {
        //layer_error = calculated output - expected output (after forwrd propogation)
        //dropout randomly turns off half the nodes on back propogation as to not overfit network to data

        //ensures that forward propogation was done nd thus that there are layer outputs
        if self.weights.len() != self.layer_output_cache.len() - 1 {
            return Err("Run forward_propogate before backward.".into())
        }


        //Vector representing the error in the weights between layers
        //This vector is reversed. That is, the last set of weights in the network (which connects to the output layer) is first in this list
        let mut weight_deltas: Vec<Array2<f64>> = Vec::new();

        //shows the input into the weights, or rather the output of the lyaer immediately before the weights
        let mut weight_input: Array2<f64>;

        //used for dropout
        let mut rng = StdRng::seed_from_u64(0);
        let range = Bernoulli::new(0.5).unwrap();

        //backpropogates from the last set of weights
        for i in (0..(self.weights.len())).rev() {

            //accointing for bias node
            weight_input = self.layer_output_cache[i].clone();
            weight_input.push_column(ArrayView::from(&[1.0])).unwrap();

            //dropout vector essentially randomly turns off half of the nodes to intorduce noise and avoid overfitting. thus multiply by 2
            if dropout {
                let dropout_vector = Array2::from_shape_fn((weight_input.shape()[0], weight_input.shape()[1]) , |(_,_)| range.sample(&mut rng) as i8 as f64);
                weight_input = weight_input * dropout_vector * 2.0;
            }

            //calculates the error for each layer depending on the function
            let (new_weight_delta, new_layer_error) = self.act_func.delta(layer_error, weight_input, &self.weights[i]);
            
            weight_deltas.push(new_weight_delta);

            //saves this iteration of the layer error, to be used in the following caluclation
            layer_error = new_layer_error;

            //removes error associated with bias node as the bias node doesn't backpropogate, that is: it has nothing connecitng to it
            let last_row_index = layer_error.shape()[0] - 1;
            layer_error = layer_error.slice(s![0..last_row_index, 0..]).to_owned();
        }

        
        Ok(weight_deltas)

    }

    fn train_epoch(&mut self, data: &TData, settings: &TSettings) -> Result<f64, String>  {
        let mut current_error = 0.0;
        for i in 0..data.input().len() {

            self.forward_propogate(data.input()[i].clone()).unwrap();
            let layer_delta = (self.cached_output() - data.output()[i].clone()).reversed_axes();

            current_error += (layer_delta.clone() * &layer_delta).sum();

            //println!("\n{:?}\n", nn);
            let mut weight_deltas = self.backward_propogate(layer_delta, settings.dropout()).unwrap();
            for i in 0..self.weights.len() {
                //-settings alpha so that this scaled dd turns into a scaled minus
                self.weights[i].scaled_add(-settings.alpha(), &weight_deltas.pop().unwrap());
            }
        } 

        Ok(current_error / data.input().len() as f64)
    }

    pub fn cached_output(&self) -> &Array2<f64> {
        self.layer_output_cache.last().unwrap()
    }

    pub fn get_shape(&self) -> &NetShape {
        &self.shape
    }

    pub fn get_weights(&self) -> &Vec<Array2<f64>> {
        &self.weights
    }
}

impl Debug for NeuralNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Neural Net: \nActivation Function: {:?}\nShape: {:?}\nOutput Cache: {:#?}\nWeights {:?}", self.act_func, self.shape, self.layer_output_cache.last().unwrap(), self.weights)
    }
}