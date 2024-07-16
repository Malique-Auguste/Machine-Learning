use std::fmt::Debug;

use super::act_func::ActFunc;
use ndarray::{s, Array2, Array3, ArrayView, Axis, linalg::kron};
use ndarray_stats::QuantileExt;
use rand::{distributions::{Distribution, Uniform, Bernoulli}, rngs::StdRng, SeedableRng};

pub struct NetLayer {
    layer_type: NetLayerType,
    weights: Array2<f64>,
    output: Array2<f64>,
}

impl NetLayer {
    pub fn new(layer_type: NetLayerType, rng: &mut StdRng, range: &Uniform<f64>) -> Result<NetLayer, String> {
        match layer_type {
            NetLayerType::DenseLayer{input_node_num, output_node_num, ..} => {
                if input_node_num == 0 {
                    return Err(format!("Input node num cannot equal zero"))
                }
                else if output_node_num == 0 {
                    return Err(format!("Output node num cannot equal zero"))
                }
        
                Ok(NetLayer {
                    layer_type,
                    //+1 to account for bias node
                    weights: Array2::from_shape_fn((input_node_num + 1, output_node_num), |(_,_)| range.sample(rng)),
                    output: Array2::from_elem((1, output_node_num), 0.0)
                })
            },

            NetLayerType::PrimaryConvolutionalLayer { input_width, kernel_num, kernel_width, pool_step , ..} => {
                if kernel_width >= input_width as usize {
                    return  Err(format!("Kernal width ({}) must be less than input_width ({}).", kernel_width, input_width ));
                }

                let feature_map_width = input_width + 1 - kernel_width;
                let pooled_feature_map_width: usize = feature_map_width / pool_step;
        
                Ok(NetLayer {
                    layer_type,
                    //Weights has number of rows equal to area of kernel and number of columns equal to number of kernels
                    weights: Array2::from_shape_fn((kernel_width.pow(2), kernel_num), |(_,_)| range.sample(rng)),
                    output: Array2::from_elem((1, pooled_feature_map_width.pow(2) * kernel_num), 0.0)
                })
            },

            NetLayerType::SecondaryConvolutionalLayer { input_feature_map_width, input_feature_map_num, kernel_num, kernel_width, pool_step, ..} => {
                let feature_map_width = input_feature_map_width + 1 - kernel_width;
                let pooled_feature_map_width: usize = feature_map_width / pool_step;

                Ok(NetLayer {
                    layer_type,

                    weights: Array2::from_shape_fn((input_feature_map_num * kernel_width.pow(2), kernel_num), |(_,_)| range.sample(rng)),
                    output: Array2::from_elem((1, pooled_feature_map_width.pow(2) * input_feature_map_num * kernel_num), 0.0)
                })
            }
        }
    }

    pub fn layer_type(&self) -> &NetLayerType {
        &self.layer_type
    }

    pub fn forward_propogate(&mut self, mut input: Array2<f64>) {
        match self.layer_type {
            NetLayerType::DenseLayer {act_func, ..} => {
                input.push_column(ArrayView::from(&[1.0])).unwrap();
                self.output = act_func.apply(input.dot(&self.weights))
            },

            NetLayerType::PrimaryConvolutionalLayer { kernel_width, input_width, kernel_num, pool_step, act_func, .. } => {
                //converts flattened input into a square which kernels can traverse
                input = input.into_shape([input_width, input_width]).unwrap();

                //featurea map width is equal to the number of times a kernel will move right plus one
                let feature_map_width = input_width + 1 - kernel_width;

                //shape follows convention of [depth, rows, columns]
                //therefore temp output is such that each layer reprent a new feature map
                let mut kernel_input: Array2<f64> = Array2::from_elem((1, kernel_width * kernel_width), 0.0);

                //row
                for r in 0..feature_map_width {

                    //column
                    for c in 0..feature_map_width {
                        let square_kernel_input_segment = input.slice(s![r..(r + kernel_width), c..(c + kernel_width)]).to_owned();
                        let flat_kernel_input_segment = square_kernel_input_segment.into_shape([1, kernel_width * kernel_width]).unwrap();

                        if r == 0 && c == 0 {
                            kernel_input = flat_kernel_input_segment
                        }
                        else {
                            kernel_input.append(Axis(0), flat_kernel_input_segment.view()).unwrap()
                        }
                    }
                }

                let output: Array2<f64> = act_func.apply(kernel_input.dot(&self.weights)).reversed_axes();
                //self.output = output.reversed_axes().into_shape((1, feature_map_width * feature_map_width * num_of_kernels)).unwrap();


                let pooled_feature_map_width: usize = feature_map_width / pool_step;

                //each column is a new feature map.
                let cubed_output = output.into_shape((kernel_num, feature_map_width, feature_map_width)).unwrap();
                let mut vec_output = Vec::new();

                for k in 0..kernel_num {
                    for r in (0..feature_map_width).step_by(pool_step) {
                        for c in (0..feature_map_width).step_by(pool_step)  {
                            let sub_matrix = cubed_output.slice(s![k,
                                                                                                        r..(r + pool_step), 
                                                                                                        c..(c + pool_step)]);
                
                            vec_output.push(sub_matrix.max().unwrap().clone());
                            
                        }
                    }     
                }   

                let output = Array2::from_shape_vec((1, pooled_feature_map_width.pow(2) * kernel_num), vec_output).unwrap();
                self.output = output;
            },

            NetLayerType::SecondaryConvolutionalLayer { input_feature_map_width, input_feature_map_num, kernel_num, kernel_width, pool_step, act_func, .. } => {
                //unimplemented!();
                
                
                //converts flattened input into a square which kernels can traverse
                let input: Array3<f64> = input.into_shape((input_feature_map_num, input_feature_map_width, input_feature_map_width)).unwrap();

                //featurea map width is equal to the number of times a kernel will move right plus one
                let feature_map_width = input_feature_map_width + 1 - kernel_width;

                //shape follows convention of [depth, rows, columns]
                //therefore temp output is such that each layer reprent a new feature map
                let mut kernel_input: Array3<f64> = Array3::from_elem((input_feature_map_num, kernel_width, kernel_width), 0.0);

                
                //row
                for r in 0..feature_map_width {

                    //column
                    for c in 0..feature_map_width {
                        let square_kernel_input_segment = input.slice(s![.., r..(r + kernel_width), c..(c + kernel_width)]).to_owned();
                        let flat_kernel_input_segment = square_kernel_input_segment.into_shape([input_feature_map_num, 1, kernel_width * kernel_width]).unwrap();

                        if r == 0 && c == 0 {
                            kernel_input = flat_kernel_input_segment
                        }
                        else {
                            kernel_input.append(Axis(0), flat_kernel_input_segment.view()).unwrap()
                        }

                    }
                }

                let mut output: Array3<f64> = Array3::from_elem((input_feature_map_num, kernel_num, feature_map_width.pow(2),), 0.0);

                for f in 0..input_feature_map_num {
                    let temp_output = act_func.apply(kernel_input.slice(s![f, .., ..]).to_owned().dot(&self.weights)).reversed_axes();

                    output.index_axis_mut(Axis(2), f).assign(&temp_output)
                }

                //self.output = output.into_shape((1, feature_map_width.pow(2) * kernel_num * input_feature_map_num)).unwrap();

                
                let pooled_feature_map_width: usize = feature_map_width / pool_step;

                //each column is a new feature map.
                let output_4d = output.into_shape((input_feature_map_num, kernel_num, feature_map_width, feature_map_width)).unwrap();
                let mut vec_output: Vec<f64> = Vec::new();

                
                for f in 0..input_feature_map_num {
                    for k in 0..kernel_num {
                        for r in (0..feature_map_width).step_by(pool_step) {
                            for c in (0..feature_map_width).step_by(pool_step)  {
                                let sub_matrix = output_4d.slice(s![f, k,
                                                                                                        r..(r + pool_step), 
                                                                                                        c..(c + pool_step),]);
                                
                                vec_output.push(sub_matrix.max().unwrap().clone());
                            }
                        }        
                    }
                }
                
                let output = Array2::from_shape_vec((1, pooled_feature_map_width.pow(2) * kernel_num * input_feature_map_num), vec_output).unwrap();
                self.output = output;
            }
        }
    }

    pub fn back_propogate(&mut self, alpha: f64, dropout: bool, mut input: Array2<f64>, mut layer_error: Array2<f64>) -> Array2<f64> {
        match self.layer_type {
            NetLayerType::DenseLayer {act_func, ..} => {
                //Array representing the error in the weights
                let weight_deltas: Array2<f64>;

                input.push_column(ArrayView::from(&[1.0])).unwrap();

                //dropout vector essentially randomly turns off half of the nodes to intorduce noise and avoid overfitting. thus multiply by 2
                if dropout {
                    //used for dropout
                    let mut rng = StdRng::seed_from_u64(0);
                    let range = Bernoulli::new(0.5).unwrap();

                    let dropout_vector = Array2::from_shape_fn((input.shape()[0], input.shape()[1]) , |(_,_)| range.sample(&mut rng) as i8 as f64);
                    input = input * dropout_vector * 2.0;
                }

                (weight_deltas, layer_error) = {
                    let new_weight_delta = input.reversed_axes().dot(&(layer_error.t().to_owned() * act_func.deriv(&self.output)));
                    let new_layer_error = self.weights.dot(&layer_error);

                    (new_weight_delta, new_layer_error)
                    
                };

                

                //"-alpha" so that this scaled add turns into a scaled minus
                self.weights.scaled_add(-alpha, &weight_deltas);
                
                //removes error associated with bias node as the bias node doesn't backpropogate, that is: it has nothing connecitng to it
                let last_row_index = layer_error.shape()[0] - 1;
                layer_error = layer_error.slice(s![0..last_row_index, 0..]).to_owned();

                //layer_error of input layer
                layer_error
            },

            NetLayerType::PrimaryConvolutionalLayer { input_width, kernel_num, kernel_width , pool_step, act_func, ..} => {
                let act_func_deriv = act_func.deriv(&self.output);
                let layer_error = layer_error * act_func_deriv.t();
                
                let square_input: Array2<f64> = input.into_shape((input_width, input_width)).unwrap();
                
                let feature_map_width = input_width + 1 - kernel_width;
                let pooled_feature_map_width: usize = feature_map_width / pool_step;
                let cubed_layer_error: Array3<f64> = layer_error.into_shape((kernel_num, pooled_feature_map_width, pooled_feature_map_width)).unwrap();
                let mut new_cubed_layer_error: Array3<f64> = Array3::from_elem((kernel_num, feature_map_width, feature_map_width), 0.0);

                let ones = Array2::from_elem((pool_step, pool_step), 1.0);
                for k in 0..kernel_num {
                    let feature_map = cubed_layer_error.slice(s![k, .., ..]);
                    new_cubed_layer_error.index_axis_mut(Axis(0), k).assign(&kron(&feature_map, &ones));
                }

                let square_layer_error = new_cubed_layer_error.reversed_axes().into_shape((feature_map_width * feature_map_width, kernel_num)).unwrap();

                let mut input_collection: Array2<f64> = Array2::from_elem((1, kernel_width * kernel_width), 0.0);

                //row
                for r in 0..kernel_width {

                    //column
                    for c in 0..kernel_width {
                        let square_input_segment = square_input.slice(s![r..(r + feature_map_width), c..(c + feature_map_width)]).to_owned();
                        let flat_input_segment = square_input_segment.into_shape([1, feature_map_width * feature_map_width]).unwrap();

                        if r == 0 && c == 0 {
                            input_collection = flat_input_segment
                        }
                        else {
                            input_collection.append(Axis(0), flat_input_segment.view()).unwrap()
                        }
                    }
                }

                let weight_delta = input_collection.dot(&square_layer_error);

                //"-alpha" so that this scaled add turns into a scaled minus
                self.weights.scaled_add(-alpha.powi(2), &weight_delta);

                let mut input_layer_error = Array2::from_elem((input_width, input_width), 0.0);

                
                //Turned off calulating layer errors for Conv Layer
                //Note: when this code was implemented, there was no pooling of layers


                let cubed_layer_error: Array3<f64> = square_layer_error.into_shape((kernel_num, feature_map_width, feature_map_width)).unwrap();

                for k in 0..kernel_num {
                    let feature_map_error = cubed_layer_error.slice(s![k, .., ..]).to_owned();
                    let mut weight_index = 0;

                    for r in 0..kernel_width {
                        for c in 0..kernel_width {
                            let partial_input_error = &feature_map_error * self.weights[(weight_index, k)];

                            for f_row in 0..feature_map_width {
                                for f_col in 0..feature_map_width {
                                    input_layer_error[(r + f_row, c + f_col)] += partial_input_error[(f_row, f_col)];
                                }
                            }

                            weight_index += 1;
                        }
                    }
                }               
                

                input_layer_error.into_shape((1, input_width * input_width)).unwrap()
            },

            NetLayerType::SecondaryConvolutionalLayer { ..} => {
                unimplemented!()
            }
        }
    }

    pub fn output(&self) -> &Array2<f64> {
        &self.output
    }
}

impl Debug for NetLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer Type: {:?}\nWeights {:?}", self.layer_type, self.weights)
    }
}


#[derive(Debug)]
pub enum NetLayerType {
    DenseLayer{
        input_node_num: usize,
        output_node_num: usize,
        act_func: ActFunc,
    },
    PrimaryConvolutionalLayer {
        input_width: usize,
        kernel_num: usize,
        kernel_width: usize,
        pool_step: usize,
        output_node_num: usize,
        act_func: ActFunc,
    },
    SecondaryConvolutionalLayer {
        input_feature_map_width: usize,
        input_feature_map_num: usize,
        kernel_num: usize,
        kernel_width: usize,
        pool_step: usize,
        output_node_num: usize,
        act_func: ActFunc,
    }
}

impl NetLayerType {
    pub fn input_node_num(&self) -> usize {
        match self {
            NetLayerType::DenseLayer { input_node_num, .. } => {
                input_node_num.clone()
            },
            NetLayerType::PrimaryConvolutionalLayer { input_width, .. } => {
                input_width * input_width
            },
            NetLayerType::SecondaryConvolutionalLayer { ..} => {
                unimplemented!()
            }
        }
    }

    pub fn output_node_num(&self) -> usize {
        match self {
            NetLayerType::DenseLayer { output_node_num, .. } => {
                output_node_num.clone()
            }
            NetLayerType::PrimaryConvolutionalLayer { output_node_num, .. } => {
                output_node_num.clone()
            },
            NetLayerType::SecondaryConvolutionalLayer { output_node_num , ..} => {
                output_node_num.clone()
            }
        }
    }
}