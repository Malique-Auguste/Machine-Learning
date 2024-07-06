use std::fmt::Debug;

use super::act_func::ActFunc;
use ndarray::{s, Array2, Array3, ArrayView};
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
            NetLayerType::DenseLayer{input_node_num, output_node_num} => {
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

            NetLayerType::ConvolutionalLayer { input_width, num_of_kernels, kernel_width , ..} => {
                if kernel_width >= input_width as usize {
                    return  Err(format!("Kernal width ({}) must be less than input_width ({}).", kernel_width, input_width ));
                }

                let feature_map_width = input_width + 1 - kernel_width;

        
                Ok(NetLayer {
                    layer_type,
                    //Weights has number of rows equal to area of kernel and number of columns equal to number of kernels
                    weights: Array2::from_shape_fn((kernel_width * kernel_width, num_of_kernels), |(_,_)| range.sample(rng)),
                    output: Array2::from_elem((1, feature_map_width * feature_map_width), 0.0)
                })
            }
        }
    }

    pub fn layer_type(&self) -> &NetLayerType {
        &self.layer_type
    }

    pub fn forward_propogate(&mut self, act_func: &ActFunc, mut input: Array2<f64>) {
        match self.layer_type {
            NetLayerType::DenseLayer {..} => {
                input.push_column(ArrayView::from(&[1.0])).unwrap();
                self.output = act_func.apply(input.dot(&self.weights))
            },

            NetLayerType::ConvolutionalLayer { kernel_width, input_width, num_of_kernels, .. } => {
                //converts flattened input into a square which kernels can traverse
                input = input.into_shape([input_width, input_width]).unwrap();

                //featurea map width is equal to the number of times a kernel will move right plus one
                let feature_map_width = input_width + 1 - kernel_width;

                //shape follows convention of [depth, rows, columns]
                //therefore temp output is such that each layer reprent a new feature map
                let mut temp_output = Array3::from_elem([num_of_kernels, feature_map_width, feature_map_width], 0.0);

                //row
                for r in 0..feature_map_width {

                    //column
                    for c in 0..feature_map_width {
                        let square_kernel_input = input.slice(s![r..(r + kernel_width), c..(c + kernel_width)]);
                        let flat_kernel_input = square_kernel_input.into_shape([1, kernel_width * kernel_width]).unwrap();

                        //puts the feature map into the temp output. Feature maps lie on the row x column face of temp output. 
                        //Because the weights represent the output at one point for each kernel, the feature map slice is place in a single column and across the depth of tem_output
                        let feature_map_slice = act_func.apply(flat_kernel_input.dot(&self.weights));
                        let mut temp_output_slice = temp_output.slice_mut(s![.., r, c]);
                        temp_output_slice.assign(&feature_map_slice)
                    }
                }

                /*
                let pooled_feature_map_width = (feature_map_width / 2) + (feature_map_width % 2);
                let mut pooled_temp_output = Array3::from_elem([num_of_kernels, pooled_feature_map_width, pooled_feature_map_width], 0.0);
                let step: usize = 2;

                //for each 2x2 square of data in a feature map int he output, the max value is saved
                for d in 0..num_of_kernels {

                    let mut pooled_r_index = 0;
                    for r_index in (0..feature_map_width).step_by(step) {

                        // if row range is ever greater than the feature map width, it is limited
                        let mut row_range = r_index..(r_index + step);
                        if r_index + step > feature_map_width {
                            row_range = r_index..feature_map_width
                        }

                        let mut pooled_c_index = 0;
                        for c_index in (0..feature_map_width).step_by(step) {
                            let mut column_range = c_index..(c_index + step);
                            if c_index + step > feature_map_width {
                                column_range = c_index..feature_map_width
                            }

                            pooled_temp_output[(d, pooled_r_index, pooled_c_index)] = *temp_output.slice(s![d, row_range.clone(), column_range.clone()]).max().unwrap();
                            
                            pooled_c_index += 1;
                        }

                        pooled_r_index += 1;
                    }
                }

                let flattened_pooled_output = pooled_temp_output.into_shape((1, num_of_kernels * pooled_feature_map_width * pooled_feature_map_width)).unwrap();
                flattened_pooled_output
                */

                let flattened_output: Array2<f64> = temp_output.into_shape((1, num_of_kernels * feature_map_width * feature_map_width)).unwrap();

                self.output = flattened_output
            }
        }
    }

    pub fn back_propogate(&mut self, act_func: &ActFunc, alpha: f64, dropout: bool, mut input: Array2<f64>, mut layer_error: Array2<f64>) -> Array2<f64> {
        match self.layer_type {
            NetLayerType::DenseLayer {..} => {
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

                (weight_deltas, layer_error) = act_func.delta(layer_error, input, &self.weights);

                //"-alpha" so that this scaled add turns into a scaled minus
                self.weights.scaled_add(-alpha, &weight_deltas);
                
                //removes error associated with bias node as the bias node doesn't backpropogate, that is: it has nothing connecitng to it
                let last_row_index = layer_error.shape()[0] - 1;
                layer_error = layer_error.slice(s![0..last_row_index, 0..]).to_owned();

                //layer_error of input layer
                layer_error
            },

            NetLayerType::ConvolutionalLayer { input_width, num_of_kernels, kernel_width , output_node_num} => {
                let square_input: Array2<f64> = self.output.clone().into_shape((input_width, input_width)).unwrap();
                
                let feature_map_width = input_width + 1 - kernel_width;
                let cubed_layer_error: Array3<f64> = layer_error.into_shape((num_of_kernels, feature_map_width, feature_map_width)).unwrap();

                for k in 0..num_of_kernels {
                    let mut weight_index = 0;

                    for r in 0..kernel_width {
                        for c in 0..kernel_width {
                            //Gets input equal in dimensions to the output error, and multipies them.
                            let square_input_slice = square_input.slice(s![r..(r + feature_map_width), c..(c + feature_map_width)]);
                            let flattened_input_slice = square_input_slice.into_shape((1, feature_map_width * feature_map_width)).unwrap();

                            let feature_map_error = cubed_layer_error.slice(s![k, .., ..]);
                            let flattened_feature_map_error = feature_map_error.into_shape((feature_map_width * feature_map_width, 1)).unwrap();

                            self.weights[(weight_index, k)] += flattened_input_slice.dot(&flattened_feature_map_error).sum() * alpha;

                            weight_index += 1;
                        }
                    }
                }

                let mut input_layer_error = Array2::from_elem((input_width, input_width), 0.0);

                for k in 0..num_of_kernels {
                    let mut weight_index = 0;

                    for r in 0..kernel_width {
                        for c in 0..kernel_width {
                            let feature_map_error = cubed_layer_error.slice(s![k, .., ..]);
                            let partial_input_error = feature_map_error.to_owned() * self.weights[(weight_index, k)];

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
        output_node_num: usize
    },
    ConvolutionalLayer {
        input_width: usize,
        num_of_kernels: usize,
        kernel_width: usize,
        output_node_num: usize,
    }
}

impl NetLayerType {
    pub fn input_node_num(&self) -> usize {
        match self {
            NetLayerType::ConvolutionalLayer { input_width, .. } => {
                input_width * input_width
            },
            NetLayerType::DenseLayer { input_node_num, .. } => {
                input_node_num.clone()
            }
        }
    }

    pub fn output_node_num(&self) -> usize {
        match self {
            NetLayerType::ConvolutionalLayer { output_node_num, .. } => {
                output_node_num.clone()
            },
            NetLayerType::DenseLayer { output_node_num, .. } => {
                output_node_num.clone()
            }
        }
    }
}