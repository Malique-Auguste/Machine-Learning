use super::act_func::ActFunc;
use ndarray::{Array2, ArrayView};
use rand::{distributions::{Distribution, Uniform, Bernoulli}, rngs::StdRng, SeedableRng};

pub struct NetLayer {
    layer_type: NetLayerType,
    weights: Array2<f64>,
    output: Array2<f64>
}

impl NetLayer {
    pub fn new(layer_type: NetLayerType, rand_seed: u64) -> Result<NetLayer, String> {
        match layer_type {
            NetLayerType::DenseLayer{input_node_num, output_node_num} => {
                if input_node_num == 0 {
                    return Err(format!("Input node num cannot equal zero"))
                }
                else if output_node_num == 0 {
                    return Err(format!("Output node num cannot equal zero"))
                }
        
                let mut rng = StdRng::seed_from_u64(rand_seed);
                let range = Uniform::new(-1.0, 1.0);
        
                Ok(NetLayer {
                    layer_type,
                    //+1 to account for bias node
                    weights: Array2::from_shape_fn((input_node_num + 1, output_node_num), |(_,_)| range.sample(&mut rng)),
                    output: Array2::from_elem((1, output_node_num), 0.0)
                })
            },

            NetLayerType::ConvolutionalLayer { input_width, num_of_kernels, kernel_width , output_node_num} => {
                if kernel_width >= input_width as usize {
                    return  Err(format!("Kernal width ({}) must be less than input_width ({}).", kernel_width, input_width ));
                }
        
                let flattedned_output_length = (input_width + 1 - kernel_width) * (input_width + 1 - kernel_width);
                let mut rng = StdRng::seed_from_u64(rand_seed);
                let range = Uniform::new(-1.0, 1.0);
        
                Ok(NetLayer {
                    layer_type,
                    //Weights has number of rows equal to area of kernel and number of columns equal to number of kernels
                    weights: Array2::from_shape_fn((kernel_width * kernel_width, num_of_kernels), |(_,_)| range.sample(&mut rng)),
                    output: Array2::from_elem((1, flattedned_output_length), 0.0)
                })
            }
        }
    }

    fn forward_propogate(&mut self, act_func: &ActFunc, input: &Array2<f64>) -> &Array2<f64> {
        match self.layer_type {
            NetLayerType::DenseLayer {..} => {
                self.output = act_func.apply(input.dot(&self.weights));
                &self.output
            },

            NetLayerType::ConvolutionalLayer { input_width, num_of_kernels, kernel_width , output_node_num} => {
                unimplemented!()
            }
        }
    }
    fn back_propogate(&mut self, act_func: &ActFunc, alpha: f64, dropout: bool, mut input: Array2<f64>, mut layer_error: Array2<f64>) -> Array2<f64> {
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

                //layer_error of input layer
                layer_error
            },

            NetLayerType::ConvolutionalLayer { input_width, num_of_kernels, kernel_width , output_node_num} => {
                unimplemented!()
            }
        }
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
    pub fn new_dense_layer_type(input_node_num: usize, output_node_num: usize ) -> NetLayerType {
        NetLayerType::DenseLayer { input_node_num, output_node_num }
    }

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