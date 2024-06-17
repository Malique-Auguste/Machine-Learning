use std::fmt::format;

use ndarray::{Array2, s, ArrayView};
use rand::{distributions::{Distribution, Uniform, Bernoulli}, rngs::StdRng, SeedableRng};

use super::act_func::{self, ActFunc};


pub trait NetLayer {
    fn forward_propogate(&mut self, act_func: &ActFunc, input: &Array2<f64>) -> &Array2<f64>;
    fn back_propogate(&mut self, act_func: &ActFunc, alpha: f64, dropout: bool, input: Array2<f64>, layer_error: Array2<f64>) -> Array2<f64>;
}

pub struct DenseLayer {
    weights: Array2<f64>,
    output: Array2<f64>
}

impl DenseLayer {
    pub fn new(input_node_num: usize, output_node_num: usize, rand_seed: u64) -> Result<DenseLayer, String> {
        if input_node_num == 0 {
            return Err(format!("Input node num cannot equal zero"))
        }
        else if output_node_num == 0 {
            return Err(format!("Output node num cannot equal zero"))
        }

        let mut rng = StdRng::seed_from_u64(rand_seed);
        let range = Uniform::new(-1.0, 1.0);

        Ok(DenseLayer {
            weights: Array2::from_shape_fn((input_node_num, output_node_num), |(_,_)| range.sample(&mut rng)),
            output: Array2::from_elem((1, output_node_num), 0.0)
        })
    }
}

impl NetLayer for DenseLayer {
    fn forward_propogate(&mut self, act_func: &ActFunc, input: &Array2<f64>) -> &Array2<f64> {
        self.output = act_func.apply(input.dot(&self.weights));
        &self.output
    }

    fn back_propogate(&mut self, act_func: &ActFunc, alpha: f64, dropout: bool, mut input: Array2<f64>, mut layer_error: Array2<f64>) -> Array2<f64> {
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
    }
}

pub struct ConvolutionalLayer {
    kernel_width: usize,
    input_shape: [usize; 2],
    weights: Array2<f64>,
    output: Array2<f64>
}

impl ConvolutionalLayer {
    pub fn new(input_shape: [usize; 2], kernel_num: usize, kernel_width: usize, rand_seed: u64) -> Result<ConvolutionalLayer, String> {
        if kernel_width >= input_shape[0] {
            return Err(format!("Kenerl width ({}) cannot be greater than or equal to input width ({})", kernel_width, input_shape[0]))
        }
        else if kernel_width >= input_shape[1] {
            return Err(format!("Kenerl width ({}) cannot be greater than or equal to input height ({})", kernel_width, input_shape[1]))
        }

        let flattedned_output_length = (input_shape[0] + 1 - kernel_width) * (input_shape[1] + 1 - kernel_width);
        let mut rng = StdRng::seed_from_u64(rand_seed);
        let range = Uniform::new(-1.0, 1.0);

        Ok(ConvolutionalLayer {
            kernel_width,
            input_shape,
            //Weights has number of rows equal to area of kernel and number of columns equal to number of kernels
            weights: Array2::from_shape_fn((kernel_width * kernel_width, kernel_num), |(_,_)| range.sample(&mut rng)),
            output: Array2::from_elem((1, flattedned_output_length), 0.0)
        })
    }
}

impl NetLayer for ConvolutionalLayer {
    fn forward_propogate(&mut self, act_func: &ActFunc, input: &Array2<f64>) -> &Array2<f64> {

        //has cols equal to number of kernels, rows equal to total number of samples a kernel takes from an image.
        let mut temp_output = Array2::from_elem((self.output.shape()[1], self.weights.ncols()), 0.0);
        let mut temp_output_index: usize = 0;

        let output_column = self.input_shape[0] + 1 - self.kernel_width;
        let output_row = self.input_shape[1] + 1 - self.kernel_width;

        for r in 0..output_row {
            for c in 0..output_column {
                let square_kernel_input = input.slice(s![r..(r + self.kernel_width), c..(c + self.kernel_width)]);
                let flat_kernel_input = square_kernel_input.to_owned().into_shape((1, self.kernel_width * self.kernel_width)).unwrap();

                let output = act_func.apply(flat_kernel_input.dot(&self.weights));
                
                for kernel_num in 0..self.weights.ncols() {
                    temp_output[(temp_output_index, kernel_num)] = output[(0, kernel_num)]
                }

                temp_output_index += 1;
            }
        }

        &self.output
    }

    fn back_propogate(&mut self, act_func: &ActFunc, alpha: f64, dropout: bool, mut input: Array2<f64>, mut layer_error: Array2<f64>) -> Array2<f64> {
        unimplemented!()
    }
}