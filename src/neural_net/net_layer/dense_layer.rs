use super::{NetLayer, NetLayerType};
use super::super::act_func::ActFunc;

use ndarray::Array2;
use rand::{distributions::{Distribution, Uniform, Bernoulli}, rngs::StdRng, SeedableRng};


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

