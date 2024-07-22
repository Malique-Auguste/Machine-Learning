use ndarray::Array2;
use serde::{Serialize, Deserialize};
use std::f64::consts::E;


//Activation Funciton
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ActFunc {
    ReLU,
    Sigmoid
}

impl ActFunc {
    pub fn apply(&self, mut input: Array2<f64>) -> Array2<f64> {
        match self {
            ActFunc::ReLU => {
                input.map_inplace(|elem| {
                    if *elem < 0.0 {
                        *elem = 0.0
                    }
                })
            },

            ActFunc::Sigmoid => {
                input.map_inplace(|elem| {
                    *elem = 1.0 / (1.0 + E.powf(-*elem))
                })
            }
        }

        input
    }

    pub fn deriv(&self, output: &Array2<f64>) -> Array2<f64> {
        //input must be a single row
        //weights are 2d with rows = numebr of input nodes, and columns = #output nodes

        match self {
            ActFunc::ReLU => {
                let shape = (output.shape()[0], output.shape()[1]);
                Array2::from_elem(shape, 1.0)
                /*
                //dW (layer 2) = Error (layer 3) * Input (layer 2)
                new_weight_delta = layer_error.dot(&input).t().to_owned();

                //dError (layer 2) = Weights (layer 2) * Error (layer 3)
                new_layer_error = weights.dot(&layer_error);
                */
            },

            ActFunc::Sigmoid => {
                let mut sigmoid_derivative: Array2<f64> = output.clone() * -1.0;
                sigmoid_derivative += 1.0;
                sigmoid_derivative = output * &sigmoid_derivative;

                sigmoid_derivative

                /*
                // let s = sigmoid function
                //ds = s(1-s)
                let sigmoid_result: Array2<f64> = self.apply(input.dot(weights));
                let mut sigmoid_derivative: Array2<f64> = sigmoid_result.clone() * -1.0;
                sigmoid_derivative += 1.0;
                sigmoid_derivative = sigmoid_result * &sigmoid_derivative;

                // let s = sigmoid function
                //ds = s(1-s)
                //dW (layer 2) = Input (layer 2) * Error (layer 3) *  ds (layer 3)
                //println!("le: {:?}, sd: {:?}", layer_error.shape(), sigmoid_derivative.shape());
                new_weight_delta = input.reversed_axes().dot(&(layer_error.t().to_owned() * &sigmoid_derivative));
                
                new_layer_error = weights.dot(&layer_error);//.component_mul(&sigmoid_derivative.transpose());
                */
            }
            
        }

    }
}