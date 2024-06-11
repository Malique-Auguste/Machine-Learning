use nalgebra::{DMatrix, Vector};
use std::{alloc::Layout, f64::consts::E, fmt::DebugList, process::Output};


//Activation Funciton
#[derive(Debug)]
pub enum ActFunc {
    ReLU,
    Sigmoid
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
            },

            ActFunc::Sigmoid => {
                for elem in input.iter_mut() {
                    *elem = 1.0 / (1.0 + E.powf(-*elem))
                }
            }
        }

        input
    }

    pub fn delta(&self, layer_error: DMatrix<f64>, input: DMatrix<f64>, weights: DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        //input must be a single row
        //weights are 2d with rows = numebr of input nodes, and columns = #output nodes

        let new_weight_delta: DMatrix<f64>;
        let new_layer_error: DMatrix<f64>;

        match self {
            ActFunc::ReLU => {
                //dW (layer 2) = Error (layer 3) * Input (layer 2)
                new_weight_delta = (layer_error.clone() * input).transpose();

                //dError (layer 2) = Weights (layer 2) * Error (layer 3)
                new_layer_error = weights * layer_error;
            },

            ActFunc::Sigmoid => {
                // let s = sigmoid function
                //ds = s(1-s)
                let sigmoid_result: DMatrix<f64> = self.apply(input.clone() * weights.clone());
                let mut sigmoid_derivative: DMatrix<f64> = sigmoid_result.clone() * -1.0;
                sigmoid_derivative = sigmoid_result.component_mul(&sigmoid_derivative.add_scalar(1.0));

                // let s = sigmoid function
                //ds = s(1-s)
                //dW (layer 2) = Input (layer 2) * Error (layer 3) *  ds (layer 3)
                //println!("le: {:?}, sd: {:?}", layer_error.shape(), sigmoid_derivative.shape());
                new_weight_delta = input.transpose() * layer_error.transpose().component_mul(&sigmoid_derivative);
                
                new_layer_error = weights * layer_error;//.component_mul(&sigmoid_derivative.transpose());
            }
            
        }

        (new_weight_delta, new_layer_error)

    }
}

pub fn generate_diagonal_matrix(input: DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    if input.shape().0 > 1 {
        return Err("Matrix must only have 1 row".into())
    }

    let num_columns = input.shape().1;

    let mut output = DMatrix::zeros(num_columns, num_columns);

    for i in 0..num_columns {
        output[(i,i)] = input[(0, i)]
    }

    Ok(output)
}