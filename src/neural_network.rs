pub mod act_func;
pub mod mlp;
pub mod cnn;

use act_func::ActFunc;
use ndarray::Array2;


pub trait NeuralNet: Sized {
    fn new(act_func: ActFunc, shape: Vec<usize>, rand_seed: u64) -> Result<Self, String>;

    fn train(&mut self, training_data: TData, testing_data: Option<TData>, settings: &TSettings) -> Result<(), String>;

    fn test(&mut self, input: Array2<f64>) -> Result<&Array2<f64>, String>;
}


#[derive( Clone)]
pub struct TData {
    input: Vec<Array2<f64>>,
    output: Vec<Array2<f64>>
}

impl TData {
    pub fn new(input: Vec<Array2<f64>>, output: Vec<Array2<f64>>) -> Result<TData, String> {
        //ensures that the matricies for the inputs and outputs are correct
        for i in 0..input.len() {
            if input[i].shape() != input[0].shape() {
                return Err(format!("{}th input matrix doesn't have the expected shape. {:?} != {:?}", i, input[i].shape(), input[0].shape()))
            }
        }

        for i in 0..output.len() {
            if output[i].shape() != output[0].shape() {
                return Err(format!("{}th output matrix doesn't have the expected shape. {:?} != {:?}", i, output[i].shape(), output[0].shape()))
            }
        }

        Ok(TData {
            input,
            output
        })
    }
}

#[derive(Copy, Clone)]
//training settings
pub struct TSettings{
    iterations: usize,
    alpha: f64,
    dropout: bool,
    print_frequency: usize
}

impl TSettings {
    pub fn new(iterations: usize, alpha: f64, dropout: bool, print_frequency: usize) -> Result<TSettings, String> {
        if print_frequency > iterations {
            Err("Print frequency cannot be greater than #iterations".into())
        }
        else if alpha < 1.0 && alpha > 0.0 {
            Ok(TSettings {
                iterations,
                alpha,
                dropout,
                print_frequency
            })
        }
        else {
            Err("Alpha must be between 1 and 0".into())
        }
    }
}