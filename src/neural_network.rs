pub mod act_func;
pub mod mlp;
pub mod cnn;

use act_func::ActFunc;
use ndarray::Array2;


pub trait NeuralNet: Sized {
    fn new(act_func: ActFunc, shape: NetShape, rand_seed: u64) -> Result<Self, String>;

    fn train(&mut self, training_data: TData, testing_data: Option<TData>, settings: &TSettings) -> Result<(), String>;

    fn test(&mut self, input: Array2<f64>) -> Result<&Array2<f64>, String>;
}

//Network Shape
#[derive(Debug)]
pub struct NetShape {
    input_node_num: usize,
    hidden_node_num: Vec<usize>,
    output_node_num: usize,
    
    kernel_size: Option<usize>,
}

impl NetShape {
    pub fn new(input_node_num: usize, hidden_node_num: Vec<usize>, output_node_num: usize, kernel_size: Option<usize>) -> Result<NetShape, String> {
        if input_node_num == 0 {
            return Err("Input node num must be greater than zero.".into());
        }
        else if output_node_num == 0 {
            return Err("Output node num must be greater than zero.".into());
        }
        
        Ok(NetShape {
            input_node_num,
            hidden_node_num,
            output_node_num,
            kernel_size,
        })
    }

    pub fn input_node_num(&self) -> &usize {
        &self.input_node_num
    }

    pub fn hidden_node_num(&self) -> &Vec<usize> {
        &self.hidden_node_num
    }

    pub fn output_node_num(&self) -> &usize {
        &self.output_node_num()
    }

    pub fn kernel_size(&self) -> &Option<usize> {
        &self.kernel_size
    }
}

//Training Data
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