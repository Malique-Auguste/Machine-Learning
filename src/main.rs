mod neural_net;
mod act_func;
mod dataset_handler;

use neural_net::*;
use act_func::*;
use nalgebra::DMatrix;
use dataset_handler::read_mnist;

fn main() {
    let mut nn = NeuralNet::new(ActFunc::Sigmoid, vec![784, 40, 10], 2).unwrap();
    
    let (mut input , mut expected_output) = read_mnist("mnist/mnist_train.csv".into()).unwrap();
    let input: Vec<DMatrix<f64>> = input.drain(0..1000).collect();
    let expected_output: Vec<DMatrix<f64>> = expected_output.drain(0..1000).collect();


    let tsettings = TSettings::new(30, 0.005, false, 10).unwrap();

    nn.train(input, expected_output, &tsettings).unwrap();

    let (mut test_input, mut test_output) = read_mnist("mnist/mnist_test.csv".into()).unwrap();
    let test_input: Vec<DMatrix<f64>> = test_input.drain(0..1000).collect();
    let test_output: Vec<DMatrix<f64>> = test_output.drain(0..1000).collect();

    let mut sum_error = 0.0;
    for i in 0..test_input.len() {
        let output = nn.test(test_input[i].clone()).unwrap();
        let error = (output - test_output[i].clone()).sum();

        sum_error += error * error;
    }

    println!("\nTest error: {}", sum_error / test_output.len() as f64)
}
