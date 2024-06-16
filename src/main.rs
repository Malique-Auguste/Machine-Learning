use machine_learning::neural_network::{*, act_func::*, mlp::*};
use machine_learning::dataset_handler::*;

use ndarray::Array2;

fn main() {
    println!("Running...");

    let mut nn = MLP::new(ActFunc::Sigmoid, vec![784, 40, 10], 2).unwrap();
    
    let (mut input , mut expected_output) = read_mnist(".git/lfs/objects/fb/60/fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d".into()).unwrap();
    let input: Vec<Array2<f64>> = input.drain(0..1000).collect();
    let expected_output: Vec<Array2<f64>> = expected_output.drain(0..1000).collect();
    let training_data = TData::new(input, expected_output).unwrap();


    let (mut test_input, mut test_output) = read_mnist(".git/lfs/objects/51/c2/51c292478d94ec3a01461bdfa82eb0885d262eb09e615679b2d69dedb6ad09e7".into()).unwrap();
    let test_input: Vec<Array2<f64>> = test_input.drain(0..100).collect();
    let test_output: Vec<Array2<f64>> = test_output.drain(0..100).collect();
    let testing_data = TData::new(test_input.clone(), test_output.clone()).unwrap();


    let tsettings = TSettings::new(100, 0.005, false, 15).unwrap();
    nn.train(training_data, Some(testing_data), &tsettings).unwrap();

    

    for i in (0..test_input.len()).step_by(test_input.len() / 3) {
        let output = nn.test(test_input[i].clone()).unwrap();
        let error = output - test_output[i].clone();
        
        let error = (&error * &error).sum();

        println!("\nExpected: {:?}\nCalculated: {:?}\nError: {}", test_output[i], output, error);
        
    }
}
