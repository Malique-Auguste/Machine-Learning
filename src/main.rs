use ml::neural_net::{*, act_func::*, training_helpers::*, net_shape::*, net_layer::*};
use ml::dataset_handler::*;

use ndarray::Array2;

fn main() {
    println!("Running...");

    let net_shape = NetShape::new(vec![NetLayerType::PrimaryConvolutionalLayer { input_width: 28, kernel_num: 4, kernel_width: 3, pool_step: 1, output_node_num: 2704, act_func: ActFunc::ReLU }, 
                                                                NetLayerType::DenseLayer { input_node_num: 2704, output_node_num: 10, act_func: ActFunc::Sigmoid }]).unwrap();


    let mut nn = NeuralNet::new(net_shape, 2).unwrap();

    println!("\nNN: \n{:?}", nn);
    
    let (mut input , mut expected_output) = read_mnist(".git/lfs/objects/fb/60/fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d".into()).unwrap();
    let input: Vec<Array2<f64>> = input.drain(0..1000).collect();
    let expected_output: Vec<Array2<f64>> = expected_output.drain(0..1000).collect();
    let training_data = TData::new(input, expected_output).unwrap();


    let (mut test_input, mut test_output) = read_mnist(".git/lfs/objects/51/c2/51c292478d94ec3a01461bdfa82eb0885d262eb09e615679b2d69dedb6ad09e7".into()).unwrap();
    let test_input: Vec<Array2<f64>> = test_input.drain(0..100).collect();
    let test_output: Vec<Array2<f64>> = test_output.drain(0..100).collect();
    let testing_data = TData::new(test_input.clone(), test_output.clone()).unwrap();


    let tsettings = TSettings::new(300, 0.005, false, 20, None, None).unwrap();
    nn.train(training_data, Some(testing_data), &tsettings).unwrap();

    
    
    for i in (0..test_input.len()).step_by(test_input.len() / 5) {
        nn.forward_propogate(test_input[i].clone());
        let output = nn.cached_output();
        let error = output - test_output[i].clone();
        
        let error = (&error * &error).sum();

        println!("\nExpected: {:?}\nCalculated: {:?}\nError: {}", test_output[i], output, error);
        
    }
    

    println!("\nNN: \n{:?}", nn);
}
