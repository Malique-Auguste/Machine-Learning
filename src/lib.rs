pub mod neural_net;
pub mod dataset_handler;


#[cfg(test)]
mod tests {
    use super::*;
    use neural_net::{*, act_func::*, training_helpers::*, net_shape::*, net_layer::*};
    use dataset_handler::*;
    use ndarray::{arr2, Array2};

    #[test]
    fn matrix() {
        let n1: Array2<f64> = arr2(&[[0.0, 1., 2., 3.]]);
        let n2: Array2<f64> = arr2(&[[0.0, 1., 2., 3.]]);


        println!("{:?}", &n1 * &n2);
        println!("{:?}", &n1.dot(&n2));
        

    }

    #[test]
    fn basic_net_2() {
        let net_shape = NetShape::new(vec![NetLayerType::DenseLayer { input_node_num: 2, output_node_num: 2, act_func: ActFunc::ReLU}, 
                                                                NetLayerType::DenseLayer { input_node_num: 2, output_node_num: 1, act_func: ActFunc::ReLU}]).unwrap();

        let mut nn = NeuralNet::new(net_shape, 1).unwrap();
        println!("\nNN:\n{:?}\n", nn);

        let input: Vec<Array2<f64>> = vec![arr2(&[[1.0, -3.0]]),
                                            arr2(&[[3.0, 2.0]]),
                                            arr2(&[[-2.0, -4.5]]),
                                            arr2(&[[25.0, 1.5]]),
                                            arr2(&[[15.0, 14.0]]),
                                            arr2(&[[1.0, 3.0]]),
                                            arr2(&[[-5.0, 1.5]]),
                                            arr2(&[[-4.0, -1.2]]),
                                            arr2(&[[-31.0, -5.5]]),
                                            arr2(&[[27.0, 28.0]])];

        //Rules: If LHS>RHS, 1, else, 0
        let expected_output: Vec<Array2<f64>> = vec![Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 0.0),
                                                    Array2::from_elem((1, 1), 0.0),
                                                    Array2::from_elem((1, 1), 0.0),
                                                    Array2::from_elem((1, 1), 0.0),
                                                    Array2::from_elem((1, 1), 0.0),];

        let test_input: Vec<Array2<f64>> = vec![arr2(&[[10.0, 5.0]]),
                                                arr2(&[[12.0, -3.0]]),
                                                arr2(&[[-7.0, 2.5]]),
                                                arr2(&[[-16.0, -12.0]])];
        let test_output: Vec<Array2<f64>> = vec![Array2::from_elem((1, 1), 1.0),
                                                Array2::from_elem((1, 1), 1.0),
                                                Array2::from_elem((1, 1), 0.0),
                                                Array2::from_elem((1, 1), 0.0)];

        let training_data = TData::new(input, expected_output).unwrap();
        let testing_data = TData::new(test_input, test_output).unwrap();
        let tsettings = TSettings::new(150, 0.01, false, 5, None, None).unwrap();

        nn.train(training_data, Some(testing_data), &tsettings).unwrap();
    }


    #[test]
    fn basic_net_sig() {
        let net_shape = NetShape::new(vec![NetLayerType::DenseLayer { input_node_num: 2, output_node_num: 3, act_func: ActFunc::Sigmoid}, 
                                                                    NetLayerType::DenseLayer { input_node_num: 3, output_node_num: 1, act_func: ActFunc::Sigmoid}]).unwrap();

        let mut nn = NeuralNet::new(net_shape, 7).unwrap();
        let input: Vec<Array2<f64>> = vec![arr2(&[[2.0, 1.0]]),
                                            arr2(&[[3.0, -2.0]]),
                                            arr2(&[[-10.0, 5.5]]),
                                            arr2(&[[-25.0, -20.5]]),
                                            arr2(&[[35.0, 17.0]]),
                                            arr2(&[[25.5, 48.0]]),
                                            arr2(&[[21.0, -40.5]]),
                                            arr2(&[[-17.7, -30.2]]),
                                            arr2(&[[-11.0, -15.5]]),
                                            arr2(&[[6.0, 9.0]])];

        //Rules: If |LHS|>|RHS|, 1, else, 0
        let expected_output: Vec<Array2<f64>> = vec![Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 0.0),
                                                    Array2::from_elem((1, 1), 0.0),
                                                    Array2::from_elem((1, 1), 0.0),
                                                    Array2::from_elem((1, 1), 0.0),
                                                    Array2::from_elem((1, 1), 0.0),];

        let test_input: Vec<Array2<f64>> = vec![arr2(&[[6.0, 5.0]]),
                                                    arr2(&[[12.0, -7.5]]),
                                                    arr2(&[[-9.9, 18.5]]),
                                                    arr2(&[[-21.0, -32.0]]),];
        let test_output: Vec<Array2<f64>> = vec![Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 0.0),
                                                    Array2::from_elem((1, 1), 0.0)];

        let training_data = TData::new(input, expected_output).unwrap();
        let testing_data = TData::new(test_input, test_output).unwrap();
        let tsettings = TSettings::new(350, 0.5, false, 10, None, None).unwrap();



        nn.train(training_data, Some(testing_data), &tsettings).unwrap();
    }


    #[test]
    fn xor() {
        let net_shape = NetShape::new(vec![NetLayerType::DenseLayer { input_node_num: 2, output_node_num: 3, act_func: ActFunc::Sigmoid}, 
                                                                NetLayerType::DenseLayer { input_node_num: 3, output_node_num: 1, act_func: ActFunc::Sigmoid}]).unwrap();

        let mut nn = NeuralNet::new(net_shape, 7).unwrap();
        let input: Vec<Array2<f64>> = vec![arr2(&[[0.0, 0.0]]),
                                            arr2(&[[0.0, 1.0]]),
                                            arr2(&[[1.0, 0.0]]),
                                            arr2(&[[1.0, 1.0]])];

        //Rules: If LHS>RHS, 1, else, 0
        let expected_output: Vec<Array2<f64>> = vec![Array2::from_elem((1, 1), 0.0),
                                                    Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 1.0),
                                                    Array2::from_elem((1, 1), 0.0)];

        let training_data = TData::new(input.clone(), expected_output.clone()).unwrap();
        let tsettings = TSettings::new(2000, 0.09, false, 10, None, None).unwrap();

        nn.train(training_data, None, &tsettings).unwrap();
        println!("\n");
        
        for i in 0..input.len() {
            nn.forward_propogate(input[i].clone());
            let output = nn.cached_output()[(0,0)];
            let delta = output - expected_output[i][(0,0)];

            println!("Output: {}, Delta: {}, Err: {}", output, delta, delta * delta)
        }
    
    }

    #[test]
    fn dense_number_rec() {
        println!("Running...");

        let net_shape = NetShape::new(vec![NetLayerType::DenseLayer { input_node_num: 784, output_node_num: 40, act_func: ActFunc::Sigmoid }, 
                                                                    NetLayerType::DenseLayer { input_node_num: 40, output_node_num: 10, act_func: ActFunc::Sigmoid }]).unwrap();


        let mut nn = NeuralNet::new(net_shape, 2).unwrap();

        println!("a");
        
        let (mut input , mut expected_output) = read_mnist(".git/lfs/objects/fb/60/fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d".into()).unwrap();
        let input: Vec<Array2<f64>> = input.drain(0..1000).collect();
        let expected_output: Vec<Array2<f64>> = expected_output.drain(0..1000).collect();
        let training_data = TData::new(input.clone(), expected_output.clone()).unwrap();


        let (mut test_input, mut test_output) = read_mnist(".git/lfs/objects/51/c2/51c292478d94ec3a01461bdfa82eb0885d262eb09e615679b2d69dedb6ad09e7".into()).unwrap();
        let test_input: Vec<Array2<f64>> = test_input.drain(0..100).collect();
        let test_output: Vec<Array2<f64>> = test_output.drain(0..100).collect();
        let testing_data = TData::new(test_input.clone(), test_output.clone()).unwrap();


        let tsettings = TSettings::new(100, 0.005, false, 15, None, None).unwrap();
        nn.train(training_data, Some(testing_data), &tsettings).unwrap();

        

        for i in (0..input.len()).step_by(input.len() / 3) {
            nn.forward_propogate(input[i].clone());
            let output = nn.cached_output();
            let error = output - expected_output[i].clone();
            
            let error = (&error * &error).sum();

            println!("\nExpected: {:?}\nCalculated: {:?}\nError: {}", expected_output[i], output, error);
            
        }
    }
}