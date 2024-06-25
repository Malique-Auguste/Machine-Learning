pub mod neural_net;
pub mod dataset_handler;


#[cfg(test)]
mod tests {
    use super::*;
    use neural_net::{*, act_func::*, training_helpers::*, net_shape::*, net_layer::*};
    use ndarray::{arr2, Array2, Shape};

    #[test]
    fn matrix() {
        let n1: Array2<f64> = arr2(&[[0.0, 1., 2., 3.]]);
        let n2: Array2<f64> = arr2(&[[0.0, 1., 2., 3.]]);


        println!("{:?}", &n1 * &n2);
        println!("{:?}", &n1.dot(&n2));
        

    }

    #[test]
    fn basic_net_2() {
        let net_shape = NetShape::new(vec![NetLayerType::DenseLayer { input_node_num: 2, output_node_num: 2 }, 
                                                                NetLayerType::DenseLayer { input_node_num: 2, output_node_num: 1 }]).unwrap();

        let mut nn = NeuralNet::new(ActFunc::ReLU, net_shape, 1).unwrap();
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
        let tsettings = TSettings::new(150, 0.001, false, 5).unwrap();

        nn.train(training_data, Some(testing_data), &tsettings).unwrap();
    }


    #[test]
    fn basic_net_sig() {
        let net_shape = NetShape::new(vec![NetLayerType::DenseLayer { input_node_num: 2, output_node_num: 3 }, 
                                                                    NetLayerType::DenseLayer { input_node_num: 3, output_node_num: 1 }]).unwrap();

        let mut nn = NeuralNet::new(ActFunc::Sigmoid, net_shape, 8).unwrap();
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
        let tsettings = TSettings::new(350, 0.4, false, 10).unwrap();



        nn.train(training_data, Some(testing_data), &tsettings).unwrap();
    }


    #[test]
    fn xor() {
        let net_shape = NetShape::new(vec![NetLayerType::DenseLayer { input_node_num: 2, output_node_num: 3 }, 
                                                                NetLayerType::DenseLayer { input_node_num: 3, output_node_num: 1 }]).unwrap();

        let mut nn = NeuralNet::new(ActFunc::Sigmoid, net_shape, 7).unwrap();
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
        let tsettings = TSettings::new(2000, 0.09, false, 10).unwrap();

        nn.train(training_data, None, &tsettings).unwrap();
        println!("\n");
        
        for i in 0..input.len() {
            nn.forward_propogate(input[i].clone());
            let output = nn.cached_output()[(0,0)];
            let delta = output - expected_output[i][(0,0)];

            println!("Output: {}, Delta: {}, Err: {}", output, delta, delta * delta)
        }
    
    }
}