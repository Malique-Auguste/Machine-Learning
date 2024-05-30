mod neural_net;
mod act_func;


#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Read;

    use crate::neural_net::{NeuralNet};
    use crate::act_func::*;
    use nalgebra::DMatrix;

    #[test]
    fn forward_prop() {
        let input = DMatrix::from_vec(1, 2, vec![1.0, 2.0]);
        let expected_output = DMatrix::from_element(1, 1, 84.0);

        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 3, 1], None).unwrap();

        println!("NN: {:?}", nn.get_weights());

        nn.forward_propogate(input).unwrap();

        assert_eq!(expected_output, *nn.cached_output())
    }

    #[test]
    fn forward_prop_2() {
        let input = DMatrix::from_vec(1, 2, vec![-6.0, 1.0]);
        let expected_output = DMatrix::from_element(1, 1, 3.0);

        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 2, 1], None).unwrap();

        println!("{:?}", nn.get_weights());

        nn.forward_propogate(input).unwrap();

        assert_eq!(expected_output, *nn.cached_output())
    }

    #[test]
    fn matrix() {
        let mut m1: DMatrix<f64> = DMatrix::from_vec(1, 1, vec![5.0]);
        let mut m2: DMatrix<f64> = DMatrix::from_vec(1, 1, vec![7.0]);
        let mut m3: DMatrix<f64> = DMatrix::from_vec(1, 2, vec![7.0, -0.5]);


        println!("{:?}", m1.clone()-m2.clone());
        println!("{:?}", (m1-m2) * m3)


    }

    #[test]
    fn basic_net() {
        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 2, 1], Some(0)).unwrap();
        let input: DMatrix<f64> = DMatrix::from_vec(1, 2, vec![1.0, -3.0]);
        let expected_output = DMatrix::from_element(1, 1, 4.0);
        let alpha = 0.01;

        println!("\n{:?}\n", nn);


        for i in 0..5 {
            nn.forward_propogate(input.clone()).unwrap();
            let layer_delta = nn.cached_output() - expected_output.clone();

            if i % 1 == 0 {
                println!("\n{}) Output: {:?}, Pure Error: {}, Error: {}", i, nn.cached_output()[0], layer_delta.sum(), (layer_delta.clone() * layer_delta.clone()).sum());
                //println!("\n{:?}\n", nn);
            }
            nn.backward_propogate(layer_delta, alpha).unwrap();

        }

    }

    #[test]
    fn basic_basic_sig() {
        let mut nn = NeuralNet::new(ActFunc::Sigmoid, vec![2, 3, 1], Some(2)).unwrap();
        let input: DMatrix<f64> = DMatrix::from_vec(1, 2, vec![1.0, -3.0]);
        let expected_output = DMatrix::from_element(1, 1, 0.0);
        let alpha = 0.1;

        println!("\n{:?}\n", nn);


        for i in 0..301 {
            nn.forward_propogate(input.clone()).unwrap();
            let layer_delta = nn.cached_output() - expected_output.clone();

            if i % 30 == 0 {
                println!("\n{}) Output: {:?}, Pure Error: {}, Error: {}", i, nn.cached_output()[0], layer_delta.sum(), (layer_delta.clone() * layer_delta.clone()).sum());
                //println!("\n{:?}\n", nn);
            }
            nn.backward_propogate(layer_delta, alpha).unwrap();

        }

    }

    #[test]
    fn basic_net_2() {
        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 2, 1], Some(0)).unwrap();
        let input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![1.0, -3.0]),
                                            DMatrix::from_vec(1, 2, vec![3.0, 2.0]),
                                            DMatrix::from_vec(1, 2, vec![-2.0, -4.5]),
                                            DMatrix::from_vec(1, 2, vec![25.0, 1.5]),
                                            DMatrix::from_vec(1, 2, vec![15.0, 14.0]),
                                            DMatrix::from_vec(1, 2, vec![1.0, 3.0]),
                                            DMatrix::from_vec(1, 2, vec![-5.0, 1.5]),
                                            DMatrix::from_vec(1, 2, vec![-4.0, -1.2]),
                                            DMatrix::from_vec(1, 2, vec![-31.0, -5.5]),
                                            DMatrix::from_vec(1, 2, vec![27.0, 28.0])];

        //Rules: If LHS>RHS, 1, else, 0
        let expected_output: Vec<DMatrix<f64>> = vec![DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 0.0),];

        let training_error =  nn.train(input, expected_output, 50, 0.01).unwrap();

        for i in 0..(training_error.len() - 1) {
            if i % 3 == 0 {
                println!("{})Avg Error: {}", i, training_error[i])
            }
        }

        println!("Final Avg Error: {}\n", training_error.last().unwrap());


        let test_input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![12.0, -3.0]),
                                                DMatrix::from_vec(1, 2, vec![-7.0, 2.5]),
                                                DMatrix::from_vec(1, 2, vec![-13.0, -12.0])];
        let test_output: Vec<DMatrix<f64>> = vec![DMatrix::from_element(1, 1, 1.0),
                                                DMatrix::from_element(1, 1, 0.0),
                                                DMatrix::from_element(1, 1, 0.0)];
        
        for i in 0..test_input.len() {
            let output = nn.test(test_input[i].clone()).unwrap()[0];

            let delta = output - test_output[i][0];

            println!("Output: {}, Err: {}", output, delta * delta)
        }
    }

    #[test]
    fn basic_net_sig() {
        let mut nn = NeuralNet::new(ActFunc::Sigmoid, vec![2, 3, 1], Some(0)).unwrap();
        let input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![1.0, -3.0]),
                                            DMatrix::from_vec(1, 2, vec![3.0, 2.0]),
                                            DMatrix::from_vec(1, 2, vec![-2.0, -4.5]),
                                            DMatrix::from_vec(1, 2, vec![25.0, 1.5]),
                                            DMatrix::from_vec(1, 2, vec![15.0, 14.0]),
                                            DMatrix::from_vec(1, 2, vec![1.0, 3.0]),
                                            DMatrix::from_vec(1, 2, vec![-5.0, 1.5]),
                                            DMatrix::from_vec(1, 2, vec![-4.0, -1.2]),
                                            DMatrix::from_vec(1, 2, vec![-31.0, -5.5]),
                                            DMatrix::from_vec(1, 2, vec![27.0, 28.0])];

        //Rules: If LHS>RHS, 1, else, 0
        let expected_output: Vec<DMatrix<f64>> = vec![DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 0.0),];

        let training_error =  nn.train(input, expected_output, 200, 0.05).unwrap();

        for i in 0..(training_error.len() - 1) {
            if i % 15 == 0 {
                println!("{})Avg Error: {}", i, training_error[i])
            }
        }

        println!("Final Avg Error: {}\n", training_error.last().unwrap());


        let test_input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![12.0, -3.0]),
                                                DMatrix::from_vec(1, 2, vec![-7.0, 2.5]),
                                                DMatrix::from_vec(1, 2, vec![-13.0, -12.0])];
        let test_output: Vec<DMatrix<f64>> = vec![DMatrix::from_element(1, 1, 1.0),
                                                DMatrix::from_element(1, 1, 0.0),
                                                DMatrix::from_element(1, 1, 0.0)];
        
        for i in 0..test_input.len() {
            let output = nn.test(test_input[i].clone()).unwrap()[0];

            let delta = output - test_output[i][0];

            println!("Output: {}, Err: {}", output, delta * delta)
        }
    }

    #[test]
    fn xor() {
        let mut nn = NeuralNet::new(ActFunc::Sigmoid, vec![2, 3, 1], Some(3)).unwrap();
        let input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![0.0, 0.0]),
                                            DMatrix::from_vec(1, 2, vec![0.0, 1.0]),
                                            DMatrix::from_vec(1, 2, vec![1.0, 0.0]),
                                            DMatrix::from_vec(1, 2, vec![1.0, 1.0]),];

        //Rules: If LHS>RHS, 1, else, 0
        let expected_output: Vec<DMatrix<f64>> = vec![DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 0.0)];

        let training_error =  nn.train(input.clone(), expected_output.clone(), 300, 0.5).unwrap();

        for i in 0..(training_error.len() - 1) {
            if i % 10 == 0 {
                println!("{})Avg Error: {}", i, training_error[i])
            }
        }

        println!("Final Avg Error: {}\n", training_error.last().unwrap());
        
        for i in 0..input.len() {
            let output = nn.test(input[i].clone()).unwrap()[0];

            let delta = output - expected_output[i][0];

            println!("Output: {}, Err: {}", output, delta * delta)
        }
    
    }


    #[test]
    fn housing() {
        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![9, 9, 1], Some(2)).unwrap();

        let mut raw_data_file = fs::OpenOptions::new().read(true).open("ML Housing Dataset - Sheet1.tsv").unwrap();
        let mut raw_data_content = String::new();

        raw_data_file.read_to_string(&mut raw_data_content).unwrap();

        let raw_data_lines: Vec<&str> = raw_data_content.split("\r\n").collect();

        let mut input: Vec<DMatrix<f64>> = Vec::new();
        let mut expected_output: Vec<DMatrix<f64>> = Vec::new();


        for i in 3..6 {
            //println!("I: {:?}", raw_data_lines[i]);


            let mut temp: Vec<f64> = raw_data_lines[i].split('\t').map(|n| {
                //println!("N: {}", n);
                n.parse::<f64>().unwrap()
            }).collect();

            input.push(DMatrix::from_iterator(1, 9, temp.drain(0..9)));
            expected_output.push(DMatrix::from_element(1, 1, temp[0]));
        }

        println!("In: {:?}\nOut: {:?}", input, expected_output);


        let training_error =  nn.train(input.clone(), expected_output.clone(), 1000, 0.02).unwrap();

        for i in 0..(training_error.len() - 1) {
            if i % 50 == 0 {
                println!("{})Avg Error: {}", i, training_error[i])
            }
        }

        println!("Final Avg Error: {}\n", training_error.last().unwrap());
        
        /*
        for i in 0..input.len() {
            let output = nn.test(input[i].clone()).unwrap()[0];

            let delta = output - expected_output[i][0];

            println!("Output: {}, Err: {}", output, delta * delta)
        }
        */
    
    }
}