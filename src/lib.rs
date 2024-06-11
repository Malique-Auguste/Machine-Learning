mod neural_net;
mod act_func;


#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Read;

    use crate::neural_net::{NeuralNet, TSettings};
    use crate::act_func::*;
    use nalgebra::DMatrix;

    #[test]
    fn forward_prop() {
        let input = DMatrix::from_vec(1, 2, vec![1.0, 2.0]);
        let expected_output = DMatrix::from_element(1, 1, 84.0);

        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 3, 1], 0).unwrap();

        println!("NN: {:?}", nn.get_weights());

        nn.forward_propogate(input).unwrap();

        assert_eq!(expected_output, *nn.cached_output())
    }

    #[test]
    fn forward_prop_2() {
        let input = DMatrix::from_vec(1, 2, vec![-6.0, 1.0]);
        let expected_output = DMatrix::from_element(1, 1, 3.0);

        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 2, 1], 0).unwrap();

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
    fn basic_net_2() {
        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 2, 1], 0).unwrap();
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

        let tsettings = TSettings::new(150, 0.01, false, 5).unwrap();

        nn.train(input, expected_output, &tsettings).unwrap();

        let test_input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![12.0, -3.0]),
                                                DMatrix::from_vec(1, 2, vec![-7.0, 2.5]),
                                                DMatrix::from_vec(1, 2, vec![-16.0, -12.0])];
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
        let mut nn = NeuralNet::new(ActFunc::Sigmoid, vec![2, 4, 1], 2).unwrap();
        let input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![2.0, 1.0]),
                                            DMatrix::from_vec(1, 2, vec![3.0, -2.0]),
                                            DMatrix::from_vec(1, 2, vec![-10.0, 5.5]),
                                            DMatrix::from_vec(1, 2, vec![-25.0, -20.5]),
                                            DMatrix::from_vec(1, 2, vec![35.0, 17.0]),
                                            DMatrix::from_vec(1, 2, vec![25.5, 48.0]),
                                            DMatrix::from_vec(1, 2, vec![21.0, -40.5]),
                                            DMatrix::from_vec(1, 2, vec![-17.7, -30.2]),
                                            DMatrix::from_vec(1, 2, vec![-11.0, -15.5]),
                                            DMatrix::from_vec(1, 2, vec![6.0, 9.0])];

        //Rules: If |LHS|>|RHS|, 1, else, 0
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

        let tsettings = TSettings::new(350, 0.4, false, 20).unwrap();

        nn.train(input, expected_output, &tsettings).unwrap();


        let test_input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![6.0, 5.0]),
                                                DMatrix::from_vec(1, 2, vec![12.0, -7.5]),
                                                DMatrix::from_vec(1, 2, vec![-9.9, 18.5]),
                                                DMatrix::from_vec(1, 2, vec![-21.0, -32.0]),];
        let test_output: Vec<DMatrix<f64>> = vec![DMatrix::from_element(1, 1, 1.0),
                                                DMatrix::from_element(1, 1, 1.0),
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
        let mut nn = NeuralNet::new(ActFunc::Sigmoid, vec![2, 3, 1], 3).unwrap();
        let input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![0.0, 0.0]),
                                            DMatrix::from_vec(1, 2, vec![0.0, 1.0]),
                                            DMatrix::from_vec(1, 2, vec![1.0, 0.0]),
                                            DMatrix::from_vec(1, 2, vec![1.0, 1.0]),];

        //Rules: If LHS>RHS, 1, else, 0
        let expected_output: Vec<DMatrix<f64>> = vec![DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 0.0)];

        let tsettings = TSettings::new(1000, 0.4, false, 15).unwrap();

        nn.train(input.clone(), expected_output.clone(), &tsettings).unwrap();
        
        for i in 0..input.len() {
            let output = nn.test(input[i].clone()).unwrap()[0];

            let delta = output - expected_output[i][0];

            println!("Output: {}, Err: {}", output, delta * delta)
        }
    
    }


    #[test]
    fn regularising() {
        let mut nn = NeuralNet::new(ActFunc::Sigmoid, vec![2, 8, 8, 1], 1).unwrap();
        let input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![2.0, 1.0]),
                                            DMatrix::from_vec(1, 2, vec![3.0, -2.0]),
                                            DMatrix::from_vec(1, 2, vec![-10.0, 5.5]),
                                            DMatrix::from_vec(1, 2, vec![-25.0, -20.5]),
                                            DMatrix::from_vec(1, 2, vec![35.0, 17.0]),
                                            DMatrix::from_vec(1, 2, vec![25.5, 48.0]),
                                            DMatrix::from_vec(1, 2, vec![21.0, -40.5]),
                                            DMatrix::from_vec(1, 2, vec![-17.7, 30.2]),
                                            DMatrix::from_vec(1, 2, vec![-11.0, -15.5]),
                                            DMatrix::from_vec(1, 2, vec![6.0, 9.0])];

        //Rules: If |LHS|>|RHS|, 1, else, 0
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


        let tsettings = TSettings::new(1000, 0.1, true, 10).unwrap();

        nn.train(input, expected_output, &tsettings).unwrap();


        let test_input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![6.0, 5.0]),
                                                DMatrix::from_vec(1, 2, vec![12.0, -7.5]),
                                                DMatrix::from_vec(1, 2, vec![-10.9, 18.5]),
                                                DMatrix::from_vec(1, 2, vec![-21.0, -32.0]),];
        let test_output: Vec<DMatrix<f64>> = vec![DMatrix::from_element(1, 1, 1.0),
                                                DMatrix::from_element(1, 1, 1.0),
                                                DMatrix::from_element(1, 1, 0.0),
                                                DMatrix::from_element(1, 1, 0.0)];
        
        for i in 0..test_input.len() {
            let output = nn.test(test_input[i].clone()).unwrap()[0];

            let delta = output - test_output[i][0];

            println!("Output: {}, Err: {}", output, delta * delta)
        }
    }

}