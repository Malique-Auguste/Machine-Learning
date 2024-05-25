mod neural_net;


#[cfg(test)]
mod tests {
    use crate::neural_net::{ActFunc, NeuralNet};
    use nalgebra::DMatrix;

    #[test]
    fn forward_prop() {
        let input = DMatrix::from_vec(1, 2, vec![1.0, 2.0]);
        let expected_output = DMatrix::from_element(1, 1, 84.0);

        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 3, 1], false).unwrap();

        println!("NN: {:?}", nn.get_weights());

        nn.forward_propogate(input).unwrap();

        assert_eq!(expected_output, *nn.cached_output())
    }

    #[test]
    fn forward_prop_2() {
        let input = DMatrix::from_vec(1, 2, vec![-6.0, 1.0]);
        let expected_output = DMatrix::from_element(1, 1, 3.0);

        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 2, 1], false).unwrap();

        println!("{:?}", nn.get_weights());

        nn.forward_propogate(input).unwrap();

        assert_eq!(expected_output, *nn.cached_output())
    }

    #[test]
    fn matrix() {
        let m1: DMatrix<i8> = DMatrix::from_vec(1, 3, vec![1, 2, 3]);
        let m2: DMatrix<i8> = DMatrix::from_vec(1, 4, vec![1, 2, 3, 1]);


        let mut input: DMatrix<i8> = m1.clone();
        let num_columns = input.shape().1;
        input = input.insert_column(num_columns, 1);

        assert_eq!(m2, input)

    }

    #[test]
    fn basic_net() {
        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 4, 1], true).unwrap();
        let input: DMatrix<f64> = DMatrix::from_vec(1, 2, vec![1.0, -3.0]);
        let expected_output = DMatrix::from_element(1, 1, 4.0);
        let alpha = 0.01;

        println!("\n{:?}\n", nn);


        for i in 0..15 {
            nn.forward_propogate(input.clone()).unwrap();
            let layer_delta = (nn.cached_output() - expected_output.clone()).sum();

            println!("{}) Output: {:?}, Error: {}", i, nn.cached_output()[0], layer_delta * layer_delta);
            //println!("\n{:?}\n", nn);
            nn.backward_propogate(layer_delta, alpha).unwrap();

        }

        println!("\n{:?}\n", nn)
    }

    #[test]
    fn basic_net_2() {
        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 2, 1], true).unwrap();
        let input: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(1, 2, vec![1.0, -3.0]),
                                            DMatrix::from_vec(1, 2, vec![3.0, 2.0]),
                                            DMatrix::from_vec(1, 2, vec![-2.0, -4.5]),
                                            DMatrix::from_vec(1, 2, vec![1.0, 3.0]),
                                            DMatrix::from_vec(1, 2, vec![-5.0, 1.5]),
                                            DMatrix::from_vec(1, 2, vec![-4.0, -1.2])];

        let expected_output: Vec<DMatrix<f64>> = vec![DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 1.0),
                                                    DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 0.0),
                                                    DMatrix::from_element(1, 1, 0.0)];

        let alpha = 0.01;

        println!("\n{:?}\n", nn);


        for n in 0..5 {
            let mut avg_error = 0.0;
            for i in 0..input.len() {
                nn.forward_propogate(input[i].clone()).unwrap();
                let layer_delta = (nn.cached_output() - expected_output[i].clone()).sum();

                println!("{}.{}) Output: {:?}, Error: {}", n, i, nn.cached_output()[0], layer_delta * layer_delta);
                avg_error += layer_delta * layer_delta;

                //println!("\n{:?}\n", nn);
                nn.backward_propogate(layer_delta, alpha).unwrap();
            }

            println!("Avg Error: {}\n", avg_error / (input.len() as f64))
        }



        println!("\n{:?}\n", nn)
    }
}