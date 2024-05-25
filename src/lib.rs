mod neural_net;


#[cfg(test)]
mod tests {
    use crate::neural_net::{ActFunc, NeuralNet};
    use nalgebra::DMatrix;

    #[test]
    fn forward_prop() {
        let input = DMatrix::from_vec(1, 2, vec![1.0, 2.0]);
        let expected_output = DMatrix::from_element(1, 1, 2.75);

        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 3, 1]).unwrap();

        println!("NN: {:?}", nn.get_weights());

        nn.forward_propogate(input).unwrap();

        assert_eq!(expected_output, *nn.cached_output())
    }

    #[test]
    fn forward_prop_2() {
        let input = DMatrix::from_vec(1, 2, vec![-6.0, 1.0]);
        let expected_output = DMatrix::from_element(1, 1, 0.5);

        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 2, 1]).unwrap();

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
        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 2, 1]).unwrap();
        let input: DMatrix<f64> = DMatrix::from_vec(1, 2, vec![1.0, -3.0]);
        let expected_output = DMatrix::from_element(1, 1, 4.0);
        let alpha = 0.5;

        for i in 0..25 {
            nn.forward_propogate(input.clone()).unwrap();
            let layer_delta = (nn.cached_output() - expected_output.clone()).sum();

            println!("{}) Output: {:?}, Error: {}", i, nn.cached_output()[0], layer_delta * layer_delta);
            nn.backward_propogate(layer_delta, alpha).unwrap();

        }

        println!("\n{:?}\n", nn)
        

    }
}