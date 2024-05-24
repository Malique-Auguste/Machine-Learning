mod neural_net;


#[cfg(test)]
mod tests {
    use crate::neural_net::{ActFunc, NeuralNet};
    use nalgebra::DMatrix;

    #[test]
    fn forward_prop() {
        let input = DMatrix::from_vec(1, 2, vec![1.0, 2.0]);
        let expected_output = DMatrix::from_element(1, 1, 9.0);

        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 3, 1]).unwrap();

        println!("{:?}", nn.get_weights());

        nn.forward_propogate(input).unwrap();

        assert_eq!(expected_output, *nn.cached_output())
    }

    #[test]
    fn forward_prop_2() {
        let input = DMatrix::from_vec(1, 2, vec![-3.0, 2.0]);
        let expected_output = DMatrix::from_element(1, 1, 0.0);

        let mut nn = NeuralNet::new(ActFunc::ReLU, vec![2, 3, 1]).unwrap();

        println!("{:?}", nn.get_weights());

        nn.forward_propogate(input).unwrap();

        assert_eq!(expected_output, *nn.cached_output())
    }

    #[test]
    fn matrix() {
        let m1 = DMatrix::from_vec(2, 3, vec![1, 2, 3, 2, 3, 4]);
        let m1 = DMatrix::from_vec(2, 3, vec![1, 2, 3, 2, 3, 4]);

    }
}