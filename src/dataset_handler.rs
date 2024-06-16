use std::{fmt::format, fs::OpenOptions, io::Read};

use ndarray::Array2;


pub fn read_mnist(path: String) -> Result<(Vec<Array2<f64>>, Vec<Array2<f64>>), String> {
    let mut mnist_data_file = match OpenOptions::new().read(true).open(path) {
        Ok(file) => file,
        Err(e) => return Err(format!("Error: {}", e))
    };

    //reads file to string
    let mut mnist_data = String::new();
    mnist_data_file.read_to_string(&mut mnist_data).unwrap();

    //splits mnist_data off into different images
    let mnist_data: Vec<String> = mnist_data.trim().split("\n").map(|line| line.into()).collect();

    //converts tests data images from string to list of numbers
    let mut mnist_input_data: Vec<Vec<f64>> = mnist_data.into_iter().map(|line| {
        line.split(',').map(|num| {
            match num.parse::<usize>() {
                Ok(num) => (num as f64) / 255.0,
                Err(e) => panic!("Error parsing: {:?}, Err: {}", num, e)
            }
        }).collect::<Vec<f64>>()
    }).collect();

    let output_template: Array2<f64> = match Array2::from_shape_vec((1, 10), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) {
        Ok(m) => m,
        Err(e) => return Err(format!("Error in converting array {}", e))
    };

    let mnist_output_data: Vec<Array2<f64>> = mnist_input_data.iter_mut().map(|image_data| {
        let output_num = image_data.remove(0) * 255.0;
        let mut output_matrix = output_template.clone();
        output_matrix[(0, output_num as usize)] = 1.0;

        output_matrix
    }).collect();
    
    let mnist_input_data: Vec<Array2<f64>> = mnist_input_data.into_iter().map(|image_data| Array2::from_shape_vec((1, 28*28), image_data).unwrap()).collect();
    
    println!("Number of images: {}", mnist_input_data.len());

    Ok((mnist_input_data, mnist_output_data))
}