use ndarray::Array2;
use serde::{Serialize, Deserialize};
use std::path::Path;


//Training Data
#[derive( Clone, Serialize, Deserialize)]
pub struct TData {
    input: Vec<Array2<f64>>,
    output: Vec<Array2<f64>>
}

impl TData {
    pub fn new(input: Vec<Array2<f64>>, output: Vec<Array2<f64>>) -> Result<TData, String> {
        //ensures that the matricies for the inputs and outputs are correct
        for i in 0..input.len() {
            if input[i].shape() != input[0].shape() {
                return Err(format!("{}th input matrix doesn't have the expected shape. {:?} != {:?}", i, input[i].shape(), input[0].shape()))
            }
        }

        for i in 0..output.len() {
            if output[i].shape() != output[0].shape() {
                return Err(format!("{}th output matrix doesn't have the expected shape. {:?} != {:?}", i, output[i].shape(), output[0].shape()))
            }
        }

        Ok(TData {
            input,
            output
        })
    }

    pub fn input(&self) -> &Vec<Array2<f64>> {
        &self.input
    }

    pub fn output(&self) -> &Vec<Array2<f64>> {
        &self.output
    }
}

#[derive(Clone, Serialize, Deserialize)]
//training settings
pub struct TSettings{
    iterations: usize,
    alpha: f64,
    dropout: bool,
    print_frequency: usize,
    //1st string represents path to nn version with lowest error
    //2nd string represent path to current itertion of net
    save_paths: Option<[String; 2]>
}

impl TSettings {
    pub fn new(iterations: usize, alpha: f64, dropout: bool, print_frequency: usize, save_paths: Option<[String; 2]>) -> Result<TSettings, String> {
        if print_frequency > iterations {
            Err("Print frequency cannot be greater than #iterations".into())
        }
        else if alpha < 1.0 && alpha > 0.0 {
            match save_paths {
                Some(paths) => {
                    if Path::new(&paths[0]).exists() {
                        if Path::new(&paths[1]).exists() {
                            Ok(TSettings {
                                iterations,
                                alpha,
                                dropout,
                                print_frequency,
                                save_paths: Some(paths)
                            })
                        }
                        else {
                            Err(format!("Save path 2 ({}) doesn't exist.", paths[1]))
                        }
                    }
                    else {
                        Err(format!("Save path 1 ({}) doesn't exist.", paths[0]))
                    }
                },

                None => {
                    Ok(TSettings {
                        iterations,
                        alpha,
                        dropout,
                        print_frequency,
                        save_paths
                    })
                }
            }
        }
        else {
            Err("Alpha must be between 1 and 0".into())
        }
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn dropout(&self) -> bool {
        self.dropout
    }

    pub fn print_frequency(&self) -> usize {
        self.print_frequency
    }

    pub fn save_paths(&self) -> &Option<[String; 2]> {
        &self.save_paths
    }
}