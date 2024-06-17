//Network Shape
#[derive(Debug)]
pub struct NetShape {
    input_node_num: usize,
    hidden_node_num: Vec<usize>,
    output_node_num: usize,
    
    kernel_size: Option<usize>,
}

impl NetShape {
    pub fn new(input_node_num: usize, hidden_node_num: Vec<usize>, output_node_num: usize, kernel_size: Option<usize>) -> Result<NetShape, String> {
        if input_node_num == 0 {
            return Err("Input node num must be greater than zero.".into());
        }
        else if output_node_num == 0 {
            return Err("Output node num must be greater than zero.".into());
        }
        
        if let Some(k_size) = kernel_size {
            let sqr_root = f64::sqrt(input_node_num as f64);
            if sqr_root.floor() != sqr_root {
                return  Err("Input nodes cannot form a square but a square kernel is given".into());
            }
            else if k_size >= sqr_root as usize {
                return  Err(format!("Kernal size ({}) must be greater than sqrt of input_node_num ({}).", k_size, sqr_root ));
            }
        }
        
        Ok(NetShape {
            input_node_num,
            hidden_node_num,
            output_node_num,
            kernel_size,
        })
    }

    pub fn input_node_num(&self) -> &usize {
        &self.input_node_num
    }

    pub fn hidden_node_num(&self) -> &Vec<usize> {
        &self.hidden_node_num
    }

    pub fn output_node_num(&self) -> &usize {
        &self.output_node_num
    }

    pub fn kernel_size(&self) -> &Option<usize> {
        &self.kernel_size
    }
}