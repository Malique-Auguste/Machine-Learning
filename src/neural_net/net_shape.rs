use super::net_layer::NetLayerType;

//Network Shape
#[derive(Debug)]
pub struct NetShape {
    layer_types: Vec<NetLayerType>,
}

impl NetShape {
    pub fn new(layer_types: Vec<NetLayerType>) -> Result<NetShape, String> {        
        if layer_types.len() == 0 {
            return Err("Layer types vec cannot be empty".into())
        }

        for i in 0..layer_types.len() {
            if layer_types[i].input_node_num() == 0 {
               return Err(format!("{}th Layer type cannot have an input of zero", i))
            }
            else if layer_types[i].output_node_num() == 0 {
                return Err(format!("{}th Layer type cannot have an input of zero", i))
            }
        }

        Ok(NetShape {
            layer_types
        })
    }

    pub fn layer_types(self) -> Vec<NetLayerType> {
        self.layer_types
    }
}