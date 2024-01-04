use crate::{matrix::{Matrix, MatrixError}, activations::Activation};

#[derive(Debug)]
pub enum NetworkError {
    InvalidFeedForwardInputSize,
    InvalidBackProgragationParameters,
    InternalComputationError(MatrixError)
}

impl From<MatrixError> for NetworkError {
    fn from(value: MatrixError) -> Self {
        Self::InternalComputationError(value)
    }
}

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation<'a>,
    learning_rate: f64
}

impl Network<'_> {
    pub fn new<'a>(layers: Vec<usize>, learning_rate: f64, activation: Activation<'a>) -> Network<'a> {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i+1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate
        }
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> anyhow::Result<Vec<f64>, NetworkError> {
        if inputs.len() != self.layers[0] {
            return Err(NetworkError::InvalidFeedForwardInputSize)
        }

        let mut current = Matrix::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];

        for x in 0..self.layers.len() - 1 {
            current = self.weights[x]
                .mul(&current)?
                .add(&self.biases[x])?
                .map(self.activation.function)?;
            
            self.data.push(current.clone());
        }

        Ok(current.data[0].to_owned())
    }

    pub fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) -> anyhow::Result<(), NetworkError> {
        if targets.len() != self.layers[self.layers.len() - 1] {
            return Err(NetworkError::InvalidBackProgragationParameters)
        }

        let parsed = Matrix::from(vec![outputs]);
        let mut errors = Matrix::from(vec![targets]).sub(&parsed)?;
        let mut gradients = parsed.map(self.activation.derivative)?;

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot_product(&errors)?.map(&|x| x * self.learning_rate)?;

            let res = self.weights[i].add(&gradients.mul(&self.data[i].transpose())?)?;
            self.weights[i] = res;

            self.biases[i] = self.biases[i].add(&gradients)?;

            let res = self.weights[i].transpose().mul(&errors);

            errors = match res {
                Ok (m) => m,
                Err(e) => {
                    eprintln!("Error 2!");
                    Matrix::zeros(2, 2)
                }
            };

            gradients = self.data[i].map(self.activation.derivative)?;
        }

        Ok(())
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                println!("Epoch {} of {}", i, epochs);
            }

            for j in 0..inputs.len() {
                let outputs = self.feed_forward(inputs[j].clone()).unwrap();
                self.back_propagate(outputs, targets[j].clone()).unwrap();
            }
        }
    }
}