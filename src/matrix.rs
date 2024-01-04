use rand::{thread_rng, Rng};

#[derive(Debug)]
pub enum MatrixError {
    OutOfBounds,
    DimensionMismatch,
    UnexpectedMappingResult
}

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>
}

impl Matrix {
    pub fn from(data: Vec<Vec<f64>>) -> Self {
        Self {
            rows: data.len(),
            cols: data[0].len(),
            data
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![0f64; cols]; rows]
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = thread_rng();

        let mut matrix = Matrix::zeros(rows, cols);
        
        for x in 0..rows {
            for y in 0..cols {
                matrix.data[x][y] = rng.gen::<f64>();
            }
        }

        matrix
    }

    pub fn add(&self, target: &Self) -> anyhow::Result<Self, MatrixError> {
        // Check for possible mismatch in size.
        if self.rows != target.rows || self.cols != target.cols {
            return Err(MatrixError::DimensionMismatch)
        }

        // Do the summation.
        let mut result = Matrix::zeros(self.rows, self.cols);

        for x in 0..self.rows {
            for y in 0..self.cols {
                result.data[x][y] = self.data[x][y] + target.data[x][y];
            }
        }

        Ok(result)
    }

    pub fn sub(&self, target: &Self) -> anyhow::Result<Self, MatrixError> {
        // Check for possible mismatch in size.
        if self.rows != target.rows || self.cols != target.cols {
            return Err(MatrixError::DimensionMismatch)
        }

        // Do the subtraction.
        let mut result = Matrix::zeros(self.rows, self.cols);

        for x in 0..self.rows {
            for y in 0..self.cols {
                result.data[x][y] = self.data[x][y] - target.data[x][y];
            }
        }

        Ok(result)
    }

    pub fn mul(&self, target: &Self) -> anyhow::Result<Self, MatrixError> {
        if self.cols != target.rows {
            return Err(MatrixError::DimensionMismatch)
        }

        let mut result = Matrix::zeros(self.rows, target.cols);

        for x in 0..self.rows {
            for y in 0..target.cols {
                let mut sum = 0f64;
                
                for z in 0..self.cols {
                    sum += self.data[x][z] * target.data[z][y];
                }

                result.data[x][y] = sum;
            }
        }

        Ok(result)
    }

    pub fn dot_product(&self, target: &Self) -> anyhow::Result<Self, MatrixError> {
        // Check for possible mismatch in size.
        if self.rows != target.rows || self.cols != target.cols {
            return Err(MatrixError::DimensionMismatch)
        }

        // Do the subtraction.
        let mut result = Matrix::zeros(self.rows, self.cols);

        for x in 0..self.rows {
            for y in 0..self.cols {
                result.data[x][y] = self.data[x][y] * target.data[x][y];
            }
        }

        Ok(result)
    }

    pub fn map(&self, function: &dyn Fn(f64) -> f64) -> anyhow::Result<Self, MatrixError> {
        let mut result = Matrix::zeros(self.rows, self.cols);

        for x in 0..self.rows {
            for y in 0..self.cols {
                result.data[x][y] = function(self.data[x][y]);
            }
        }

        Ok(result)
    }

    pub fn transpose(&self) -> Self {
        let mut result = Matrix::zeros(self.cols, self.rows);

        for x in 0..self.rows {
            for y in 0..self.cols {
                result.data[y][x] = self.data[x][y];
            }
        }

        result
    }
}
