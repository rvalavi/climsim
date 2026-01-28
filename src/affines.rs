use std::convert::From;

// Define the Affine struct for transforming ij to xy
#[derive(Debug, Clone, Copy)]
pub struct Affine {
    pub x_scale: f64,   // pixel width
    pub x_skew: f64,    // rotation/skew term (usually 0)
    pub x_origin: f64,  // top-left corner X
    pub y_skew: f64,    // rotation/skew term (usually 0)
    pub y_scale: f64,   // pixel height (negative if north-up)
    pub y_origin: f64,  // top-left corner Y
}

impl Affine {
    /// Convert (row, col) to (x, y) at pixel centre by default.
    pub fn xy(&self, row: usize, col: usize) -> (f64, f64) {
        let col_f = (col as f64) + 0.5;
        let row_f = (row as f64) + 0.5;

        let x = self.x_scale * col_f + self.x_skew * row_f + self.x_origin;
        let y = self.y_skew * col_f + self.y_scale * row_f + self.y_origin;
        (x, y)
    }

    /// Convert (x, y) coordinates to continuous (row, col) indices.
    /// It performs the inverse affine transformation.
    pub fn ij(&self, x: f64, y: f64, nrows: usize, ncols: usize) -> Option<(usize, usize)> {
        let det = self.x_scale * self.y_scale - self.x_skew * self.y_skew;
        
        if det.abs() < 1e-10 {
            return None;
        }
        
        let dx = x - self.x_origin;
        let dy = y - self.y_origin;
        let col = (self.y_scale * dx - self.x_skew * dy) / det;
        let row = (-self.y_skew * dx + self.x_scale * dy) / det;

        if row < 0.0 || col < 0.0 || row >= nrows as f64 || col >= ncols as f64 {
            None
        } else {
            Some((row.floor() as usize, col.floor() as usize))
        }
    }
}

impl From<(f64, f64, f64, f64, f64, f64)> for Affine {
    fn from(t: (f64, f64, f64, f64, f64, f64)) -> Self {
        Self {
            x_scale: t.0,
            x_skew: t.1,
            x_origin: t.2,
            y_skew: t.3,
            y_scale: t.4,
            y_origin: t.5,
        }
    }
}
