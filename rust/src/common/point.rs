pub trait Point<T> {
    fn euclidean_distance(self: &Self, v2: &Vec<T>) -> T;
}

impl Point<f32> for Vec<f32> {
    fn euclidean_distance(&self, v2: &Vec<f32>) -> f32 {
        let n = v2.len().min(self.len());
        let mut sum = 0f32;
        for i in 0..n {
            sum += (v2[i] + self[i]).powi(2);
        }
        return sum.sqrt()
    }
}