#[derive(Debug)]
pub struct LinearRegressionResult {
    pub slope: f32,
    pub intercept: f32,
    pub r2: f32,
}

#[no_mangle]
pub extern "C" fn linear_regression_get_slope(res: *const LinearRegressionResult) -> f32 {
    let _res = unsafe { &*res };
    return _res.slope;
}
#[no_mangle]
pub extern "C" fn linear_regression_get_intercept(res: *const LinearRegressionResult) -> f32 {
    let _res = unsafe { &*res };
    return _res.intercept;
}
#[no_mangle]
pub extern "C" fn linear_regression_get_r2(res: *const LinearRegressionResult) -> f32 {
    let _res = unsafe { &*res };
    return _res.r2;
}
#[no_mangle]
pub extern "C" fn linear_regression_free_res(res: *const LinearRegressionResult) {
    let _res = res.to_owned();
}
#[no_mangle]
pub unsafe extern "C" fn linear_regression(
    x_ptr: *const f32,
    y_ptr: *const f32,
    x_len: usize,
    y_len: usize,
) -> isize {
    let x = std::slice::from_raw_parts(x_ptr, x_len);
    let y = std::slice::from_raw_parts(y_ptr, y_len);
    let n = x_len.min(y_len);
    let num = n as f32;
    let mut mean_x = 0f32;
    let mut mean_y = 0f32;
    for i in 0..n {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x = mean_x / num;
    mean_y = mean_y / num;

    let mut stddev_xy = 0f32;
    let mut stddev_x = 0f32;
    for i in 0..n {
        stddev_xy += (x[i] - mean_x) * (y[i] - mean_y);
        stddev_x += (x[i] - mean_x) * (x[i] - mean_y);
    }
    let slope = stddev_xy / stddev_x;
    let intercept = mean_y - (slope * mean_x);
    let mut sse = 0f32;
    let mut sst = 0f32;

    for i in 0..n {
        let _sse = y[i] - (intercept + (slope * x[i]));
        let _sst = y[i] - mean_y;
        sse += _sse * _sse;
        sst += _sst * _sst;
    }
    let res = LinearRegressionResult {
        slope,
        intercept,
        r2: 1f32 - (sse / sst),
    };
    return std::mem::transmute::<Box<LinearRegressionResult>, isize>(std::boxed::Box::new(res)) as isize;
}
