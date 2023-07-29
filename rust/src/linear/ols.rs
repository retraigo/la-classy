#[no_mangle]
pub unsafe extern "C" fn ordinary_least_squares(
    x_ptr: *const f64,
    y_ptr: *const f64,
    x_len: usize,
    y_len: usize,
    res_ptr: *mut f64,
) {
    let x = std::slice::from_raw_parts(x_ptr, x_len);
    let y = std::slice::from_raw_parts(y_ptr, y_len);
    let n = x_len.min(y_len);
    let num = n as f64;
    let mut mean_x = 0f64;
    let mut mean_y = 0f64;
    for i in 0..n {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x = mean_x / num;
    mean_y = mean_y / num;
    let mut stddev_xy = 0f64;
    let mut stddev_x = 0f64;
    for i in 0..n {
        stddev_xy += ((x[i] - mean_x) * (y[i] - mean_y)) as f64;
        stddev_x += ((x[i] - mean_x).powi(2)) as f64;
    }
    let slope = (stddev_xy / stddev_x) as f64;
    let intercept = mean_y - (slope * mean_x);
    let mut sse = 0f64;
    let mut sst = 0f64;

    for i in 0..n {
        let _sse = y[i] - (intercept + (slope * x[i]));
        let _sst = y[i] - mean_y;
        sse += _sse * _sse;
        sst += _sst * _sst;
    }
    let r2 = 1f64 - (sse / sst);
    let out = std::slice::from_raw_parts_mut(res_ptr, 3);
    out[0] = slope;
    out[1] = intercept;
    out[2] = r2;
}
