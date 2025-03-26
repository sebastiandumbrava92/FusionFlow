// fusionflow_backend.h
#ifndef FUSIONFLOW_BACKEND_H
#define FUSIONFLOW_BACKEND_H

// --- 1. Standard Includes ---
// MUST come first to define basic types used below
#include <stddef.h> // Defines size_t
#include <stdbool.h> // Defines bool, true, false (C99+)
#include <stdint.h> // Defines int32_t, int64_t, etc. (C99+)

// --- 2. Enums ---

// Supported Data Types
typedef enum {
    FF_FLOAT32,
    FF_FLOAT64,
    FF_INT32,
    FF_INT64,
    FF_BOOL
} FFDataType;

// Supported Devices
typedef enum {
    FF_CPU,
    FF_GPU_CUDA, // Placeholder
    FF_GPU_ROCM  // Placeholder
} FFDevice;

// --- 3. Struct Forward Declaration (Optional but good practice) ---
struct FFTensor;

// --- 4. Struct Definition ---
typedef struct FFTensor {
    void* data;
    size_t* shape;
    size_t* strides;
    size_t ndim;
    FFDataType dtype;
    FFDevice device;
    size_t size;
    size_t nbytes;
    int ref_count;
    bool requires_grad;
    struct FFTensor* grad; // Uses forward-declared struct name okay here
} FFTensor; // Typedef name defined *after* struct definition

// --- 5. Function Prototypes (Now all types are known) ---

// Utility Functions
size_t ff_dtype_size(FFDataType dtype);
size_t ff_calculate_size(const size_t* shape, size_t ndim);
void ff_calculate_contiguous_strides(const size_t* shape, size_t ndim, size_t element_size, size_t* out_strides);
int ff_tensor_fill_with_scalar_div(FFTensor* tensor, const FFTensor* scalar_tensor, size_t divisor);

// Tensor Lifecycle & State
FFTensor* ff_tensor_create(const size_t* shape, size_t ndim, FFDataType dtype, FFDevice device, bool requires_grad);
FFTensor* ff_tensor_create_from_data(const void* host_data, const size_t* shape, size_t ndim, FFDataType dtype, FFDevice device, bool requires_grad);
int ff_tensor_copy_from_host(FFTensor* tensor, const void* host_data);
int ff_tensor_fill(FFTensor* tensor, double value);
void ff_tensor_retain(FFTensor* tensor);
void ff_tensor_release(FFTensor* tensor);
int ff_tensor_ensure_zero_grad(FFTensor* tensor);
int ff_tensor_zero_data(FFTensor* tensor);
FFTensor* ff_tensor_copy(const FFTensor* source);
FFTensor* ff_tensor_ones(const size_t* shape, size_t ndim, FFDataType dtype, FFDevice device, bool requires_grad);
FFTensor* ff_tensor_eye(size_t dim, FFDataType dtype, FFDevice device, bool requires_grad);
FFTensor* ff_tensor_uniform(double low, double high, const size_t* shape, size_t ndim, FFDataType dtype, FFDevice device, bool requires_grad);
FFTensor* ff_tensor_astype(const FFTensor* source, FFDataType new_dtype);
FFTensor* ff_tensor_transpose(const FFTensor* tensor); // Basic 2D

// Forward Operations
FFTensor* ff_tensor_add(const FFTensor* a, const FFTensor* b);
FFTensor* ff_tensor_sub(const FFTensor* a, const FFTensor* b);
FFTensor* ff_tensor_mul(const FFTensor* a, const FFTensor* b);
FFTensor* ff_tensor_matmul(const FFTensor* a, const FFTensor* b);
FFTensor* ff_tensor_pow_scalar(const FFTensor* tensor, double value);
FFTensor* ff_tensor_mean(const FFTensor* tensor);
FFTensor* ff_tensor_mul_scalar(const FFTensor* tensor, double value);
FFTensor* ff_tensor_div_scalar(const FFTensor* tensor, double value);
FFTensor* ff_tensor_rdiv_scalar(double value, const FFTensor* tensor);
FFTensor* ff_tensor_add_scalar(const FFTensor* tensor, double value);
FFTensor* ff_tensor_tanh(const FFTensor* tensor);
FFTensor* ff_tensor_exp(const FFTensor* tensor);
FFTensor* ff_tensor_sigmoid(const FFTensor* tensor);
FFTensor* ff_tensor_relu(const FFTensor* tensor);
FFTensor* ff_tensor_abs(const FFTensor* tensor);
FFTensor* ff_tensor_clip(const FFTensor* tensor, double min_val, double max_val);
FFTensor* ff_tensor_gt_scalar(const FFTensor* tensor, double value);
FFTensor* ff_tensor_lt_scalar(const FFTensor* tensor, double value);
FFTensor* ff_tensor_outer(const FFTensor* vec_a, const FFTensor* vec_b);
FFTensor* ff_tensor_sign(const FFTensor* tensor);

// Autograd Backward Kernels
int ff_tensor_add_backward(FFTensor* grad_output, FFTensor* input_a, FFTensor* input_b);
int ff_tensor_sub_backward(FFTensor* grad_output, FFTensor* input_a, FFTensor* input_b);
int ff_tensor_mul_backward(FFTensor* grad_output, FFTensor* input_a, FFTensor* input_b);
int ff_tensor_matmul_backward(FFTensor* grad_output, FFTensor* input_a, FFTensor* input_b);
int ff_tensor_pow_scalar_backward(FFTensor* grad_output, FFTensor* input_a, double exponent);
int ff_tensor_mean_backward(FFTensor* grad_output, FFTensor* input_a);
int ff_tensor_transpose_backward(FFTensor* grad_output, FFTensor* input_a);
int ff_tensor_mul_scalar_backward(FFTensor* grad_output, FFTensor* input_a, double scalar_value);
int ff_tensor_div_scalar_backward(FFTensor* grad_output, FFTensor* input_a, double scalar_value);
int ff_tensor_rdiv_scalar_backward(FFTensor* grad_output, FFTensor* input_a, double scalar_value, FFTensor* forward_output);
int ff_tensor_add_scalar_backward(FFTensor* grad_output, FFTensor* input_a);
int ff_tensor_tanh_backward(FFTensor* grad_output, FFTensor* input_a, FFTensor* forward_output);
int ff_tensor_exp_backward(FFTensor* grad_output, FFTensor* input_a, FFTensor* forward_output);
int ff_tensor_sigmoid_backward(FFTensor* grad_output, FFTensor* input_a, FFTensor* forward_output);
int ff_tensor_relu_backward(FFTensor* grad_output, FFTensor* input_a);
int ff_tensor_abs_backward(FFTensor* grad_output, FFTensor* input_a);
int ff_tensor_clip_backward(FFTensor* grad_output, FFTensor* input_a, double min_val, double max_val);
int ff_tensor_outer_backward(FFTensor* grad_output, FFTensor* input_a, FFTensor* input_b);
// Add prototype for astype backward if/when implemented
// int ff_tensor_astype_backward(FFTensor* grad_output, FFTensor* input_a);


// Optimizer Kernels
int ff_optim_sgd_step(FFTensor* param, const FFTensor* grad, double lr);


#endif // FUSIONFLOW_BACKEND_H
