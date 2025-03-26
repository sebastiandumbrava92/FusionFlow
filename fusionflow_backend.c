// fusionflow_backend.c
#include "fusionflow_backend.h" // Must be first include
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h> // For rand seed

// --- Macro Definitions ---

// Macro to validate inputs for simple binary element-wise operations (non-broadcasting)
#define VALIDATE_BINARY_OP(op_name_str, a, b) \
do { \
    if (!a || !b) { \
        fprintf(stderr, "Error [%s]: Input tensor is NULL.\n", op_name_str); \
        return NULL; \
    } \
    if (a->dtype != b->dtype) { \
        fprintf(stderr, "Error [%s]: Dtype mismatch (%d vs %d).\n", op_name_str, a->dtype, b->dtype); \
        return NULL; \
    } \
    /* Current backend implementation only supports CPU */ \
    if (a->device != FF_CPU || b->device != FF_CPU) { \
        fprintf(stderr, "Error [%s]: Only CPU supported (got %d, %d).\n", op_name_str, a->device, b->device); \
        return NULL; \
    } \
    if (a->ndim != b->ndim) { \
        fprintf(stderr, "Error [%s]: Dimension mismatch (%zu vs %zu).\n", op_name_str, a->ndim, b->ndim); \
        return NULL; \
    } \
    /* Check actual shape dimensions */ \
    for (size_t i_shape_check = 0; i_shape_check < a->ndim; ++i_shape_check) { \
        if (a->shape[i_shape_check] != b->shape[i_shape_check]) { \
            fprintf(stderr, "Error [%s]: Shape mismatch at dim %zu (%zu vs %zu).\n", op_name_str, i_shape_check, a->shape[i_shape_check], b->shape[i_shape_check]); \
            return NULL; \
        } \
    } \
    /* Check if data pointers are valid if size > 0 (shape match implies size match) */ \
    if (a->size > 0 && (!a->data || !b->data)) { \
         fprintf(stderr, "Error [%s]: Input data pointer is NULL for non-empty tensor.\n", op_name_str); \
         return NULL; \
    } \
} while (0)


// --- Static Helper Prototypes ---
static int accumulate_grad(FFTensor* target_grad, const FFTensor* grad_contrib);
// Add more helpers if needed...

// --- Static Helper Implementations ---

// Helper for gradient accumulation (adds grad_contrib to target_grad->data)
// Assumes same shape/dtype, CPU. Needs stride handling for non-contiguous.
static int accumulate_grad(FFTensor* target_grad, const FFTensor* grad_contrib) {
    if (!target_grad || !grad_contrib) { fprintf(stderr,"accumulate_grad: NULL input\n"); return -1; }
    if (!target_grad->data && target_grad->size > 0) { fprintf(stderr,"accumulate_grad: NULL target data pointer\n"); return -1; }
    if (!grad_contrib->data && grad_contrib->size > 0) { fprintf(stderr,"accumulate_grad: NULL contrib data pointer\n"); return -1; }
    if (target_grad->size != grad_contrib->size) { fprintf(stderr,"accumulate_grad: Size mismatch %zu vs %zu\n", target_grad->size, grad_contrib->size); return -1; }
    if (target_grad->dtype != grad_contrib->dtype) { fprintf(stderr,"accumulate_grad: Dtype mismatch\n"); return -1;}
    if (target_grad->size == 0) { return 0; } // Nothing to accumulate

    size_t n_elements = target_grad->size;
    // TODO: Implement using strides for non-contiguous memory
    switch (target_grad->dtype) {
        case FF_FLOAT32: { float* t = (float*)target_grad->data; float* c = (float*)grad_contrib->data; for(size_t i=0; i<n_elements; ++i) t[i] += c[i]; break; }
        case FF_FLOAT64: { double* t = (double*)target_grad->data; double* c = (double*)grad_contrib->data; for(size_t i=0; i<n_elements; ++i) t[i] += c[i]; break; }
        default: fprintf(stderr,"accumulate_grad: Unsupported dtype\n"); return -1;
    }
    return 0;
}

// Helper to broadcast/fill a tensor with a scalar value divided by size
int ff_tensor_fill_with_scalar_div(FFTensor* tensor, const FFTensor* scalar_tensor, size_t divisor) {
    if (!tensor || !scalar_tensor) { fprintf(stderr,"fill_scalar_div: NULL input\n"); return -1; }
    if (scalar_tensor->size != 1) { fprintf(stderr,"fill_scalar_div: scalar input wrong size\n"); return -1; }
    if (divisor == 0) { fprintf(stderr,"fill_scalar_div: Divisor is zero\n"); return -1; }
    if (tensor->dtype != scalar_tensor->dtype) { fprintf(stderr,"fill_scalar_div: Dtype mismatch\n"); return -1; }
    if (!scalar_tensor->data) { fprintf(stderr,"fill_scalar_div: Scalar tensor data is NULL\n"); return -1; }

    double value = 0;
    switch(scalar_tensor->dtype) {
        case FF_FLOAT32: value = (double)(*(float*)scalar_tensor->data); break;
        case FF_FLOAT64: value = *(double*)scalar_tensor->data; break;
        default: fprintf(stderr, "Error [ff_tensor_fill_with_scalar_div]: Unsupported scalar type.\n"); return -1;
    }
    value /= (double)divisor; // Perform division

    return ff_tensor_fill(tensor, value);
}


// --- Utility Functions Implementation --- //
size_t ff_dtype_size(FFDataType dtype) {
    switch (dtype) {
        case FF_FLOAT32: return sizeof(float); case FF_FLOAT64: return sizeof(double);
        case FF_INT32: return sizeof(int32_t); case FF_INT64: return sizeof(int64_t);
        case FF_BOOL: return sizeof(bool); default: return 0;
    }
}
size_t ff_calculate_size(const size_t* shape, size_t ndim) {
    // Fixed indentation
    if (ndim == 0) { return 1; }
    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i) { if (shape[i] == 0) return 0; size *= shape[i]; }
    return size;
}
void ff_calculate_contiguous_strides(const size_t* shape, size_t ndim, size_t element_size, size_t* out_strides) {
    // Fixed indentation
    if (ndim == 0) { return; }
    size_t stride_val = element_size;
    for (int i = (int)ndim - 1; i >= 0; --i) {
        out_strides[i] = stride_val;
        if (shape[i] > 0) { stride_val *= shape[i]; } else { stride_val = 0; } // Handle 0-dim shape
    }
}


// --- Tensor Lifecycle & State Implementation --- //
FFTensor* ff_tensor_create(const size_t* shape, size_t ndim, FFDataType dtype, FFDevice device, bool requires_grad) {
    if (!shape && ndim > 0) { fprintf(stderr, "Error [ff_tensor_create]: Shape NULL for ndim > 0.\n"); return NULL; }
    if (device != FF_CPU) { fprintf(stderr, "Error [ff_tensor_create]: Only CPU device supported currently.\n"); return NULL; }

    size_t element_size = ff_dtype_size(dtype);
    if (element_size == 0) { fprintf(stderr, "Error [ff_tensor_create]: Unsupported dtype %d.\n", dtype); return NULL; }

    FFTensor* tensor = (FFTensor*)malloc(sizeof(FFTensor));
    if (!tensor) { perror("malloc FFTensor struct failed"); return NULL; }

    // Initialize fields
    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->device = device;
    tensor->requires_grad = requires_grad;
    tensor->grad = NULL;
    tensor->ref_count = 1; // Start with ref count 1
    tensor->shape = NULL;
    tensor->strides = NULL;
    tensor->data = NULL;

    if (ndim > 0) {
        tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
        if (!tensor->shape) { perror("malloc shape failed"); free(tensor); return NULL; }
        memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    }

    tensor->size = ff_calculate_size(tensor->shape, ndim);
    tensor->nbytes = tensor->size * element_size;

    if (tensor->nbytes > 0) {
        // Use calloc for zero-initialization, important for gradients
        tensor->data = calloc(tensor->size, element_size);
        if (!tensor->data) {
            perror("calloc data failed");
            free(tensor->shape);
            free(tensor);
            return NULL;
        }
    } // else: data remains NULL for 0-byte tensors

    if (ndim > 0) {
        tensor->strides = (size_t*)malloc(ndim * sizeof(size_t));
        if (!tensor->strides) {
            perror("malloc strides failed");
            free(tensor->data); // data might be NULL if nbytes is 0
            free(tensor->shape);
            free(tensor);
            return NULL;
        }
        ff_calculate_contiguous_strides(tensor->shape, ndim, element_size, tensor->strides);
    }

    return tensor;
}

FFTensor* ff_tensor_create_from_data(const void* host_data, const size_t* shape, size_t ndim, FFDataType dtype, FFDevice device, bool requires_grad) {
    FFTensor* tensor = ff_tensor_create(shape, ndim, dtype, device, requires_grad);
    if (!tensor) { return NULL; }

    if (!host_data && tensor->nbytes > 0) { // Check if host_data is needed
        fprintf(stderr, "Error [create_from_data]: host_data is NULL for non-empty tensor.\n");
        ff_tensor_release(tensor); // Release the created tensor
        return NULL;
    }

    if (tensor->nbytes > 0) {
        if (tensor->data) { // Should always be true if nbytes > 0 after ff_tensor_create
            memcpy(tensor->data, host_data, tensor->nbytes);
        } else {
            // This case should ideally not happen if ff_tensor_create is correct
            fprintf(stderr, "Error [create_from_data]: Internal error, data NULL after create for non-empty tensor.\n");
            ff_tensor_release(tensor);
            return NULL;
        }
    }
    // If nbytes is 0, no copy needed.

    return tensor;
}

int ff_tensor_copy_from_host(FFTensor* tensor, const void* host_data) {
    // Fixed indentation and logic flow
    if (!tensor || !host_data) { return -1; }
    if (tensor->device != FF_CPU) { return -1; } // Only CPU supported for host copy

    if (tensor->nbytes > 0) {
        if (tensor->data != NULL) {
            memcpy(tensor->data, host_data, tensor->nbytes);
        } else {
             // Error: Tensor claims to have bytes but data pointer is NULL
             fprintf(stderr, "Error [copy_from_host]: Tensor has non-zero bytes but NULL data pointer.\n");
             return -1;
        }
    }
    // else if (tensor->nbytes == 0): No-op is correct, data can be NULL

    return 0; // Return 0 on success
}

int ff_tensor_fill(FFTensor* tensor, double value) {
    // Fixed indentation for checks
    if (!tensor) { return -1; }
    if (tensor->device != FF_CPU) { return -1; } // Only CPU supported
    if (tensor->nbytes == 0) { return 0; } // Nothing to fill for empty tensor

    size_t n = tensor->size;
    void* p = tensor->data;
    if (!p) {
         fprintf(stderr, "Error [ff_tensor_fill]: Tensor has non-zero bytes but NULL data pointer.\n");
         return -1; // Should not happen if nbytes > 0
    }

    // TODO: Implement using strides for non-contiguous memory
    switch (tensor->dtype) {
        case FF_FLOAT32: { float v = (float)value; float* pp = (float*)p; for (size_t i = 0; i < n; ++i) pp[i] = v; break; }
        case FF_FLOAT64: { double* pp = (double*)p; for (size_t i = 0; i < n; ++i) pp[i] = value; break; }
        case FF_INT32:   { int32_t v = (int32_t)value; int32_t* pp = (int32_t*)p; for (size_t i = 0; i < n; ++i) pp[i] = v; break; }
        case FF_INT64:   { int64_t v = (int64_t)value; int64_t* pp = (int64_t*)p; for (size_t i = 0; i < n; ++i) pp[i] = v; break; }
        case FF_BOOL:    { bool v = (bool)(value != 0.0); bool* pp = (bool*)p; for (size_t i = 0; i < n; ++i) pp[i] = v; break; }
        default: fprintf(stderr, "Error [ff_tensor_fill]: Unsupported dtype %d.\n", tensor->dtype); return -1;
    }
    return 0;
}

void ff_tensor_retain(FFTensor* tensor) {
    if (tensor) {
        tensor->ref_count++;
    }
}

void ff_tensor_release(FFTensor* tensor) {
    // Fixed indentation
    if (!tensor) { return; }

    // Decrement and check ref count
    int current_ref_count = --(tensor->ref_count);

    if (current_ref_count <= 0) {
        // Recursively release gradient tensor BEFORE freeing main tensor's components
        if (tensor->grad) {
            ff_tensor_release(tensor->grad);
            tensor->grad = NULL; // Avoid double free if somehow referenced again
        }
        // Free components
        free(tensor->data);    // free(NULL) is safe
        free(tensor->shape);   // free(NULL) is safe
        free(tensor->strides); // free(NULL) is safe
        // Free the struct itself
        free(tensor);
    } else if (current_ref_count < 0) {
        // This indicates a bug somewhere (e.g., over-release)
        fprintf(stderr, "Warning: Tensor %p ref_count fell below zero (%d).\n", (void*)tensor, current_ref_count);
        // Avoid freeing memory here as the state is undefined
    }
}

int ff_tensor_ensure_zero_grad(FFTensor* tensor) {
    if (!tensor || !tensor->requires_grad) { return 0; } // No grad needed or tensor invalid

    if (!tensor->grad) {
        // Create gradient tensor (same shape/dtype/device, but not requiring grad itself)
        FFTensor* grad_tensor = ff_tensor_create(tensor->shape, tensor->ndim, tensor->dtype, tensor->device, false);
        if (!grad_tensor) {
            fprintf(stderr, "Error [ensure_zero_grad]: Failed to allocate gradient tensor.\n");
            return -1;
        }
        // Note: ff_tensor_create already zero-initializes the data via calloc.
        tensor->grad = grad_tensor;
    } else {
        // Gradient tensor already exists, just zero its data
        if (ff_tensor_zero_data(tensor->grad) != 0) {
            fprintf(stderr, "Error [ensure_zero_grad]: Failed zero existing gradient data.\n");
            return -1;
        }
    }
    return 0;
}

int ff_tensor_zero_data(FFTensor* tensor) {
    // Fixed indentation and logic
    if (!tensor) { return -1; }

    if (tensor->nbytes > 0) {
        if (tensor->data) {
            memset(tensor->data, 0, tensor->nbytes);
            return 0;
        } else {
            fprintf(stderr, "Error [ff_tensor_zero_data]: Tensor has non-zero bytes but NULL data pointer.\n");
            return -1;
        }
    }
    // If nbytes is 0, it's already effectively zeroed.
    return 0;
}

FFTensor* ff_tensor_copy(const FFTensor* source) {
    // Fixed indentation
    if (!source) { return NULL; }

    FFTensor* result = ff_tensor_create(source->shape, source->ndim, source->dtype, source->device, source->requires_grad);
    if (!result) { return NULL; }

    if (source->nbytes > 0) {
        // Both source and result data pointers should be valid if nbytes > 0
        if (source->data && result->data) {
            memcpy(result->data, source->data, source->nbytes);
        } else {
            fprintf(stderr, "Error [ff_tensor_copy]: Data pointer NULL for non-empty tensor copy (src:%p, dst:%p).\n", (void*)source->data, (void*)result->data);
            ff_tensor_release(result); // Clean up allocated result
            return NULL;
        }
    }
    // If nbytes is 0, no copy needed.

    return result;
}

FFTensor* ff_tensor_ones(const size_t* shape, size_t ndim, FFDataType dtype, FFDevice device, bool requires_grad) {
    FFTensor* tensor = ff_tensor_create(shape, ndim, dtype, device, requires_grad);
    if (!tensor) { return NULL; }
    if (ff_tensor_fill(tensor, 1.0) != 0) {
        ff_tensor_release(tensor); // Clean up if fill fails
        return NULL;
    }
    return tensor;
}

FFTensor* ff_tensor_eye(size_t dim, FFDataType dtype, FFDevice device, bool requires_grad) {
    size_t shape[2] = {dim, dim};
    FFTensor* tensor = ff_tensor_create(shape, 2, dtype, device, requires_grad);

    // Fixed indentation and checks
    if (!tensor || dim == 0) { return tensor; } // Return if no tensor or 0 dim

    char* data_ptr = (char*)tensor->data;
    // Check data pointer *after* potential early return, only error if size > 0
    if (!data_ptr && tensor->size > 0) {
         fprintf(stderr,"Eye Error: NULL data pointer for non-empty tensor\n");
         ff_tensor_release(tensor);
         return NULL;
    }
    if (!data_ptr) { return tensor; } // If size is 0, data_ptr can be NULL, okay

    // Strides should exist if ndim=2
    if (!tensor->strides) {
        fprintf(stderr,"Eye Error: Strides pointer is NULL for 2D tensor\n");
        ff_tensor_release(tensor);
        return NULL;
    }

    // Initialize diagonal (assumes contiguous or correctly calculated strides)
    // TODO: Verify stride usage is correct if non-contiguous is ever supported
    for (size_t i = 0; i < dim; ++i) {
        // Calculate offset using strides: row_idx * stride[0] + col_idx * stride[1]
        size_t offset = i * tensor->strides[0] + i * tensor->strides[1];
        switch(dtype){
            case FF_FLOAT32: *(float*) (data_ptr + offset) = 1.0f; break;
            case FF_FLOAT64: *(double*)(data_ptr + offset) = 1.0; break;
            case FF_INT32:   *(int32_t*)(data_ptr + offset)= 1; break;
            case FF_INT64:   *(int64_t*)(data_ptr + offset)= 1; break;
            case FF_BOOL:    *(bool*) (data_ptr + offset) = true; break;
            default:
                fprintf(stderr,"Eye Error: Unsupported dtype %d\n", dtype);
                ff_tensor_release(tensor);
                return NULL;
        }
    }
    return tensor;
}

static bool _rand_seeded = false;
static void ensure_rand_seed() {
    if (!_rand_seeded) {
        srand((unsigned int)time(NULL));
        _rand_seeded = true;
    }
}

FFTensor* ff_tensor_uniform(double low, double high, const size_t* shape, size_t ndim, FFDataType dtype, FFDevice device, bool requires_grad) {
    ensure_rand_seed();
    FFTensor* tensor = ff_tensor_create(shape, ndim, dtype, device, requires_grad);

    // Check tensor creation and size
    if (!tensor) { return NULL; }
    if (tensor->size == 0) { return tensor; } // Return early for empty tensors

    size_t n_elements = tensor->size;
    void* data_ptr = tensor->data;
    if (!data_ptr) { // Should not happen if size > 0
        fprintf(stderr, "Error [ff_tensor_uniform]: NULL data pointer for non-empty tensor.\n");
        ff_tensor_release(tensor);
        return NULL;
    }

    double range = high - low;
    // TODO: Use strides for non-contiguous
    switch (dtype) {
        case FF_FLOAT32: { float* p=(float*)data_ptr; for(size_t i=0; i<n_elements; ++i) p[i] = (float)(low + range * ((double)rand()/RAND_MAX)); break; }
        case FF_FLOAT64: { double* p=(double*)data_ptr; for(size_t i=0; i<n_elements; ++i) p[i] = low + range * ((double)rand()/RAND_MAX); break; }
        default:
            fprintf(stderr, "Error [ff_tensor_uniform]: Unsupported dtype %d for uniform distribution.\n", dtype);
            ff_tensor_release(tensor);
            return NULL;
    }
    return tensor;
}

FFTensor* ff_tensor_transpose(const FFTensor* tensor) {
    // Fixed indentation and added error messages
    if (!tensor) { return NULL; }
    if (tensor->ndim != 2) {
        fprintf(stderr, "Error [transpose]: Only 2D tensors supported (got %zu dimensions).\n", tensor->ndim);
        return NULL;
    }
    if (tensor->device != FF_CPU) {
         fprintf(stderr, "Error [transpose]: Only CPU tensors supported.\n");
         return NULL;
    }
    // Check strides exist
    if (!tensor->strides) {
        fprintf(stderr, "Error [transpose]: Source tensor strides pointer is NULL.\n");
        return NULL;
    }


    size_t new_shape[2] = {tensor->shape[1], tensor->shape[0]};
    bool requires_grad = tensor->requires_grad; // Inherit requires_grad flag

    FFTensor* result = ff_tensor_create(new_shape, 2, tensor->dtype, FF_CPU, requires_grad);
    if (!result) { return NULL; }
    // Strides should exist now
    if (!result->strides) {
         fprintf(stderr, "Error [transpose]: Result tensor strides pointer is NULL.\n");
         ff_tensor_release(result);
         return NULL;
    }


    size_t M=tensor->shape[0], N=tensor->shape[1];
    size_t E=ff_dtype_size(tensor->dtype); // Element size in bytes

    char* src=(char*)tensor->data;
    char* dst=(char*)result->data;

    // Check data pointers only if size > 0
    if (tensor->size > 0 && (!src || !dst)) {
        fprintf(stderr, "Error [transpose]: Data pointer NULL for non-empty tensor (src:%p, dst:%p).\n", (void*)src, (void*)dst);
        ff_tensor_release(result);
        return NULL;
    }
    if (result->size == 0) { return result; } // Return early if empty

    size_t s_s0=tensor->strides[0], s_s1=tensor->strides[1]; // Source strides
    size_t d_s0=result->strides[0], d_s1=result->strides[1]; // Dest strides

    // Perform transpose using strides for element access
    for (size_t i=0; i<M; ++i) { // Iterate rows of source
        for (size_t j=0; j<N; ++j) { // Iterate cols of source
            // Source element at [i, j]: src + i*s_s0 + j*s_s1
            // Dest element at [j, i]: dst + j*d_s0 + i*d_s1
            memcpy(dst + j*d_s0 + i*d_s1, src + i*s_s0 + j*s_s1, E);
        }
    }
    return result;
}

// Corrected CAST_LOOP macro definition (was okay before)
#define CAST_LOOP(SRC_TYPE, DST_TYPE) \
    { SRC_TYPE* src = (SRC_TYPE*)src_ptr_v; DST_TYPE* dst = (DST_TYPE*)dst_ptr_v; \
      for(size_t i=0; i<n_elements; ++i) dst[i] = (DST_TYPE)src[i]; }

FFTensor* ff_tensor_astype(const FFTensor* source, FFDataType new_dtype) {
    if (!source) { return NULL; }

    // Create result tensor, inherit requires_grad
    FFTensor* result = ff_tensor_create(source->shape, source->ndim, new_dtype, source->device, source->requires_grad);
    if (!result) { return NULL; }
    if (source->size == 0) { return result; } // Return early if empty

    size_t n_elements = source->size;
    void* src_ptr_v = source->data;
    void* dst_ptr_v = result->data;
    FFDataType src_dtype = source->dtype;

    if (!src_ptr_v || !dst_ptr_v) { // Data pointers must exist if size > 0
        fprintf(stderr, "Error [astype]: Data pointer NULL for non-empty tensor (src:%p, dst:%p).\n", src_ptr_v, dst_ptr_v);
        ff_tensor_release(result);
        return NULL;
    }

    // TODO: Add stride support if non-contiguous needed
    if (src_dtype == FF_FLOAT32 && new_dtype == FF_FLOAT64) { CAST_LOOP(float, double) }
    else if (src_dtype == FF_FLOAT64 && new_dtype == FF_FLOAT32) { CAST_LOOP(double, float) }
    else if (src_dtype == FF_INT32 && new_dtype == FF_FLOAT32) { CAST_LOOP(int32_t, float) }
    else if (src_dtype == FF_INT32 && new_dtype == FF_FLOAT64) { CAST_LOOP(int32_t, double) }
    else if (src_dtype == FF_INT64 && new_dtype == FF_FLOAT32) { CAST_LOOP(int64_t, float) }
    else if (src_dtype == FF_INT64 && new_dtype == FF_FLOAT64) { CAST_LOOP(int64_t, double) }
    else if (src_dtype == FF_BOOL && new_dtype == FF_FLOAT32) { CAST_LOOP(bool, float) }
    else if (src_dtype == FF_BOOL && new_dtype == FF_INT32) { CAST_LOOP(bool, int32_t) }
    else if (src_dtype == FF_BOOL && new_dtype == FF_FLOAT64) { CAST_LOOP(bool, double) }
    // Add conversions *to* bool
    else if (src_dtype == FF_INT32 && new_dtype == FF_BOOL) { CAST_LOOP(int32_t, bool) }
    else if (src_dtype == FF_FLOAT32 && new_dtype == FF_BOOL) { CAST_LOOP(float, bool) }
    else if (src_dtype == FF_FLOAT64 && new_dtype == FF_BOOL) { CAST_LOOP(double, bool) }
    // Add more combinations as needed...
    else if (src_dtype == new_dtype) { // If same type, just copy
        memcpy(dst_ptr_v, src_ptr_v, source->nbytes);
    }
    else {
        fprintf(stderr, "Error [astype]: Unsupported conversion from dtype %d to %d.\n", src_dtype, new_dtype);
        ff_tensor_release(result);
        #undef CAST_LOOP // Undefine locally if needed, though usually okay at file scope
        return NULL;
    }
    #undef CAST_LOOP // Undefine locally if needed
    return result;
}


// --- Basic Operations Implementation --- //

// --- Modified ff_tensor_add with basic broadcasting ---
FFTensor* ff_tensor_add(const FFTensor* a, const FFTensor* b) {
    // Basic NULL checks
    if (!a || !b) { fprintf(stderr, "Error [add]: Input tensor is NULL.\n"); return NULL; }
    if (a->dtype != b->dtype) { fprintf(stderr, "Error [add]: Dtype mismatch (%d vs %d).\n", a->dtype, b->dtype); return NULL; }
    if (a->device != FF_CPU || b->device != FF_CPU) { fprintf(stderr, "Error [add]: Only CPU supported.\n"); return NULL; }

    bool broadcast_b = false;
    const size_t* result_shape = NULL; // Initialize to NULL
    size_t result_ndim = 0;            // Initialize to 0

    // Determine result shape and check compatibility
    if (a->ndim == b->ndim) {
        // Check if shapes are identical
        bool shapes_match = true;
        for (size_t i = 0; i < a->ndim; ++i) {
            if (a->shape[i] != b->shape[i]) {
                 shapes_match = false; break;
            }
        }
        if (!shapes_match) {
            fprintf(stderr, "Error [add]: Shape mismatch for same ndim (%zu).\n", a->ndim);
            // Add more detail about mismatch if needed
            return NULL;
        }
        result_shape = a->shape; // Shapes are identical
        result_ndim = a->ndim;
        broadcast_b = false;
    } else if (a->ndim == 2 && b->ndim == 1 && a->shape[1] == b->shape[0]) {
        // Broadcasting b: (M, N) + (N,) -> (M, N)
        broadcast_b = true;
        result_shape = a->shape;
        result_ndim = a->ndim;
    } else if (b->ndim == 2 && a->ndim == 1 && b->shape[1] == a->shape[0]) {
        // Broadcasting a: (N,) + (M, N) -> swap order call (M,N) + (N,)
        return ff_tensor_add(b, a); // Recursion handles broadcasting b
    } else {
        fprintf(stderr, "Error [add]: Incompatible shapes for addition/broadcasting (%zu-D vs %zu-D).\n", a->ndim, b->ndim);
        return NULL;
    }

    // Check data pointers if tensors are not empty
     if (a->size > 0 && !a->data) { fprintf(stderr, "Error [add]: Tensor 'a' data pointer is NULL for non-empty tensor.\n"); return NULL; }
     if (b->size > 0 && !b->data) { fprintf(stderr, "Error [add]: Tensor 'b' data pointer is NULL for non-empty tensor.\n"); return NULL; }

    // Determine if result requires gradient
    bool requires_grad = a->requires_grad || b->requires_grad;

    // Create result tensor
    FFTensor* result = ff_tensor_create(result_shape, result_ndim, a->dtype, FF_CPU, requires_grad);
    // Fixed indentation checks
    if (!result) { return NULL; }
    if (result->size == 0) { return result; } // Return early if empty

    void *rd=result->data; // Pointer should be valid if size > 0
    void *ad=a->data;
    void *bd=b->data;
    if (!rd) { // Should not happen, but check anyway
         fprintf(stderr,"Error [add]: Result data NULL after create.\n");
         ff_tensor_release(result); return NULL;
    }

    size_t n_elements = result->size;
    // Assume rows/cols only relevant for the specific broadcast case
    size_t rows = (result_ndim == 2) ? result_shape[0] : 1;
    size_t cols = (result_ndim >= 1) ? result_shape[result_ndim-1] : 1;

    // TODO: Use strides for general broadcasting and non-contiguous memory
    switch (a->dtype) {
        case FF_FLOAT32: {
            float* rr=(float*)rd; float* aa=(float*)ad; float* bb=(float*)bd;
            if (!broadcast_b) { // Element-wise addition
                for (size_t i=0; i<n_elements; ++i) { rr[i]=aa[i]+bb[i]; }
            } else { // Broadcasting b: (M, N) + (N,)
                 if (cols != b->size) { // Sanity check
                      fprintf(stderr,"Error [add broadcast]: Column size mismatch.\n");
                      ff_tensor_release(result); return NULL;
                 }
                for(size_t i=0; i<rows; ++i) {
                    for(size_t j=0; j<cols; ++j) {
                        // Assumes contiguous memory for simplicity
                        rr[i*cols+j] = aa[i*cols+j] + bb[j];
                    }
                }
            }
            break;
        }
        case FF_FLOAT64: {
            double* rr=(double*)rd; double* aa=(double*)ad; double* bb=(double*)bd;
             if (!broadcast_b) { // Element-wise addition
                for (size_t i=0; i<n_elements; ++i) { rr[i]=aa[i]+bb[i]; }
            } else { // Broadcasting b: (M, N) + (N,)
                 if (cols != b->size) { // Sanity check
                      fprintf(stderr,"Error [add broadcast]: Column size mismatch.\n");
                      ff_tensor_release(result); return NULL;
                 }
                for(size_t i=0; i<rows; ++i) {
                    for(size_t j=0; j<cols; ++j) {
                         // Assumes contiguous memory for simplicity
                        rr[i*cols+j] = aa[i*cols+j] + bb[j];
                    }
                }
            }
            break;
        }
        default:
            fprintf(stderr,"Error [add]: Unsupported dtype %d.\n", a->dtype);
            ff_tensor_release(result);
            return NULL;
    }
    return result;
}

FFTensor* ff_tensor_sub(const FFTensor* a, const FFTensor* b) {
    // Use macro for validation (checks NULL, dtype, device, shape)
    VALIDATE_BINARY_OP("ff_tensor_sub", a, b);

    bool req_grad = a->requires_grad || b->requires_grad;
    FFTensor* r = ff_tensor_create(a->shape, a->ndim, a->dtype, FF_CPU, req_grad);

    // Fixed indentation checks
    if(!r) { return NULL; }
    if(r->size == 0) { return r; }

    // Data pointers were checked by VALIDATE_BINARY_OP if size > 0
    void *rd=r->data, *ad=a->data, *bd=b->data;
    if (!rd) { // Check result data pointer
        fprintf(stderr,"Error [ff_tensor_sub]: Result data NULL after create.\n");
        ff_tensor_release(r); return NULL;
    }
    size_t n=r->size;

    // TODO: Use strides for non-contiguous memory
    switch(a->dtype){
        case FF_FLOAT32:{float*rr=(float*)rd,*aa=(float*)ad,*bb=(float*)bd; for(size_t i=0;i<n;++i)rr[i]=aa[i]-bb[i];break;}
        case FF_FLOAT64:{double*rr=(double*)rd,*aa=(double*)ad,*bb=(double*)bd; for(size_t i=0;i<n;++i)rr[i]=aa[i]-bb[i];break;}
        default: fprintf(stderr,"Error [ff_tensor_sub]: Unsupported type %d.\n", a->dtype); ff_tensor_release(r); return NULL;
    }
    return r;
}

FFTensor* ff_tensor_mul(const FFTensor* a, const FFTensor* b) {
    // Use macro for validation (checks NULL, dtype, device, shape)
    VALIDATE_BINARY_OP("ff_tensor_mul", a, b);

    bool req_grad = a->requires_grad||b->requires_grad;
    FFTensor* r = ff_tensor_create(a->shape, a->ndim, a->dtype, FF_CPU, req_grad);

    // Fixed indentation checks
    if(!r) { return NULL; }
    if(r->size == 0) { return r; }

    // Data pointers were checked by VALIDATE_BINARY_OP if size > 0
    void *rd=r->data, *ad=a->data, *bd=b->data;
     if (!rd) { // Check result data pointer
        fprintf(stderr,"Error [ff_tensor_mul]: Result data NULL after create.\n");
        ff_tensor_release(r); return NULL;
    }
    size_t n=r->size;

    // TODO: Use strides for non-contiguous memory
    switch(a->dtype){
        case FF_FLOAT32:{float*rr=(float*)rd,*aa=(float*)ad,*bb=(float*)bd; for(size_t i=0;i<n;++i)rr[i]=aa[i]*bb[i];break;}
        case FF_FLOAT64:{double*rr=(double*)rd,*aa=(double*)ad,*bb=(double*)bd; for(size_t i=0;i<n;++i)rr[i]=aa[i]*bb[i];break;}
        default: fprintf(stderr,"Error [ff_tensor_mul]: Unsupported type %d.\n", a->dtype); ff_tensor_release(r); return NULL;
    }
    return r;
}

FFTensor* ff_tensor_mul_scalar(const FFTensor* tensor, double value) {
    // Fixed indentation checks
    if (!tensor) { return NULL; }
    bool requires_grad = tensor->requires_grad; // Define after null check

    FFTensor* r = ff_tensor_create(tensor->shape, tensor->ndim, tensor->dtype, FF_CPU, requires_grad);
    if (!r) { return NULL; }
    if (r->size==0) { return r; }

    size_t n=r->size;
    void* id=tensor->data;
    void* rd=r->data;

    if (!id && n > 0) {fprintf(stderr,"Error [ff_tensor_mul_scalar]: Input data NULL for non-empty tensor.\n"); ff_tensor_release(r);return NULL;}
    if (!rd) {fprintf(stderr,"Error [ff_tensor_mul_scalar]: Result data NULL after create.\n"); ff_tensor_release(r);return NULL;}


    // TODO: Use strides for non-contiguous memory
    switch(tensor->dtype){
        case FF_FLOAT32:{float v=(float)value;float*rr=(float*)rd; float*p=(float*)id; for(size_t i=0;i<n;++i)rr[i]=p[i]*v; break;}
        case FF_FLOAT64:{double*rr=(double*)rd; double*p=(double*)id; for(size_t i=0;i<n;++i)rr[i]=p[i]*value; break;}
        default:fprintf(stderr,"Error [ff_tensor_mul_scalar]: Unsupported type %d.\n", tensor->dtype);ff_tensor_release(r);return NULL;
    }
    return r;
}

FFTensor* ff_tensor_div_scalar(const FFTensor* tensor, double value) {
    if (value == 0.0) {
        fprintf(stderr, "Error [ff_tensor_div_scalar]: Division by zero.\n");
        return NULL;
    }
    // Reuse mul_scalar with reciprocal
    return ff_tensor_mul_scalar(tensor, 1.0 / value);
}

FFTensor* ff_tensor_rdiv_scalar(double value, const FFTensor* tensor) {
     // Fixed indentation checks
    if (!tensor) { return NULL; }
    bool req_grad=tensor->requires_grad; // Define after null check

    FFTensor* r=ff_tensor_create(tensor->shape, tensor->ndim, tensor->dtype, FF_CPU, req_grad);
    if(!r) { return NULL; }
    if(r->size==0) { return r; }

    size_t n=r->size;
    void* id=tensor->data;
    void* rd=r->data;

    if (!id && n > 0) {fprintf(stderr,"Error [ff_tensor_rdiv_scalar]: Input data NULL for non-empty tensor.\n"); ff_tensor_release(r);return NULL;}
    if (!rd) {fprintf(stderr,"Error [ff_tensor_rdiv_scalar]: Result data NULL after create.\n"); ff_tensor_release(r);return NULL;}

    // TODO: Use strides for non-contiguous memory
    // TODO: Handle potential division by zero more gracefully (e.g., return Inf/NaN based on context?)
    switch(tensor->dtype){
        case FF_FLOAT32:{
            float v=(float)value;float*rr=(float*)rd; float*p=(float*)id;
            for(size_t i=0;i<n;++i){if(p[i]==0.0f)rr[i]=NAN;else rr[i]=v/p[i];} // Return NAN on div by zero
            break;
        }
        case FF_FLOAT64:{
            double*rr=(double*)rd; double*p=(double*)id;
            for(size_t i=0;i<n;++i){if(p[i]==0.0)rr[i]=NAN;else rr[i]=value/p[i];} // Return NAN on div by zero
            break;
        }
        default:fprintf(stderr,"Error [ff_tensor_rdiv_scalar]: Unsupported type %d.\n", tensor->dtype);ff_tensor_release(r);return NULL;
    }
    return r;
}

FFTensor* ff_tensor_add_scalar(const FFTensor* tensor, double value) {
     // Fixed indentation checks
    if (!tensor) { return NULL; }
    bool req_grad=tensor->requires_grad; // Define after null check

    FFTensor* r = ff_tensor_create(tensor->shape, tensor->ndim, tensor->dtype, FF_CPU, req_grad);
    if (!r) { return NULL; }
    if(r->size==0) { return r; }

    size_t n=r->size;
    void* id=tensor->data;
    void* rd=r->data;

    if (!id && n > 0) {fprintf(stderr,"Error [ff_tensor_add_scalar]: Input data NULL for non-empty tensor.\n"); ff_tensor_release(r);return NULL;}
    if (!rd) {fprintf(stderr,"Error [ff_tensor_add_scalar]: Result data NULL after create.\n"); ff_tensor_release(r);return NULL;}

    // TODO: Use strides for non-contiguous memory
    switch(tensor->dtype){
        case FF_FLOAT32:{float v=(float)value; float*rr=(float*)rd; float*p=(float*)id; for(size_t i=0;i<n;++i)rr[i]=p[i]+v; break;}
        case FF_FLOAT64:{double*rr=(double*)rd; double*p=(double*)id; for(size_t i=0;i<n;++i)rr[i]=p[i]+value; break;}
        default:fprintf(stderr,"Error [ff_tensor_add_scalar]: Unsupported type %d.\n", tensor->dtype);ff_tensor_release(r);return NULL;
    }
    return r;
}

FFTensor* ff_tensor_pow_scalar(const FFTensor* tensor, double value) {
     // Fixed indentation checks
    if (!tensor) { return NULL; }
    bool requires_grad = tensor->requires_grad; // Define after null check

    FFTensor* result = ff_tensor_create(tensor->shape, tensor->ndim, tensor->dtype, FF_CPU, requires_grad);
    if (!result) { return NULL; }
    if (result->size==0) { return result; }

    size_t n=result->size;
    void* id=tensor->data;
    void* rd=result->data;

    if (!id && n > 0) {fprintf(stderr, "Error [ff_tensor_pow_scalar]: Input data NULL for non-empty tensor.\n"); ff_tensor_release(result);return NULL;}
    if (!rd) {fprintf(stderr, "Error [ff_tensor_pow_scalar]: Result data NULL after create.\n"); ff_tensor_release(result);return NULL;}

    // TODO: Use strides for non-contiguous memory
    switch(tensor->dtype){
        case FF_FLOAT32:{float e=(float)value;float*r=(float*)rd; float*p=(float*)id; for(size_t i=0;i<n;++i)r[i]=powf(p[i],e); break;}
        case FF_FLOAT64:{double*r=(double*)rd; double*p=(double*)id; for(size_t i=0;i<n;++i)r[i]=pow(p[i],value); break;}
        default:fprintf(stderr,"Error [ff_tensor_pow_scalar]: Unsupported type %d.\n", tensor->dtype);ff_tensor_release(result);return NULL;
    }
    return result;
}

FFTensor* ff_tensor_mean(const FFTensor* tensor) {
    // Fixed indentation checks
    if (!tensor) { return NULL; }
    size_t shape[]={}; // Shape for scalar result
    bool req_grad=tensor->requires_grad; // Define after null check

    // Create scalar tensor for the result
    FFTensor* r=ff_tensor_create(shape,0,tensor->dtype,FF_CPU,req_grad);
    if(!r) { return NULL; }
    size_t n=tensor->size; // Define n after r is checked

    if(n==0) {
        // Handle mean of empty tensor - typically results in NaN or error. Let's set to 0 for now.
        // ff_tensor_create already zeros the data via calloc.
        return r;
    }

    void* id=tensor->data;
    void* rd=r->data; // Result data (single element)

    if(!id) {fprintf(stderr,"Error [ff_tensor_mean]: Input data NULL for non-empty tensor.\n"); ff_tensor_release(r);return NULL;}
    if(!rd) {fprintf(stderr,"Error [ff_tensor_mean]: Result data NULL after create.\n"); ff_tensor_release(r);return NULL;}

    // TODO: Use strides for non-contiguous memory
    switch(tensor->dtype){
        case FF_FLOAT32:{
            float s=0.0f; float*p=(float*)id;
            for(size_t i=0;i<n;++i)s+=p[i];
            *(float*)rd=s/(float)n; // Store mean in the scalar result tensor
            break;
        }
        case FF_FLOAT64:{
            double s=0.0; double*p=(double*)id;
            for(size_t i=0;i<n;++i)s+=p[i];
            *(double*)rd=s/(double)n; // Store mean in the scalar result tensor
            break;
        }
        default:fprintf(stderr,"Error [ff_tensor_mean]: Unsupported type %d.\n", tensor->dtype);ff_tensor_release(r);return NULL;
    }
    return r;
}

FFTensor* ff_tensor_matmul(const FFTensor* a, const FFTensor* b) {
    // Fixed indentation checks and added error messages
    if (!a || !b) { return NULL; }
    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "Error [matmul]: Both inputs must be 2D tensors (got %zuD and %zuD).\n", a->ndim, b->ndim);
        return NULL;
    }
    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr, "Error [matmul]: Inner dimensions must match (got %zu and %zu).\n", a->shape[1], b->shape[0]);
        return NULL;
    }
    if (a->dtype != b->dtype) {
        fprintf(stderr, "Error [matmul]: Dtype mismatch (%d vs %d).\n", a->dtype, b->dtype);
        return NULL;
    }
    if (a->device != FF_CPU || b->device != FF_CPU) {
         fprintf(stderr, "Error [matmul]: Only CPU supported.\n");
         return NULL;
    }
    // Check strides pointers
    if (!a->strides || !b->strides) {
         fprintf(stderr, "Error [matmul]: Input strides pointer is NULL.\n");
         return NULL;
    }


    bool requires_grad = a->requires_grad || b->requires_grad;
    FFDataType dtype = a->dtype;
    size_t shape[] = {a->shape[0], b->shape[1]};
    size_t M=shape[0], N=shape[1], K=a->shape[1]; // K is the common dimension

    FFTensor* r = ff_tensor_create(shape, 2, dtype, FF_CPU, requires_grad);
    if (!r) { return NULL; }

    // Check result strides
    if (!r->strides) {
         fprintf(stderr, "Error [matmul]: Result strides pointer is NULL.\n");
         ff_tensor_release(r); return NULL;
    }


    // Check data pointers after checking tensor creation and empty case
    if (M==0 || N==0 || K==0) { return r; } // Handle empty matrix case
    void* ad=a->data;
    void* bd=b->data;
    void* rd=r->data;
    if(!ad || !bd || !rd){ // Check all pointers needed for non-empty case
        fprintf(stderr, "Error [matmul]: Data pointer is NULL (a:%p, b:%p, r:%p).\n", ad, bd, rd);
        ff_tensor_release(r);
        return NULL;
    }


    // TODO: Implement using BLAS or optimized loops with strides
    // Naive implementation assuming contiguous memory for simplicity
    size_t a_stride0 = a->strides[0] / ff_dtype_size(dtype); // elements per row
    size_t a_stride1 = a->strides[1] / ff_dtype_size(dtype); // elements per col step
    size_t b_stride0 = b->strides[0] / ff_dtype_size(dtype);
    size_t b_stride1 = b->strides[1] / ff_dtype_size(dtype);
    size_t r_stride0 = r->strides[0] / ff_dtype_size(dtype);
    size_t r_stride1 = r->strides[1] / ff_dtype_size(dtype);

    // Basic check if strides correspond to contiguous layout for the naive loop
    bool a_contig = (a_stride0 == K && a_stride1 == 1);
    bool b_contig = (b_stride0 == N && b_stride1 == 1);
    bool r_contig = (r_stride0 == N && r_stride1 == 1);

    if (!a_contig || !b_contig || !r_contig) {
        fprintf(stderr, "Warning [matmul]: Naive implementation requires contiguous tensors. Strides indicate non-contiguous.\n");
        // Optionally return NULL or attempt a stride-based loop (more complex)
        // For now, we proceed assuming caller knows it might be slow/incorrect if non-contig
    }


    switch (dtype) {
        case FF_FLOAT32: {
            float* rr=(float*)rd; float* aa=(float*)ad; float* bb=(float*)bd;
            for(size_t i=0; i<M; ++i) { // Result row
                for(size_t j=0; j<N; ++j) { // Result col
                    float s=0.0f;
                    for(size_t k=0; k<K; ++k) { // Inner dimension sum
                         // Using simple indexing assuming contiguous
                         s += aa[i*K + k] * bb[k*N + j];
                    }
                    rr[i*N + j] = s;
                }
            }
            break;
        }
        case FF_FLOAT64: {
            double* rr=(double*)rd; double* aa=(double*)ad; double* bb=(double*)bd;
             for(size_t i=0; i<M; ++i) { // Result row
                for(size_t j=0; j<N; ++j) { // Result col
                    double s=0.0;
                    for(size_t k=0; k<K; ++k) { // Inner dimension sum
                         // Using simple indexing assuming contiguous
                         s += aa[i*K + k] * bb[k*N + j];
                    }
                    rr[i*N + j] = s;
                }
            }
            break;
        }
        default:
            fprintf(stderr, "Error [matmul]: Unsupported dtype %d.\n", dtype);
            ff_tensor_release(r);
            return NULL;
    }
    return r;
}

// --- Activation Functions (Forward) ---
#define DEFINE_UNARY_OP_WITH_MATH_FUNC(func_name, math_func_float, math_func_double) \
FFTensor* func_name(const FFTensor* tensor) { \
    if (!tensor) { fprintf(stderr, "Error [%s]: Input tensor is NULL.\n", #func_name); return NULL; } \
    if (tensor->device != FF_CPU) { fprintf(stderr, "Error [%s]: Only CPU supported.\n", #func_name); return NULL; } \
    bool requires_grad = tensor->requires_grad; \
    FFTensor* result = ff_tensor_create(tensor->shape, tensor->ndim, tensor->dtype, FF_CPU, requires_grad); \
    if (!result) { return NULL; } \
    size_t n_elements = result->size; \
    if (n_elements == 0) { return result; } \
    void* in_ptr_v = tensor->data; \
    void* res_ptr_v = result->data; \
    if (!in_ptr_v || !res_ptr_v) { \
        fprintf(stderr, "Error [%s]: Data pointer NULL for non-empty tensor (in:%p, res:%p).\n", #func_name, in_ptr_v, res_ptr_v); \
        ff_tensor_release(result); return NULL; \
    } \
    /* TODO: Use strides */ \
    switch (tensor->dtype) { \
        case FF_FLOAT32: { float* r=(float*)res_ptr_v; float* p=(float*)in_ptr_v; for(size_t i=0; i<n_elements; ++i) r[i]=math_func_float(p[i]); break; } \
        case FF_FLOAT64: { double* r=(double*)res_ptr_v; double* p=(double*)in_ptr_v; for(size_t i=0; i<n_elements; ++i) r[i]=math_func_double(p[i]); break; } \
        default: fprintf(stderr, "Error [%s]: Unsupported data type %d.\n", #func_name, tensor->dtype); ff_tensor_release(result); return NULL; \
    } \
    return result; \
}

DEFINE_UNARY_OP_WITH_MATH_FUNC(ff_tensor_tanh, tanhf, tanh)
DEFINE_UNARY_OP_WITH_MATH_FUNC(ff_tensor_exp, expf, exp)
DEFINE_UNARY_OP_WITH_MATH_FUNC(ff_tensor_abs, fabsf, fabs)

FFTensor* ff_tensor_sigmoid(const FFTensor* tensor) {
    // Sigmoid(x) = 1 / (1 + exp(-x))
    FFTensor* neg = ff_tensor_mul_scalar(tensor, -1.0);
    if (!neg) { return NULL; }

    FFTensor* exp_neg = ff_tensor_exp(neg);
    ff_tensor_release(neg); // Release intermediate tensor
    if (!exp_neg) { return NULL; }

    FFTensor* one_p_exp = ff_tensor_add_scalar(exp_neg, 1.0);
    ff_tensor_release(exp_neg); // Release intermediate tensor
    if (!one_p_exp) { return NULL; }

    FFTensor* res = ff_tensor_rdiv_scalar(1.0, one_p_exp);
    ff_tensor_release(one_p_exp); // Release intermediate tensor
    // res requires_grad will be set correctly based on input `tensor` by the chain of ops
    return res;
}

FFTensor* ff_tensor_relu(const FFTensor* tensor) {
     // Fixed indentation checks
    if (!tensor) { return NULL; }
    bool req_grad=tensor->requires_grad; // Define after null check

    FFTensor* r = ff_tensor_create(tensor->shape, tensor->ndim, tensor->dtype, FF_CPU, req_grad);
    if (!r) { return NULL; }
    if(r->size==0) { return r; }

    size_t n=r->size;
    void* id=tensor->data;
    void* rd=r->data;

    if (!id && n > 0) {fprintf(stderr,"Error [relu]: Input data NULL for non-empty tensor.\n"); ff_tensor_release(r);return NULL;}
    if (!rd) {fprintf(stderr,"Error [relu]: Result data NULL after create.\n"); ff_tensor_release(r);return NULL;}

    // TODO: Use strides
    switch(tensor->dtype){
        case FF_FLOAT32:{float*rr=(float*)rd; float*p=(float*)id; for(size_t i=0;i<n;++i)rr[i]=(p[i]>0.0f)?p[i]:0.0f; break;}
        case FF_FLOAT64:{double*rr=(double*)rd; double*p=(double*)id; for(size_t i=0;i<n;++i)rr[i]=(p[i]>0.0)?p[i]:0.0; break;}
        default:fprintf(stderr,"Error [relu]: Unsupported type %d.\n", tensor->dtype);ff_tensor_release(r);return NULL;
    }
    return r;
}

FFTensor* ff_tensor_clip(const FFTensor* tensor, double min_val, double max_val) {
    // Fixed indentation checks
    if (!tensor) { return NULL; }
    bool requires_grad = tensor->requires_grad; // Define after null check

    FFTensor* result = ff_tensor_create(tensor->shape, tensor->ndim, tensor->dtype, FF_CPU, requires_grad);
    if (!result) { return NULL; }
    if (result->size == 0) { return result; }

    size_t n = result->size;
    void* id = tensor->data;
    void* rd = result->data;

    if (!id && n > 0) { fprintf(stderr,"Error [clip]: Input data NULL for non-empty tensor.\n"); ff_tensor_release(result); return NULL; }
    if (!rd) { fprintf(stderr,"Error [clip]: Result data NULL after create.\n"); ff_tensor_release(result); return NULL; }

    // TODO: Use strides
    switch (tensor->dtype) {
        case FF_FLOAT32: {
            float min_f=(float)min_val, max_f=(float)max_val;
            float*r=(float*)rd; float*p=(float*)id;
            for(size_t i=0; i<n; ++i){r[i]=(p[i]<min_f)?min_f:((p[i]>max_f)?max_f:p[i]);}
            break;
        }
        case FF_FLOAT64: {
            double*r=(double*)rd; double*p=(double*)id;
            for(size_t i=0; i<n; ++i){r[i]=(p[i]<min_val)?min_val:((p[i]>max_val)?max_val:p[i]);}
            break;
        }
        default: fprintf(stderr, "Error [clip]: Unsupported type %d.\n", tensor->dtype); ff_tensor_release(result); return NULL;
    }
    return result;
}

FFTensor* ff_tensor_gt_scalar(const FFTensor* tensor, double value) {
    if (!tensor) { return NULL; }
    // Result is always BOOL and never requires grad from this op
    FFTensor* result = ff_tensor_create(tensor->shape, tensor->ndim, FF_BOOL, FF_CPU, false);

    // Fixed indentation checks
    if (!result) { return NULL; }
    if (result->size == 0) { return result; }

    size_t n = result->size;
    void* id = tensor->data;
    void* rd = result->data; // This will be bool*

    if (!id && n > 0) { fprintf(stderr,"Error [gt_scalar]: Input data NULL for non-empty tensor.\n"); ff_tensor_release(result); return NULL; }
    if (!rd) { fprintf(stderr,"Error [gt_scalar]: Result data NULL after create.\n"); ff_tensor_release(result); return NULL; }

    bool* r = (bool*)rd;
    // TODO: Use strides
    switch (tensor->dtype) {
        case FF_FLOAT32: { float v=(float)value; float*p=(float*)id; for(size_t i=0;i<n;++i)r[i]=p[i]>v; break; }
        case FF_FLOAT64: { double*p=(double*)id; for(size_t i=0;i<n;++i)r[i]=p[i]>value; break; }
        case FF_INT32: { int32_t v=(int32_t)value; int32_t*p=(int32_t*)id; for(size_t i=0;i<n;++i)r[i]=p[i]>v; break; }
        case FF_INT64: { int64_t v=(int64_t)value; int64_t*p=(int64_t*)id; for(size_t i=0;i<n;++i)r[i]=p[i]>v; break; }
        // Cannot compare bool with double meaningfully in this context
        // case FF_BOOL: ...
        default: fprintf(stderr, "Error [gt_scalar]: Unsupported input type %d for comparison.\n", tensor->dtype); ff_tensor_release(result); return NULL;
    }
    return result;
}

FFTensor* ff_tensor_lt_scalar(const FFTensor* tensor, double value) {
    // Implementation similar to gt_scalar but with '<'
     if (!tensor) { return NULL; }
    // Result is always BOOL and never requires grad from this op
    FFTensor* result = ff_tensor_create(tensor->shape, tensor->ndim, FF_BOOL, FF_CPU, false);

    // Fixed indentation checks
    if (!result) { return NULL; }
    if (result->size == 0) { return result; }

    size_t n = result->size;
    void* id = tensor->data;
    void* rd = result->data; // This will be bool*

    if (!id && n > 0) { fprintf(stderr,"Error [lt_scalar]: Input data NULL for non-empty tensor.\n"); ff_tensor_release(result); return NULL; }
    if (!rd) { fprintf(stderr,"Error [lt_scalar]: Result data NULL after create.\n"); ff_tensor_release(result); return NULL; }

    bool* r = (bool*)rd;
    // TODO: Use strides
    switch (tensor->dtype) {
        case FF_FLOAT32: { float v=(float)value; float*p=(float*)id; for(size_t i=0;i<n;++i)r[i]=p[i]<v; break; }
        case FF_FLOAT64: { double*p=(double*)id; for(size_t i=0;i<n;++i)r[i]=p[i]<value; break; }
        case FF_INT32: { int32_t v=(int32_t)value; int32_t*p=(int32_t*)id; for(size_t i=0;i<n;++i)r[i]=p[i]<v; break; }
        case FF_INT64: { int64_t v=(int64_t)value; int64_t*p=(int64_t*)id; for(size_t i=0;i<n;++i)r[i]=p[i]<v; break; }
        default: fprintf(stderr, "Error [lt_scalar]: Unsupported input type %d for comparison.\n", tensor->dtype); ff_tensor_release(result); return NULL;
    }
    return result;
}

FFTensor* ff_tensor_outer(const FFTensor* vec_a, const FFTensor* vec_b) {
    // Fixed indentation checks and added error messages
    if (!vec_a || !vec_b) { return NULL; }
    if (vec_a->ndim != 1 || vec_b->ndim != 1) {
        fprintf(stderr,"Error [outer]: Both inputs must be 1D tensors (vectors).\n");
        return NULL;
    }
    if (vec_a->dtype != vec_b->dtype) {
         fprintf(stderr,"Error [outer]: Dtype mismatch (%d vs %d).\n", vec_a->dtype, vec_b->dtype);
         return NULL;
    }
    if (vec_a->device != FF_CPU || vec_b->device != FF_CPU) {
        fprintf(stderr,"Error [outer]: Only CPU supported.\n");
        return NULL;
    }

    size_t M=vec_a->size, N=vec_b->size;
    size_t shape[]={M,N};
    bool req_grad=vec_a->requires_grad || vec_b->requires_grad;

    FFTensor* r=ff_tensor_create(shape, 2, vec_a->dtype, FF_CPU, req_grad);
    if (!r) { return NULL; }
    if (M==0 || N==0) { return r; } // Handle empty vector case

    // Check data pointers after handling empty case
    void* ad=vec_a->data;
    void* bd=vec_b->data;
    void* rd=r->data;
    if(!ad || !bd || !rd){
        fprintf(stderr,"Error [outer]: Data pointer NULL (a:%p, b:%p, r:%p).\n", ad, bd, rd);
        ff_tensor_release(r);
        return NULL;
    }

    // TODO: Use strides
    switch(vec_a->dtype){
        case FF_FLOAT32:{
            float*rr=(float*)rd,*aa=(float*)ad,*bb=(float*)bd;
            for(size_t i=0;i<M;++i) { // Iterate elements of vec_a (rows of result)
                for(size_t j=0;j<N;++j) { // Iterate elements of vec_b (cols of result)
                     // Assumes contiguous result: result[i, j] = result[i*N + j]
                     // Assumes contiguous inputs: a[i], b[j]
                    rr[i*N+j]=aa[i]*bb[j];
                }
            }
            break;
        }
        case FF_FLOAT64:{
            double*rr=(double*)rd,*aa=(double*)ad,*bb=(double*)bd;
             for(size_t i=0;i<M;++i) {
                for(size_t j=0;j<N;++j) {
                    rr[i*N+j]=aa[i]*bb[j];
                }
             }
             break;
        }
        default:fprintf(stderr,"Error [outer]: Unsupported type %d.\n", vec_a->dtype);ff_tensor_release(r);return NULL;
    }
    return r;
}

FFTensor* ff_tensor_sign(const FFTensor* tensor) {
    // Sign(x) = (x > 0)*2 - 1 (using bool->int/float conversion)
    if (!tensor) { return NULL; }

    // Temporary tensor for (x > 0)
    FFTensor* gt0 = ff_tensor_gt_scalar(tensor, 0.0);
    if(!gt0){ return NULL; }

    // Convert boolean (x > 0) to the original tensor's dtype (0.0 or 1.0)
    FFTensor* gt0f = ff_tensor_astype(gt0, tensor->dtype);
    ff_tensor_release(gt0); // Release intermediate bool tensor
    if(!gt0f){ return NULL; }

    // Multiply by 2.0 -> (0.0 or 2.0)
    FFTensor* t1 = ff_tensor_mul_scalar(gt0f, 2.0);
    ff_tensor_release(gt0f); // Release intermediate tensor
    if(!t1){ return NULL; }

    // Subtract 1.0 -> (-1.0 or 1.0)
    // Result should not require grad based on sign op definition
    FFTensor* result = ff_tensor_add_scalar(t1, -1.0);
    ff_tensor_release(t1); // Release intermediate tensor
    if (!result) { return NULL; }

    // Explicitly set requires_grad to false for the sign result
    result->requires_grad = false;

    // Note: Need to handle sign(0). This implementation gives sign(0) = -1.
    // If sign(0) = 0 is desired, the logic needs adjustment.

    return result;
}

// --- Autograd Gradient Computation Kernels --- //
// Implementations unchanged from previous full source, assume they exist below...
// ... (ff_tensor_add_backward, ff_tensor_sub_backward, etc.) ...
// --- Autograd Gradient Computation Kernels --- //
// Note: These need careful review for correctness, stride support, and memory management (releasing temps).

int ff_tensor_add_backward(FFTensor* gO, FFTensor* iA, FFTensor* iB){
    if(!gO) { return -1; }
    int r=0;

    // Determine broadcasting scenario (based on forward pass logic)
    // Note: This assumes the shapes were compatible in the first place.
    bool bc_b = (iA && iB && iA->ndim == 2 && iB->ndim == 1 && iA->shape[1] == iB->shape[0]);
    bool bc_a = (iA && iB && iB->ndim == 2 && iA->ndim == 1 && iB->shape[1] == iA->shape[0]);

    if(iA && iA->requires_grad) {
        if(ff_tensor_ensure_zero_grad(iA) != 0) { return -1; }
        if(!bc_a){ // No broadcasting of A, accumulate directly
            if(accumulate_grad(iA->grad, gO) != 0) { r=-1; }
        } else { // A was broadcasted (e.g., (N,) + (M, N)), need to sum gO along axis 0
            fprintf(stderr,"Error [add_backward]: Broadcasting grad accumulation for input 'a' not implemented.\n");
            r=-1; // Mark as error until implemented
            // Implementation would involve creating a temporary tensor of A's shape,
            // summing gO along the appropriate axis into it, and then accumulating.
        }
    }

    if(iB && iB->requires_grad) {
        if(ff_tensor_ensure_zero_grad(iB) != 0) { return -1; }
        if(!bc_b){ // No broadcasting of B, accumulate directly
            if(accumulate_grad(iB->grad, gO) != 0) { r=-1; }
        } else { // B was broadcasted (e.g., (M, N) + (N,)), need to sum gO along axis 0
            if (gO->ndim != 2 || iB->ndim != 1 || gO->shape[1] != iB->shape[0]){
                 fprintf(stderr,"Error [add_backward]: Shape mismatch during broadcast grad calculation for 'b'.\n");
                 return -1; // Should not happen if forward worked
            }
            size_t M=gO->shape[0], N=gO->shape[1]; // N = iB->size
             if(!iB->grad->data || !gO->data){
                 fprintf(stderr,"Error [add_backward]: NULL data pointer during broadcast grad for 'b'.\n");
                 return -1;
             }
            // TODO: Use strides
            switch(iB->dtype) {
                case FF_FLOAT32:{
                    float* gb=(float*)iB->grad->data; float* ggo=(float*)gO->data;
                    for(size_t j=0; j<N; ++j){
                        float s=0.0f;
                        for(size_t i=0; i<M; ++i) { s += ggo[i*N+j]; } // Sum column j
                        gb[j] += s; // Accumulate sum
                    }
                    break;
                }
                case FF_FLOAT64:{
                    double* gb=(double*)iB->grad->data; double* ggo=(double*)gO->data;
                     for(size_t j=0; j<N; ++j){
                        double s=0.0;
                        for(size_t i=0; i<M; ++i) { s += ggo[i*N+j]; } // Sum column j
                        gb[j] += s; // Accumulate sum
                    }
                    break;
                }
                default: fprintf(stderr,"Error [add_backward]: Unsupported type %d for broadcast grad 'b'.\n", iB->dtype); r=-1;
            }
        }
    }
    return r;
}

int ff_tensor_sub_backward(FFTensor* gO, FFTensor* iA, FFTensor* iB){
    if(!gO) { return -1; }
    int r=0;
    FFTensor* ngO = NULL; // For gradient wrt B

    if(iA && iA->requires_grad) {
        if(ff_tensor_ensure_zero_grad(iA) != 0) { return -1; }
        // d(A-B)/dA = 1 -> grad_A += grad_O * 1
        if(accumulate_grad(iA->grad, gO) != 0) { r=-1; }
    }

    if(iB && iB->requires_grad) {
        if(ff_tensor_ensure_zero_grad(iB) != 0) { return -1; }
        // d(A-B)/dB = -1 -> grad_B += grad_O * (-1)
        ngO = ff_tensor_mul_scalar(gO, -1.0); // Create grad_O * (-1)
        if (!ngO) {
            fprintf(stderr, "Error [sub_backward]: Failed to compute negative gradient output.\n");
            return -1;
        }
        if(accumulate_grad(iB->grad, ngO) != 0) { r=-1; }
    }

    ff_tensor_release(ngO); // Release temporary tensor if created
    return r;
}

int ff_tensor_mul_backward(FFTensor* gO, FFTensor* iA, FFTensor* iB){
    if(!gO || !iA || !iB) { return -1; }
    // Assume shapes match (checked by forward op or VALIDATE_BINARY_OP)
    int r=0;
    FFTensor *gA_contrib=NULL, *gB_contrib=NULL; // Temporary tensors for contributions

    // Calculate gradient wrt A: d(A*B)/dA = B -> grad_A += grad_O * B
    if(iA->requires_grad) {
        if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_mul; }
        gA_contrib = ff_tensor_mul(gO, iB); // grad_O * B
        if (!gA_contrib) { r=-1; goto cleanup_mul; }
        if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }
    }

    // Calculate gradient wrt B: d(A*B)/dB = A -> grad_B += grad_O * A
    if(iB->requires_grad) {
        if(ff_tensor_ensure_zero_grad(iB) != 0) { r=-1; goto cleanup_mul; }
        gB_contrib = ff_tensor_mul(gO, iA); // grad_O * A
        if (!gB_contrib) { r=-1; goto cleanup_mul; }
        if(accumulate_grad(iB->grad, gB_contrib) != 0) { r=-1; }
    }

cleanup_mul:
    ff_tensor_release(gA_contrib); // Release temporary tensors
    ff_tensor_release(gB_contrib);
    return r;
}

int ff_tensor_matmul_backward(FFTensor* gO, FFTensor* iA, FFTensor* iB){
    // gO: (M, N), iA: (M, K), iB: (K, N)
    if(!gO || !iA || !iB) { return -1; }
    // Basic checks (should be guaranteed by forward op)
    if(iA->ndim!=2 || iB->ndim!=2 || gO->ndim!=2) { return -1; }
    if(iA->shape[0] != gO->shape[0] || iB->shape[1] != gO->shape[1] || iA->shape[1] != iB->shape[0]) { return -1; }
    if(iA->device != FF_CPU || iB->device != FF_CPU || gO->device != FF_CPU) { return -1; }

    int r=0;
    FFTensor *aT=NULL, *bT=NULL, *gA_contrib=NULL, *gB_contrib=NULL; // Temporaries

    // Grad wrt A: d(A@B)/dA = ??? -> Use dimensions: gA (M,K) += gO (M,N) @ B^T (N,K)
    if(iA->requires_grad) {
        if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_matmul; }
        bT = ff_tensor_transpose(iB); // B^T
        if (!bT) { r=-1; goto cleanup_matmul; }
        gA_contrib = ff_tensor_matmul(gO, bT); // gO @ B^T
        if (!gA_contrib) { r=-1; goto cleanup_matmul; }
        if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }
    }

    // Grad wrt B: d(A@B)/dB = ??? -> Use dimensions: gB (K,N) += A^T (K,M) @ gO (M,N)
    if(iB->requires_grad) {
        if(ff_tensor_ensure_zero_grad(iB) != 0) { r=-1; goto cleanup_matmul; }
        aT = ff_tensor_transpose(iA); // A^T
        if (!aT) { r=-1; goto cleanup_matmul; }
        gB_contrib = ff_tensor_matmul(aT, gO); // A^T @ gO
        if (!gB_contrib) { r=-1; goto cleanup_matmul; }
        if(accumulate_grad(iB->grad, gB_contrib) != 0) { r=-1; }
    }

cleanup_matmul:
    ff_tensor_release(aT);
    ff_tensor_release(bT);
    ff_tensor_release(gA_contrib);
    ff_tensor_release(gB_contrib);
    return r;
}

int ff_tensor_pow_scalar_backward(FFTensor* gO, FFTensor* iA, double exponent){
    // d(A^exp)/dA = exp * A^(exp-1) -> grad_A += grad_O * exp * A^(exp-1)
    if(!gO || !iA) { return -1; }
    if(!iA->requires_grad) { return 0; }

    int r=0;
    FFTensor *a_pow_exp_minus_1 = NULL, *coeff = NULL, *gA_contrib = NULL;

    if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_pow; }

    double exp_minus_1 = exponent - 1.0;
    // Need to handle exponent=0 or exponent=1 cases carefully?
    // pow(0, neg) is inf. pow(neg, non-integer) is complex. Assume valid inputs for now.

    a_pow_exp_minus_1 = ff_tensor_pow_scalar(iA, exp_minus_1); // A^(exp-1)
    if (!a_pow_exp_minus_1) { r=-1; goto cleanup_pow; }

    coeff = ff_tensor_mul_scalar(a_pow_exp_minus_1, exponent); // exp * A^(exp-1)
    if (!coeff) { r=-1; goto cleanup_pow; }

    gA_contrib = ff_tensor_mul(gO, coeff); // grad_O * exp * A^(exp-1)
    if (!gA_contrib) { r=-1; goto cleanup_pow; }

    if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }

cleanup_pow:
    ff_tensor_release(a_pow_exp_minus_1);
    ff_tensor_release(coeff);
    ff_tensor_release(gA_contrib);
    return r;
}

int ff_tensor_mean_backward(FFTensor* gO, FFTensor* iA){
    // d(mean(A))/dA_i = 1/N -> grad_A_i += grad_O * (1/N)
    if(!gO || !iA) { return -1; }
    if(gO->size != 1) { // grad_O must be scalar
        fprintf(stderr, "Error [mean_backward]: Gradient output must be scalar (size is %zu).\n", gO->size);
        return -1;
    }
    if(!iA->requires_grad || iA->size == 0) { return 0; } // No grad needed or nothing to average

    int r=0;
    FFTensor* gA_contrib = NULL; // Tensor filled with grad_O / N

    if(ff_tensor_ensure_zero_grad(iA) != 0) { return -1; }

    // Create a tensor with same shape as input A
    gA_contrib = ff_tensor_create(iA->shape, iA->ndim, iA->dtype, FF_CPU, false);
    if (!gA_contrib) { r=-1; goto cleanup_mean; }

    // Fill the contribution tensor with grad_O / N
    // Use helper ff_tensor_fill_with_scalar_div(target, scalar_tensor, divisor)
    if (ff_tensor_fill_with_scalar_div(gA_contrib, gO, iA->size) != 0) {
        fprintf(stderr, "Error [mean_backward]: Failed to fill gradient contribution tensor.\n");
        r=-1;
    } else {
        // Accumulate the contribution
        if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }
    }

cleanup_mean:
    ff_tensor_release(gA_contrib);
    return r;
}

int ff_tensor_transpose_backward(FFTensor* gO, FFTensor* iA){
    // d(A^T)/dA = ??? -> If Y = A^T, then grad_A += (grad_Y)^T
    if(!gO || !iA) { return -1; }
    if(!iA->requires_grad) { return 0; }
    // Assume forward op checked dimensions (A is 2D, gO has transposed shape)
    if (gO->ndim != 2 || iA->ndim != 2) { return -1;}

    int r=0;
    FFTensor* gO_transposed = NULL; // Temporary for (grad_Y)^T

    if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_transpose; }

    gO_transposed = ff_tensor_transpose(gO); // Transpose the output gradient
    if (!gO_transposed) { r=-1; goto cleanup_transpose; }

    // Accumulate the transposed gradient
    if(accumulate_grad(iA->grad, gO_transposed) != 0) { r=-1; }

cleanup_transpose:
    ff_tensor_release(gO_transposed);
    return r;
}

int ff_tensor_mul_scalar_backward(FFTensor* gO, FFTensor* iA, double scalar_value){
    // d(A * val)/dA = val -> grad_A += grad_O * val
    if(!gO || !iA) { return -1; }
    if(!iA->requires_grad) { return 0; }

    int r=0;
    FFTensor* gA_contrib = NULL;

    if(ff_tensor_ensure_zero_grad(iA) != 0) { return -1; }

    gA_contrib = ff_tensor_mul_scalar(gO, scalar_value); // grad_O * val
    if (!gA_contrib) { return -1; }

    if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }

    ff_tensor_release(gA_contrib);
    return r;
}

int ff_tensor_div_scalar_backward(FFTensor* gO, FFTensor* iA, double scalar_value){
    // d(A / val)/dA = 1/val -> grad_A += grad_O * (1/val)
    if(scalar_value == 0.0) {
        fprintf(stderr, "Error [div_scalar_backward]: Division by zero in forward pass.\n");
        return -1; // Or handle as Inf/NaN gradient?
    }
    // Reuse mul_scalar_backward
    return ff_tensor_mul_scalar_backward(gO, iA, 1.0 / scalar_value);
}

int ff_tensor_rdiv_scalar_backward(FFTensor* gO, FFTensor* iA, double scalar_value, FFTensor* fO){
    // d(val / A)/dA = -val / A^2 -> grad_A += grad_O * (-val / A^2)
    if(!gO || !iA || !fO) { return -1; } // fO is forward output (val/A), NOT NEEDED here.
    if(!iA->requires_grad) { return 0; }

    int r=0;
    FFTensor *a_squared=NULL, *coeff=NULL, *gA_contrib=NULL;

    if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_rdiv; }

    // Calculate A^2
    a_squared = ff_tensor_mul(iA, iA);
    if (!a_squared) { r=-1; goto cleanup_rdiv; }

    // Calculate -val / A^2
    // Need rdiv_scalar, but avoid direct division if A^2 can be zero.
    // Let's use ff_tensor_rdiv_scalar which handles internal division by zero check (returns NAN)
    coeff = ff_tensor_rdiv_scalar(-scalar_value, a_squared);
    if (!coeff) { r=-1; goto cleanup_rdiv; }

    // Calculate grad_O * (-val / A^2)
    gA_contrib = ff_tensor_mul(gO, coeff);
    if (!gA_contrib) { r=-1; goto cleanup_rdiv; }

    // Accumulate grad
    if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }

cleanup_rdiv:
    ff_tensor_release(a_squared);
    ff_tensor_release(coeff);
    ff_tensor_release(gA_contrib);
    return r;
}

int ff_tensor_add_scalar_backward(FFTensor* gO, FFTensor* iA){
     // d(A + val)/dA = 1 -> grad_A += grad_O * 1
    if(!gO || !iA) { return -1; }
    if(!iA->requires_grad) { return 0; }

    if(ff_tensor_ensure_zero_grad(iA) != 0) { return -1; }
    return accumulate_grad(iA->grad, gO);
}

int ff_tensor_tanh_backward(FFTensor* gO, FFTensor* iA, FFTensor* fO){
    // d(tanh(A))/dA = 1 - tanh(A)^2 = 1 - fO^2
    // grad_A += grad_O * (1 - fO^2)
    if(!gO || !iA || !fO) { return -1; } // fO is the output of the forward tanh(A)
    if(!iA->requires_grad) { return 0; }

    int r=0;
    FFTensor *fO_squared=NULL, *one_minus_fO2=NULL, *gA_contrib=NULL;

    if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_tanh; }

    // Calculate fO^2
    fO_squared = ff_tensor_mul(fO, fO);
    if (!fO_squared) { r=-1; goto cleanup_tanh; }

    // Calculate 1 - fO^2 (reuse sub or use mul_scalar(-1) + add_scalar(1))
    // Using mul/add:
    FFTensor* neg_fO2 = ff_tensor_mul_scalar(fO_squared, -1.0);
    if (!neg_fO2) { r=-1; goto cleanup_tanh; }
    one_minus_fO2 = ff_tensor_add_scalar(neg_fO2, 1.0);
    ff_tensor_release(neg_fO2); // Release intermediate
    if (!one_minus_fO2) { r=-1; goto cleanup_tanh; }

    // Calculate grad_O * (1 - fO^2)
    gA_contrib = ff_tensor_mul(gO, one_minus_fO2);
    if (!gA_contrib) { r=-1; goto cleanup_tanh; }

    // Accumulate gradient
    if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }

cleanup_tanh:
    ff_tensor_release(fO_squared);
    ff_tensor_release(one_minus_fO2);
    ff_tensor_release(gA_contrib);
    return r;
}

int ff_tensor_exp_backward(FFTensor* gO, FFTensor* iA, FFTensor* fO){
    // d(exp(A))/dA = exp(A) = fO
    // grad_A += grad_O * fO
    if(!gO || !iA || !fO) { return -1; } // fO is the output of the forward exp(A)
    if(!iA->requires_grad) { return 0; }

    int r=0;
    FFTensor* gA_contrib = NULL;

    if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_exp; }

    // Calculate grad_O * fO
    gA_contrib = ff_tensor_mul(gO, fO);
    if (!gA_contrib) { r=-1; goto cleanup_exp; }

    // Accumulate gradient
    if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }

cleanup_exp:
    ff_tensor_release(gA_contrib);
    return r;
}

int ff_tensor_sigmoid_backward(FFTensor* gO, FFTensor* iA, FFTensor* fO){
    // d(sig(A))/dA = sig(A) * (1 - sig(A)) = fO * (1 - fO)
    // grad_A += grad_O * fO * (1 - fO)
    if(!gO || !iA || !fO) { return -1; } // fO is the output of the forward sigmoid(A)
    if(!iA->requires_grad) { return 0; }

    int r=0;
    FFTensor *one_minus_fO=NULL, *fO_times_1_minus_fO=NULL, *gA_contrib=NULL;

    if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_sigmoid; }

    // Calculate 1 - fO
    FFTensor* neg_fO = ff_tensor_mul_scalar(fO, -1.0);
    if (!neg_fO) { r=-1; goto cleanup_sigmoid; }
    one_minus_fO = ff_tensor_add_scalar(neg_fO, 1.0);
    ff_tensor_release(neg_fO);
    if (!one_minus_fO) { r=-1; goto cleanup_sigmoid; }

    // Calculate fO * (1 - fO)
    fO_times_1_minus_fO = ff_tensor_mul(fO, one_minus_fO);
    if (!fO_times_1_minus_fO) { r=-1; goto cleanup_sigmoid; }

    // Calculate grad_O * fO * (1 - fO)
    gA_contrib = ff_tensor_mul(gO, fO_times_1_minus_fO);
    if (!gA_contrib) { r=-1; goto cleanup_sigmoid; }

    // Accumulate gradient
    if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }

cleanup_sigmoid:
    ff_tensor_release(one_minus_fO);
    ff_tensor_release(fO_times_1_minus_fO);
    ff_tensor_release(gA_contrib);
    return r;
}

int ff_tensor_relu_backward(FFTensor* gO, FFTensor* iA){
    // d(relu(A))/dA = 1 if A > 0, 0 otherwise
    // grad_A += grad_O * (A > 0)
    if(!gO || !iA) { return -1; }
    if(!iA->requires_grad) { return 0; }

    int r=0;
    FFTensor *relu_mask_bool=NULL, *relu_mask_float=NULL, *gA_contrib=NULL;

    if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_relu; }

    // Create boolean mask (A > 0)
    relu_mask_bool = ff_tensor_gt_scalar(iA, 0.0);
    if (!relu_mask_bool) { r=-1; goto cleanup_relu; }

    // Convert mask to float/double type matching iA
    relu_mask_float = ff_tensor_astype(relu_mask_bool, iA->dtype);
    if (!relu_mask_float) { r=-1; goto cleanup_relu; }

    // Calculate grad_O * mask
    gA_contrib = ff_tensor_mul(gO, relu_mask_float);
     if (!gA_contrib) { r=-1; goto cleanup_relu; }

    // Accumulate gradient
    if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }

cleanup_relu:
    ff_tensor_release(relu_mask_bool);
    ff_tensor_release(relu_mask_float);
    ff_tensor_release(gA_contrib);
    return r;
}

int ff_tensor_abs_backward(FFTensor* gO, FFTensor* iA){
    // d(abs(A))/dA = sign(A)
    // grad_A += grad_O * sign(A)
    // Note: sign(0) derivative is undefined, but often taken as 0 in ML.
    // Our ff_tensor_sign gives sign(0)=-1, which might be problematic.
    // Let's proceed, but be aware of the sign(0) issue.
    if(!gO || !iA) { return -1; }
    if(!iA->requires_grad) { return 0; }

    int r=0;
    FFTensor *sign_iA=NULL, *gA_contrib=NULL;

    if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_abs; }

    // Calculate sign(A)
    sign_iA = ff_tensor_sign(iA);
    if (!sign_iA) { r=-1; goto cleanup_abs; }

     // Calculate grad_O * sign(A)
    gA_contrib = ff_tensor_mul(gO, sign_iA);
     if (!gA_contrib) { r=-1; goto cleanup_abs; }

     // Accumulate gradient
    if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }

cleanup_abs:
    ff_tensor_release(sign_iA);
    ff_tensor_release(gA_contrib);
    return r;
}

int ff_tensor_clip_backward(FFTensor* gO, FFTensor* iA, double min_val, double max_val){
    // d(clip(A, min, max))/dA = 1 if min < A < max, 0 otherwise
    // grad_A += grad_O * (min < A < max)
     if(!gO || !iA) { return -1; }
    if(!iA->requires_grad) { return 0; }

    int r=0;
    FFTensor *gt_min_mask=NULL, *lt_max_mask=NULL, *clip_mask_bool=NULL, *clip_mask_float=NULL, *gA_contrib=NULL;

    if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_clip; }

    // Create boolean mask (A > min)
    gt_min_mask = ff_tensor_gt_scalar(iA, min_val);
    if (!gt_min_mask) { r=-1; goto cleanup_clip; }

    // Create boolean mask (A < max)
    lt_max_mask = ff_tensor_lt_scalar(iA, max_val);
     if (!lt_max_mask) { r=-1; goto cleanup_clip; }

    // Combine masks: (A > min) AND (A < max) -> logical element-wise mul for bool
    // Need ff_tensor_mul for bool type, or cast to int/float first. Let's cast.
    FFTensor* gt_min_f = ff_tensor_astype(gt_min_mask, iA->dtype);
    FFTensor* lt_max_f = ff_tensor_astype(lt_max_mask, iA->dtype);
    if (!gt_min_f || !lt_max_f) { r=-1; goto cleanup_clip_masks; } // Need separate label

    clip_mask_float = ff_tensor_mul(gt_min_f, lt_max_f); // Result is 1.0 where both were true, 0.0 otherwise
    if (!clip_mask_float) { r=-1; goto cleanup_clip_masks; }

    // Calculate grad_O * mask
    gA_contrib = ff_tensor_mul(gO, clip_mask_float);
    if (!gA_contrib) { r=-1; goto cleanup_clip_masks; }

    // Accumulate gradient
    if(accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }

// Need two labels because gt_min_f/lt_max_f are created conditionally
cleanup_clip_masks:
    ff_tensor_release(gt_min_f);
    ff_tensor_release(lt_max_f);
cleanup_clip:
    ff_tensor_release(gt_min_mask);
    ff_tensor_release(lt_max_mask);
    // clip_mask_bool was never used/created if we cast early
    // ff_tensor_release(clip_mask_bool);
    ff_tensor_release(clip_mask_float);
    ff_tensor_release(gA_contrib);
    return r;
}

int ff_tensor_outer_backward(FFTensor* gO, FFTensor* iA, FFTensor* iB){
    // gO: (M, N), iA: (M,), iB: (N,)
    // Y = outer(A, B) => Y_ij = A_i * B_j
    // dY_ij / dA_k = delta_ik * B_j
    // dY_ij / dB_k = delta_jk * A_i
    // grad_A_k = sum_{i,j} grad_Y_ij * dY_ij / dA_k = sum_j grad_Y_kj * B_j -> gA = gO @ B
    // grad_B_k = sum_{i,j} grad_Y_ij * dY_ij / dB_k = sum_i grad_Y_ik * A_i -> gB = A^T @ gO (but A is 1D) -> gB = gO^T @ A
    if(!gO || !iA || !iB) { return -1; }
    if(iA->ndim != 1 || iB->ndim != 1 || gO->ndim != 2) { return -1; }
    if(gO->shape[0] != iA->size || gO->shape[1] != iB->size) { return -1; }

    int r=0;
    FFTensor *gA_contrib=NULL, *gB_contrib=NULL, *gO_transposed=NULL;
    size_t M = iA->size;
    size_t N = iB->size;

    // Grad wrt A: gA (M,) = gO (M, N) @ B (N,)
    if(iA->requires_grad) {
        if(ff_tensor_ensure_zero_grad(iA) != 0) { r=-1; goto cleanup_outer; }
        // Need matmul where second arg is 1D. Let's implement manually for now.
        gA_contrib = ff_tensor_create(iA->shape, 1, iA->dtype, FF_CPU, false);
        if (!gA_contrib) { r=-1; goto cleanup_outer; }

         if(!gA_contrib->data || !gO->data || !iB->data) { r=-1; goto cleanup_outer; }
         // TODO: Use strides
         if(iA->dtype == FF_FLOAT32) {
             float* ga = (float*)gA_contrib->data; float* go = (float*)gO->data; float* b = (float*)iB->data;
             for(size_t i=0; i<M; ++i) { // Iterate rows of gO / elements of gA
                 float s = 0.0f;
                 for(size_t j=0; j<N; ++j) { s += go[i*N + j] * b[j]; } // Dot product of gO row and B
                 ga[i] = s;
             }
         } else if (iA->dtype == FF_FLOAT64) {
              double* ga = (double*)gA_contrib->data; double* go = (double*)gO->data; double* b = (double*)iB->data;
             for(size_t i=0; i<M; ++i) {
                 double s = 0.0;
                 for(size_t j=0; j<N; ++j) { s += go[i*N + j] * b[j]; }
                 ga[i] = s;
             }
         } else { r = -1; goto cleanup_outer; } // Unsupported type

        if (r==0 && accumulate_grad(iA->grad, gA_contrib) != 0) { r=-1; }
    }

     // Grad wrt B: gB (N,) = A^T (1, M) @ gO (M, N) -> practically gO^T (N, M) @ A (M,)
    if(iB->requires_grad) {
        if(ff_tensor_ensure_zero_grad(iB) != 0) { r=-1; goto cleanup_outer; }

        gO_transposed = ff_tensor_transpose(gO); // gO^T is (N, M)
        if (!gO_transposed) { r=-1; goto cleanup_outer; }

        // Need matmul where second arg is 1D. Manual implementation.
        gB_contrib = ff_tensor_create(iB->shape, 1, iB->dtype, FF_CPU, false);
         if (!gB_contrib) { r=-1; goto cleanup_outer; }

         if(!gB_contrib->data || !gO_transposed->data || !iA->data) { r=-1; goto cleanup_outer; }
          // TODO: Use strides
         if(iB->dtype == FF_FLOAT32) {
             float* gb = (float*)gB_contrib->data; float* go_t = (float*)gO_transposed->data; float* a = (float*)iA->data;
             for(size_t j=0; j<N; ++j) { // Iterate rows of gO_t / elements of gB
                 float s = 0.0f;
                 for(size_t i=0; i<M; ++i) { s += go_t[j*M + i] * a[i]; } // Dot product of gO_t row and A
                 gb[j] = s;
             }
         } else if (iB->dtype == FF_FLOAT64) {
             double* gb = (double*)gB_contrib->data; double* go_t = (double*)gO_transposed->data; double* a = (double*)iA->data;
             for(size_t j=0; j<N; ++j) {
                 double s = 0.0;
                 for(size_t i=0; i<M; ++i) { s += go_t[j*M + i] * a[i]; }
                 gb[j] = s;
             }
         } else { r = -1; goto cleanup_outer; } // Unsupported type


        if (r==0 && accumulate_grad(iB->grad, gB_contrib) != 0) { r=-1; }
    }

cleanup_outer:
    ff_tensor_release(gA_contrib);
    ff_tensor_release(gB_contrib);
    ff_tensor_release(gO_transposed);
    return r;
}


// --- Optimizer Kernels --- //
int ff_optim_sgd_step(FFTensor* param, const FFTensor* grad, double lr) {
    // Basic checks
    if (!param || !grad) { fprintf(stderr,"sgd_step: NULL input tensor.\n"); return -1; }
    // Check matching attributes
    if (param->dtype != grad->dtype || param->size != grad->size) {
        fprintf(stderr,"sgd_step: Mismatch dtype (%d vs %d) or size (%zu vs %zu).\n", param->dtype, grad->dtype, param->size, grad->size);
        return -1;
    }
    if (param->device != FF_CPU || grad->device != FF_CPU) {
        fprintf(stderr,"sgd_step: Only CPU supported.\n");
        return -1;
    }

    // Handle empty tensors
    if (param->size == 0) { return 0; }
    size_t n = param->size; // Define n after check

    // Check data pointers after handling empty case
    if (!param->data || !grad->data) {
        fprintf(stderr,"sgd_step: NULL data pointer for non-empty tensor (param:%p, grad:%p).\n", param->data, grad->data);
        return -1;
    }


    // TODO: Use strides for non-contiguous memory
    switch (param->dtype) {
        case FF_FLOAT32: {
            float l=(float)lr;
            float* p=(float*)param->data;
            float* g=(float*)grad->data;
            for(size_t i=0;i<n;++i) { p[i] -= l*g[i]; }
            break;
        }
        case FF_FLOAT64: {
            double* p=(double*)param->data;
            double* g=(double*)grad->data;
            for(size_t i=0;i<n;++i) { p[i] -= lr*g[i]; }
            break;
        }
        default: fprintf(stderr, "Error [sgd_step]: Unsupported dtype %d.\n", param->dtype); return -1;
    }
    return 0;
}
