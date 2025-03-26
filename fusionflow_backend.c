// fusionflow_backend.c
#include "fusionflow_backend.h"
#include <stdlib.h> // For malloc, free
#include <string.h> // For memcpy, memset
#include <stdio.h>  // For error printing (temporary)
#include <assert.h> // For basic checks

// --- Utility Functions Implementation ---

size_t ff_dtype_size(FFDataType dtype) {
    switch (dtype) {
        case FF_FLOAT32: return sizeof(float);
        case FF_FLOAT64: return sizeof(double);
        case FF_INT32:   return sizeof(int32_t);
        case FF_INT64:   return sizeof(int64_t);
        case FF_BOOL:    return sizeof(bool); // Or uint8_t depending on representation
        default:         return 0; // Indicate error or unknown type
    }
}

size_t ff_calculate_size(const size_t* shape, size_t ndim) {
    if (ndim == 0) return 1; // Scalar case
    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i) {
        size *= shape[i];
    }
    return size;
}

void ff_calculate_contiguous_strides(const size_t* shape, size_t ndim, size_t element_size, size_t* out_strides) {
    // Strides are calculated in reverse order of dimensions for C-style contiguous
    if (ndim == 0) return; // Scalar has no strides in the array sense
    size_t current_stride = element_size;
    for (size_t i = ndim; i > 0; --i) {
        out_strides[i - 1] = (shape[i - 1] > 1) ? current_stride : 0; // Stride is 0 for dim size 1 ? Or just current_stride? Let's keep it simple: current_stride unless size 0? No, standard stride calc
        // Standard stride: Number of bytes to skip to get to the next element in that dimension.
        // For the very last dimension, it's the element size.
        // For the second to last, it's element_size * shape[last_dim]
        // Let's recalculate standard contiguous strides properly.
    }
    // Correct calculation for C-contiguous strides
    size_t stride_val = element_size;
    for (int i = (int)ndim - 1; i >= 0; --i) {
        out_strides[i] = stride_val;
        stride_val *= shape[i];
    }
     // Example: shape (2, 3), elem_size 4
     // stride_val = 4
     // i = 1: strides[1] = 4; stride_val = 4 * 3 = 12
     // i = 0: strides[0] = 12; stride_val = 12 * 2 = 24
     // Strides: (12, 4) -> Correct
}


// --- Tensor Lifecycle Implementation ---

FFTensor* ff_tensor_create(const size_t* shape, size_t ndim, FFDataType dtype, FFDevice device, bool requires_grad) {
    // Basic validation
    if (!shape && ndim > 0) {
        fprintf(stderr, "Error: Shape cannot be NULL for ndim > 0.\n");
        return NULL;
    }
    if (device != FF_CPU) {
        fprintf(stderr, "Error: Only FF_CPU device is supported currently.\n");
        return NULL;
    }
    size_t element_size = ff_dtype_size(dtype);
    if (element_size == 0) {
        fprintf(stderr, "Error: Unsupported data type.\n");
        return NULL;
    }

    FFTensor* tensor = (FFTensor*)malloc(sizeof(FFTensor));
    if (!tensor) {
        perror("Failed to allocate memory for FFTensor struct");
        return NULL;
    }

    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->device = device;
    tensor->requires_grad = requires_grad;
    tensor->grad = NULL; // Gradient not allocated initially
    tensor->ref_count = 1; // Initial reference count

    // Allocate and copy shape
    if (ndim > 0) {
        tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
        if (!tensor->shape) {
            perror("Failed to allocate memory for shape");
            free(tensor);
            return NULL;
        }
        memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    } else {
        tensor->shape = NULL; // Scalar
    }

    // Calculate size and allocate data buffer
    tensor->size = ff_calculate_size(shape, ndim);
    tensor->nbytes = tensor->size * element_size;

    // Use calloc to initialize data to zero
    tensor->data = calloc(tensor->size, element_size);
    if (!tensor->data) {
        perror("Failed to allocate memory for tensor data");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    // Allocate and calculate strides
    if (ndim > 0) {
        tensor->strides = (size_t*)malloc(ndim * sizeof(size_t));
        if (!tensor->strides) {
            perror("Failed to allocate memory for strides");
            free(tensor->data);
            free(tensor->shape);
            free(tensor);
            return NULL;
        }
        ff_calculate_contiguous_strides(shape, ndim, element_size, tensor->strides);
    } else {
        tensor->strides = NULL; // Scalar
    }

    return tensor;
}

void ff_tensor_retain(FFTensor* tensor) {
    if (tensor) {
        // Consider atomic increment if using multi-threading later
        tensor->ref_count++;
    }
}

void ff_tensor_release(FFTensor* tensor) {
    if (!tensor) {
        return;
    }

    // Consider atomic decrement if using multi-threading later
    tensor->ref_count--;

    if (tensor->ref_count <= 0) {
        // Ref count is zero, free resources

        // Recursively release gradient tensor if it exists
        if (tensor->grad) {
            ff_tensor_release(tensor->grad);
            tensor->grad = NULL; // Avoid double free if tensor references itself via grad somehow
        }

        // Free data buffer, shape, and strides arrays
        // TODO: Add check if this tensor is a view - if so, don't free data, just decr source ref count
        free(tensor->data);
        free(tensor->shape);
        free(tensor->strides);

        // Free the tensor struct itself
        free(tensor);
    }
}

// --- Basic Operations Implementation ---

// Helper macro for element-wise loops (simplistic, assumes contiguous or uses strides)
#define ELEMENT_WISE_LOOP(tensor, ptr_type, code_block) \
    do { \
        ptr_type* data_ptr = (ptr_type*)tensor->data; \
        for (size_t i = 0; i < tensor->size; ++i) { \
            code_block; \
            data_ptr++; \
        } \
    } while(0)

// Slightly better helper using strides (only works for compatible strides/broadcasting needed)
// This still requires a more complex iteration logic for general case
// For simple ADD with same shape/contiguous, direct indexing is fine

FFTensor* ff_tensor_add(const FFTensor* a, const FFTensor* b) {
    // --- Validation ---
    if (!a || !b) {
        fprintf(stderr, "Error: Input tensors for add cannot be NULL.\n");
        return NULL;
    }
    if (a->ndim != b->ndim) { // Basic shape check
        fprintf(stderr, "Error: Tensor dimensions do not match for add (%zu vs %zu).\n", a->ndim, b->ndim);
        return NULL; // TODO: Implement broadcasting
    }
    for (size_t i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Error: Tensor shapes do not match for add at dim %zu (%zu vs %zu).\n", i, a->shape[i], b->shape[i]);
            return NULL; // TODO: Implement broadcasting
        }
    }
    if (a->dtype != b->dtype) {
        fprintf(stderr, "Error: Tensor data types do not match for add (%d vs %d).\n", a->dtype, b->dtype);
        return NULL; // TODO: Implement type promotion
    }
    if (a->device != FF_CPU || b->device != FF_CPU) {
         fprintf(stderr, "Error: Addition only supported on CPU currently.\n");
        return NULL;
    }

    // Determine output properties
    bool requires_grad = a->requires_grad || b->requires_grad;
    FFDataType dtype = a->dtype; // Assume same dtype for now

    // Create result tensor
    FFTensor* result = ff_tensor_create(a->shape, a->ndim, dtype, FF_CPU, requires_grad);
    if (!result) {
        fprintf(stderr, "Error: Failed to create result tensor for add.\n");
        return NULL;
    }

    // --- Perform Computation ---
    // Simplistic loop assuming contiguous data for now
    // A robust implementation needs to handle strides properly
    size_t n_elements = result->size;
    switch (dtype) {
        case FF_FLOAT32: {
            float* a_ptr = (float*)a->data;
            float* b_ptr = (float*)b->data;
            float* res_ptr = (float*)result->data;
            for (size_t i = 0; i < n_elements; ++i) {
                res_ptr[i] = a_ptr[i] + b_ptr[i];
            }
            break;
        }
        case FF_FLOAT64: {
            double* a_ptr = (double*)a->data;
            double* b_ptr = (double*)b->data;
            double* res_ptr = (double*)result->data;
            for (size_t i = 0; i < n_elements; ++i) {
                res_ptr[i] = a_ptr[i] + b_ptr[i];
            }
            break;
        }
        // Add cases for INT32, INT64 etc.
        default:
            fprintf(stderr, "Error: Unsupported data type in add computation.\n");
            ff_tensor_release(result); // Clean up created tensor
            return NULL;
    }

    // --- Autograd Link (Placeholder) ---
    // In a full system, this operation would also store context
    // needed for the backward pass and link the result tensor
    // to its parents (a, b). This C backend *only* does the computation.
    // The graph building happens at a higher level (e.g., Python wrapper).

    return result;
}
