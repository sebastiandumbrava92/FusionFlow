// fusionflow_backend.h
#ifndef FUSIONFLOW_BACKEND_H
#define FUSIONFLOW_BACKEND_H

#include <stddef.h> // For size_t
#include <stdbool.h> // For bool type (C99 onwards)
#include <stdint.h> // For fixed-width integer types

// --- Enums ---

// Supported Data Types
typedef enum {
    FF_FLOAT32,
    FF_FLOAT64,
    FF_INT32,
    FF_INT64,
    FF_BOOL // Add more as needed (uint8, int8, etc.)
} FFDataType;

// Supported Devices (Expand later for GPU etc.)
typedef enum {
    FF_CPU,
    FF_GPU_CUDA, // Placeholder
    FF_GPU_ROCM  // Placeholder
} FFDevice;

// --- Tensor Structure ---

// Forward declaration for gradient tensor pointer
struct FFTensor;

typedef struct FFTensor {
    void* data;         // Pointer to the raw data buffer
    size_t* shape;      // Array of dimension sizes
    size_t* strides;    // Array of strides in bytes for each dimension
    size_t ndim;        // Number of dimensions
    FFDataType dtype;   // Data type enum
    FFDevice device;    // Device where the tensor resides

    size_t size;        // Total number of elements
    size_t nbytes;      // Total size of the data buffer in bytes

    // Memory Management (Simple Reference Counting)
    int ref_count;

    // Autograd related (Basic placeholders)
    bool requires_grad;
    struct FFTensor* grad; // Pointer to the gradient tensor (same shape/dtype)

    // Internal flags/metadata (optional)
    // e.g., bool is_contiguous;
    //       void* allocation; // Pointer to original allocation if this is a view

} FFTensor;


// --- Utility Functions ---

// Get size of a data type in bytes
size_t ff_dtype_size(FFDataType dtype);

// Calculate total number of elements from shape
size_t ff_calculate_size(const size_t* shape, size_t ndim);

// Calculate contiguous strides for a given shape and element size
void ff_calculate_contiguous_strides(const size_t* shape, size_t ndim, size_t element_size, size_t* out_strides);

// --- Tensor Lifecycle Functions ---

/**
 * @brief Creates a new FFTensor with allocated data buffer.
 * Initializes data to zeros by default.
 * Initializes strides for contiguous memory layout.
 * Initializes ref_count to 1.
 * Gradient tensor is NOT allocated by default.
 *
 * @param shape Array of dimension sizes.
 * @param ndim Number of dimensions.
 * @param dtype Data type.
 * @param device Device (currently only FF_CPU supported).
 * @param requires_grad Whether the tensor should track gradients.
 * @return Pointer to the newly created FFTensor, or NULL on failure.
 */
FFTensor* ff_tensor_create(const size_t* shape, size_t ndim, FFDataType dtype, FFDevice device, bool requires_grad);

/**
 * @brief Creates a tensor that is a view of another tensor's data.
 * Shares data buffer but has own shape, strides, ndim.
 * Increments ref_count of the source tensor's data allocation mechanism (if applicable, complex).
 * For simplicity now, maybe just point to source data and manage ref_count externally?
 * --> Let's defer view creation for now, it adds complexity to memory management.
 */
// FFTensor* ff_tensor_create_view(...);

/**
 * @brief Increments the reference count of the tensor.
 * Needed when assigning tensor pointer to another variable or structure.
 *
 * @param tensor The tensor to retain.
 */
void ff_tensor_retain(FFTensor* tensor);

/**
 * @brief Decrements the reference count of the tensor.
 * If the reference count reaches zero, frees the associated memory
 * (data buffer, shape, strides, gradient tensor, and the tensor struct itself).
 * Handles potential recursive destruction if gradient tensor also reaches ref_count zero.
 *
 * @param tensor The tensor to release.
 */
void ff_tensor_release(FFTensor* tensor);


// --- Basic Data Access (Example - needs bounds checking etc.) ---
// void* ff_tensor_get_element_ptr(const FFTensor* tensor, const size_t* indices);


// --- Basic Operations (Prototypes) ---

/**
 * @brief Performs element-wise addition of two tensors (a + b).
 * Assumes a and b have the same shape and dtype for this basic version.
 * Creates and returns a new tensor for the result.
 * Does NOT handle autograd graph building - this function is just the compute kernel.
 *
 * @param a The first input tensor.
 * @param b The second input tensor.
 * @return A new FFTensor containing the result, or NULL on failure (e.g., shape mismatch).
 */
FFTensor* ff_tensor_add(const FFTensor* a, const FFTensor* b);

// Add prototypes for other operations like subtract, multiply, matmul etc.
// FFTensor* ff_tensor_mul(const FFTensor* a, const FFTensor* b);
// FFTensor* ff_tensor_matmul(const FFTensor* a, const FFTensor* b); // More complex


#endif // FUSIONFLOW_BACKEND_H
