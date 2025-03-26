# fusionflow_core.py
import ctypes
import os
import numpy as np
import atexit
import gc
import time
from collections import deque # For topological sort
import traceback # Ensure traceback is imported

print(f"--- Loading FusionFlow C Backend ---")
print(f"Current Time: {time.strftime('%Y-%m-%d %H:%M:%S')}") # Add timestamp

# --- 1. Define Constants ---
class FFDataType:
    FLOAT32 = 0; FLOAT64 = 1; INT32 = 2; INT64 = 3; BOOL = 4
    _map = {0:"float32", 1:"float64", 2:"int32", 3:"int64", 4:"bool"}
    _np_map = {0:np.float32, 1:np.float64, 2:np.int32, 3:np.int64, 4:np.bool_}
    _ctype_map = {0:ctypes.c_float, 1:ctypes.c_double, 2:ctypes.c_int32, 3:ctypes.c_int64, 4:ctypes.c_bool}
    @classmethod
    def to_string(cls, dtype_enum): return cls._map.get(dtype_enum, "unknown")
    @classmethod
    def to_numpy(cls, dtype_enum): return cls._np_map.get(dtype_enum)
    @classmethod
    def to_ctype(cls, dtype_enum): return cls._ctype_map.get(dtype_enum)

class FFDevice:
    CPU = 0; GPU_CUDA = 1; GPU_ROCM = 2
    _map = {0: "cpu", 1: "cuda", 2: "rocm"}
    @classmethod
    def to_string(cls, device_enum): return cls._map.get(device_enum, "unknown")

# --- 2. Load the Shared Library ---
_LIB_LOADED = False
ff_lib = None
try:
    script_dir = os.path.dirname(os.path.abspath(__file__)) or '.'
    lib_path = os.path.join(script_dir, "libfusionflow_backend.so")
    if not os.path.exists(lib_path):
        print(f"Warning: Library not found at {lib_path}. Trying system load path.")
        lib_path = "libfusionflow_backend.so"
    ff_lib = ctypes.CDLL(lib_path)
    print(f"Successfully loaded C backend: {lib_path}")
    _LIB_LOADED = True
except OSError as e:
    print(f"FATAL: Error loading shared library: {e}")

# --- 3. Define C Structures and Pointers ---
if _LIB_LOADED:
    class FFTensorStruct(ctypes.Structure): pass
    FFTensorStruct._fields_ = [
        ("data", ctypes.c_void_p), ("shape", ctypes.POINTER(ctypes.c_size_t)),
        ("strides", ctypes.POINTER(ctypes.c_size_t)), ("ndim", ctypes.c_size_t),
        ("dtype", ctypes.c_int), ("device", ctypes.c_int),
        ("size", ctypes.c_size_t), ("nbytes", ctypes.c_size_t),
        ("ref_count", ctypes.c_int), ("requires_grad", ctypes.c_bool),
        ("grad", ctypes.POINTER(FFTensorStruct)), ]
    FFTensor_p = ctypes.POINTER(FFTensorStruct)

# --- 4. Define C Function Signatures ---
if _LIB_LOADED:
    # Utility
    ff_lib.ff_dtype_size.argtypes = [ctypes.c_int]; ff_lib.ff_dtype_size.restype = ctypes.c_size_t
    # Lifecycle & State
    ff_lib.ff_tensor_create.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_bool]; ff_lib.ff_tensor_create.restype = FFTensor_p
    ff_lib.ff_tensor_retain.argtypes = [FFTensor_p]; ff_lib.ff_tensor_retain.restype = None
    ff_lib.ff_tensor_release.argtypes = [FFTensor_p]; ff_lib.ff_tensor_release.restype = None
    ff_lib.ff_tensor_create_from_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_bool]; ff_lib.ff_tensor_create_from_data.restype = FFTensor_p
    ff_lib.ff_tensor_copy_from_host.argtypes = [FFTensor_p, ctypes.c_void_p]; ff_lib.ff_tensor_copy_from_host.restype = ctypes.c_int
    ff_lib.ff_tensor_fill.argtypes = [FFTensor_p, ctypes.c_double]; ff_lib.ff_tensor_fill.restype = ctypes.c_int
    ff_lib.ff_tensor_ensure_zero_grad.argtypes = [FFTensor_p]; ff_lib.ff_tensor_ensure_zero_grad.restype = ctypes.c_int
    ff_lib.ff_tensor_zero_data.argtypes = [FFTensor_p]; ff_lib.ff_tensor_zero_data.restype = ctypes.c_int
    ff_lib.ff_tensor_copy.argtypes = [FFTensor_p]; ff_lib.ff_tensor_copy.restype = FFTensor_p
    ff_lib.ff_tensor_ones.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_bool]; ff_lib.ff_tensor_ones.restype = FFTensor_p
    ff_lib.ff_tensor_eye.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_bool]; ff_lib.ff_tensor_eye.restype = FFTensor_p
    ff_lib.ff_tensor_uniform.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_bool]; ff_lib.ff_tensor_uniform.restype = FFTensor_p
    ff_lib.ff_tensor_astype.argtypes = [FFTensor_p, ctypes.c_int]; ff_lib.ff_tensor_astype.restype = FFTensor_p
    ff_lib.ff_tensor_transpose.argtypes = [FFTensor_p]; ff_lib.ff_tensor_transpose.restype = FFTensor_p
    # Forward Ops
    ff_lib.ff_tensor_add.argtypes = [FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_add.restype = FFTensor_p
    ff_lib.ff_tensor_sub.argtypes = [FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_sub.restype = FFTensor_p
    ff_lib.ff_tensor_mul.argtypes = [FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_mul.restype = FFTensor_p
    ff_lib.ff_tensor_mul_scalar.argtypes = [FFTensor_p, ctypes.c_double]; ff_lib.ff_tensor_mul_scalar.restype = FFTensor_p
    ff_lib.ff_tensor_matmul.argtypes = [FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_matmul.restype = FFTensor_p
    ff_lib.ff_tensor_pow_scalar.argtypes = [FFTensor_p, ctypes.c_double]; ff_lib.ff_tensor_pow_scalar.restype = FFTensor_p
    ff_lib.ff_tensor_mean.argtypes = [FFTensor_p]; ff_lib.ff_tensor_mean.restype = FFTensor_p
    ff_lib.ff_tensor_div_scalar.argtypes = [FFTensor_p, ctypes.c_double]; ff_lib.ff_tensor_div_scalar.restype = FFTensor_p
    ff_lib.ff_tensor_rdiv_scalar.argtypes = [ctypes.c_double, FFTensor_p]; ff_lib.ff_tensor_rdiv_scalar.restype = FFTensor_p
    ff_lib.ff_tensor_add_scalar.argtypes = [FFTensor_p, ctypes.c_double]; ff_lib.ff_tensor_add_scalar.restype = FFTensor_p
    ff_lib.ff_tensor_tanh.argtypes = [FFTensor_p]; ff_lib.ff_tensor_tanh.restype = FFTensor_p
    ff_lib.ff_tensor_exp.argtypes = [FFTensor_p]; ff_lib.ff_tensor_exp.restype = FFTensor_p
    ff_lib.ff_tensor_sigmoid.argtypes = [FFTensor_p]; ff_lib.ff_tensor_sigmoid.restype = FFTensor_p
    ff_lib.ff_tensor_relu.argtypes = [FFTensor_p]; ff_lib.ff_tensor_relu.restype = FFTensor_p
    ff_lib.ff_tensor_abs.argtypes = [FFTensor_p]; ff_lib.ff_tensor_abs.restype = FFTensor_p
    ff_lib.ff_tensor_clip.argtypes = [FFTensor_p, ctypes.c_double, ctypes.c_double]; ff_lib.ff_tensor_clip.restype = FFTensor_p
    ff_lib.ff_tensor_gt_scalar.argtypes = [FFTensor_p, ctypes.c_double]; ff_lib.ff_tensor_gt_scalar.restype = FFTensor_p
    ff_lib.ff_tensor_lt_scalar.argtypes = [FFTensor_p, ctypes.c_double]; ff_lib.ff_tensor_lt_scalar.restype = FFTensor_p # Added prototype
    ff_lib.ff_tensor_outer.argtypes = [FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_outer.restype = FFTensor_p
    ff_lib.ff_tensor_sign.argtypes = [FFTensor_p]; ff_lib.ff_tensor_sign.restype = FFTensor_p
    # Backward Ops
    ff_lib.ff_tensor_add_backward.argtypes = [FFTensor_p, FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_add_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_sub_backward.argtypes = [FFTensor_p, FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_sub_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_mul_backward.argtypes = [FFTensor_p, FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_mul_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_matmul_backward.argtypes = [FFTensor_p, FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_matmul_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_pow_scalar_backward.argtypes = [FFTensor_p, FFTensor_p, ctypes.c_double]; ff_lib.ff_tensor_pow_scalar_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_mean_backward.argtypes = [FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_mean_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_transpose_backward.argtypes = [FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_transpose_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_mul_scalar_backward.argtypes = [FFTensor_p, FFTensor_p, ctypes.c_double]; ff_lib.ff_tensor_mul_scalar_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_div_scalar_backward.argtypes = [FFTensor_p, FFTensor_p, ctypes.c_double]; ff_lib.ff_tensor_div_scalar_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_rdiv_scalar_backward.argtypes = [FFTensor_p, FFTensor_p, ctypes.c_double, FFTensor_p]; ff_lib.ff_tensor_rdiv_scalar_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_add_scalar_backward.argtypes = [FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_add_scalar_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_tanh_backward.argtypes = [FFTensor_p, FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_tanh_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_exp_backward.argtypes = [FFTensor_p, FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_exp_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_sigmoid_backward.argtypes = [FFTensor_p, FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_sigmoid_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_relu_backward.argtypes = [FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_relu_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_abs_backward.argtypes = [FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_abs_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_clip_backward.argtypes = [FFTensor_p, FFTensor_p, ctypes.c_double, ctypes.c_double]; ff_lib.ff_tensor_clip_backward.restype = ctypes.c_int
    ff_lib.ff_tensor_outer_backward.argtypes = [FFTensor_p, FFTensor_p, FFTensor_p]; ff_lib.ff_tensor_outer_backward.restype = ctypes.c_int
    # Optimizer Ops
    ff_lib.ff_optim_sgd_step.argtypes = [FFTensor_p, FFTensor_p, ctypes.c_double]; ff_lib.ff_optim_sgd_step.restype = ctypes.c_int

    print("All C function signatures defined.")
else:
    print("Skipping C function signature definition as library failed to load.")

print("-" * 30)

# --- 5. Python Tensor Wrapper Class ---
class Tensor:
    """
    Python wrapper for the FusionFlow C Tensor (FFTensor).
    Manages lifecycle and integrates autograd computation graph.
    """
    def __init__(self, c_tensor_ptr, _children=(), _op=''):
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not isinstance(c_tensor_ptr, FFTensor_p): raise TypeError(f"Expected FFTensor_p, got {type(c_tensor_ptr)}")
        if not c_tensor_ptr: raise ValueError("Cannot initialize Tensor with a NULL pointer.")
        self._ptr = c_tensor_ptr
        # Note: C functions returning new tensors start with ref_count=1.
        # We don't increment here. Retain is called when accessing .grad
        # or passing a Tensor object to C that needs to persist beyond the call.
        self._prev = set(_children)
        self._backward_fn = lambda: None
        self._ctx = {} # Always use dict for context

    def __del__(self):
        # This is called when the Python Tensor object is garbage collected.
        # It releases the C tensor pointer, decrementing its reference count.
        if _LIB_LOADED and ff_lib and hasattr(self, '_ptr') and self._ptr:
            ff_lib.ff_tensor_release(self._ptr)
            self._ptr = None # Prevent double release

    # --- Factory Methods ---
    @classmethod
    def zeros(cls, shape, dtype=FFDataType.FLOAT32, device=FFDevice.CPU, requires_grad=False):
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        ndim = len(shape); shape_array = (ctypes.c_size_t * ndim)(*[ctypes.c_size_t(s) for s in shape])
        c_ptr = ff_lib.ff_tensor_create(shape_array, ndim, dtype, device, requires_grad)
        if not c_ptr: raise MemoryError("C backend failed create tensor.")
        return cls(c_ptr, _op='zeros')

    @classmethod
    def ones(cls, shape, dtype=FFDataType.FLOAT32, device=FFDevice.CPU, requires_grad=False):
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        ndim = len(shape); shape_array = (ctypes.c_size_t * ndim)(*[ctypes.c_size_t(s) for s in shape])
        c_ptr = ff_lib.ff_tensor_ones(shape_array, ndim, dtype, device, requires_grad)
        if not c_ptr: raise MemoryError("C backend failed create ones tensor.")
        return cls(c_ptr, _op='ones')

    @classmethod
    def eye(cls, dim, dtype=FFDataType.FLOAT32, device=FFDevice.CPU, requires_grad=False):
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        c_ptr = ff_lib.ff_tensor_eye(ctypes.c_size_t(dim), dtype, device, requires_grad)
        if not c_ptr: raise MemoryError("C backend failed create eye tensor.")
        return cls(c_ptr, _op='eye')

    @classmethod
    def uniform(cls, low, high, shape, dtype=FFDataType.FLOAT32, device=FFDevice.CPU, requires_grad=False):
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        ndim = len(shape); shape_array = (ctypes.c_size_t * ndim)(*[ctypes.c_size_t(s) for s in shape])
        c_ptr = ff_lib.ff_tensor_uniform(float(low), float(high), shape_array, ndim, dtype, device, requires_grad)
        if not c_ptr: raise MemoryError("C backend failed create uniform tensor.")
        return cls(c_ptr, _op='uniform')

    @classmethod
    def from_numpy(cls, np_array: np.ndarray, device=FFDevice.CPU, requires_grad=False):
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not isinstance(np_array, np.ndarray): raise TypeError("Input must be a NumPy ndarray.")

        # Ensure data is C-contiguous (row-major) as the C backend likely assumes this
        if not np_array.flags['C_CONTIGUOUS']:
            # print("Warning: NumPy array not C-contiguous. Creating a copy.") # Optional warning
            np_array = np.ascontiguousarray(np_array)

        # Find corresponding FFDataType enum
        dtype = None
        for dt_enum, np_dt in FFDataType._np_map.items():
            if np_array.dtype == np_dt or np.issubdtype(np_array.dtype, np_dt):
                dtype = dt_enum
                break
        if dtype is None: raise TypeError(f"Unsupported NumPy dtype: {np_array.dtype}")

        shape = np_array.shape; ndim = np_array.ndim
        try:
             # Create ctypes array for shape
            shape_array = (ctypes.c_size_t * ndim)(*[ctypes.c_size_t(s) for s in shape])
        except TypeError:
            raise TypeError(f"Invalid shape derived from NumPy array: {shape}")

        # Get pointer to NumPy array data
        host_data_ptr = np_array.ctypes.data_as(ctypes.c_void_p)

        # Call C function to create tensor FROM the NumPy data (copies data)
        c_ptr = ff_lib.ff_tensor_create_from_data(host_data_ptr, shape_array, ndim, dtype, device, requires_grad)
        if not c_ptr: raise MemoryError("C backend failed create tensor from data.")

        return cls(c_ptr, _op='from_numpy')

    # --- In-place Operations ---
    def fill_(self, value):
        """Fills the tensor with a scalar value, in-place."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not self._ptr: raise RuntimeError("Cannot operate on released Tensor.")
        # TODO: Add version counter / check for autograd safety if modifying requires_grad tensors
        # if self.requires_grad: print("Warning: In-place operation fill_ on requires_grad tensor.")
        ret_code = ff_lib.ff_tensor_fill(self._ptr, float(value))
        if ret_code != 0: raise RuntimeError(f"C backend tensor fill failed (code: {ret_code}).")
        return self # Return self for chaining

    # --- Data Copying ---
    def copy_from_numpy(self, np_array: np.ndarray):
        """Copies data from a NumPy array into this tensor's existing memory."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.");
        if not self._ptr: raise RuntimeError("Cannot operate on released Tensor.");
        if not isinstance(np_array, np.ndarray): raise TypeError("Input must be ndarray.");

        # Check shape compatibility
        if np_array.shape != self.shape: raise ValueError(f"Shape mismatch for copy_from_numpy: Input {np_array.shape} vs Tensor {self.shape}");

        # Check dtype compatibility
        np_equiv_dtype = FFDataType.to_numpy(self.dtype)
        is_compat = (np_equiv_dtype is not None and (np_array.dtype==np_equiv_dtype or np.issubdtype(np_array.dtype, np_equiv_dtype)))
        if not is_compat: raise TypeError(f"Dtype mismatch for copy_from_numpy: Input {np_array.dtype} vs Tensor {FFDataType.to_string(self.dtype)}");

        # Ensure input NumPy array is C-contiguous
        if not np_array.flags['C_CONTIGUOUS']:
            # print("Warning: copy_from_numpy input not C-contiguous. Creating temp copy.") # Optional
            np_array = np.ascontiguousarray(np_array)

        # Check byte size compatibility (redundant if shape/dtype match, but good sanity check)
        c_tensor_nbytes = self._c_tensor.nbytes
        if np_array.nbytes != c_tensor_nbytes: raise ValueError(f"Byte size mismatch for copy_from_numpy: Input {np_array.nbytes} vs Tensor {c_tensor_nbytes}. Should not happen if shape/dtype match.");

        # Get pointer to NumPy data
        host_data_ptr = np_array.ctypes.data_as(ctypes.c_void_p)

        # Call C function to copy data
        ret_code = ff_lib.ff_tensor_copy_from_host(self._ptr, host_data_ptr)
        if ret_code != 0: raise RuntimeError(f"C backend tensor copy_from_host failed (code: {ret_code}).")
        # No return value needed as it modifies self

    def copy(self):
        """Creates a new Tensor with a copy of the data."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not self._ptr: raise RuntimeError("Cannot operate on released Tensor.")

        result_ptr = ff_lib.ff_tensor_copy(self._ptr)
        if not result_ptr: raise RuntimeError("C backend tensor copy failed.")

        # The new C tensor inherits requires_grad. Wrap it.
        # The new tensor is independent, so no children/op for autograd.
        return Tensor(result_ptr, _op='copy')

    def astype(self, dtype):
        """Creates a new Tensor with the data cast to a different dtype."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not self._ptr: raise RuntimeError("Cannot operate on released Tensor.")
        if not isinstance(dtype, int) or dtype not in FFDataType._map:
             raise TypeError(f"Invalid target dtype: {dtype}. Use FFDataType enum values.")

        result_ptr = ff_lib.ff_tensor_astype(self._ptr, dtype)
        if not result_ptr: raise RuntimeError(f"C backend tensor astype failed (to dtype {dtype}).")

        # Check if the operation supports gradient propagation
        # The C function might reset requires_grad if the cast isn't differentiable
        # Here, we assume it IS potentially differentiable if the input required grad.
        # A more robust C API might return a flag indicating differentiability.
        requires_grad = self.requires_grad and result_ptr.contents.requires_grad

        # Create the Python wrapper
        result_tensor = Tensor(result_ptr,
                               _children={self} if requires_grad else (),
                               _op='astype' if requires_grad else '')
        # Ensure requires_grad flag consistency (might be redundant if C sets it)
        if result_tensor._ptr: result_tensor._ptr.contents.requires_grad = requires_grad

        # Define backward function IF gradients are required
        if requires_grad:
            # Capture self's pointer AT THIS TIME for the backward closure
            self_ptr_cap = self._ptr # Crucial capture

            def _backward_fn():
                grad_output_ptr = result_tensor._c_tensor.grad
                if grad_output_ptr:
                    # We need a C function ff_tensor_astype_backward(grad_output, input_a)
                    # It should handle accumulating the gradient considering the type change.
                    # Simplest case (e.g. float->double) might just cast grad_output back
                    # and accumulate. More complex cases (e.g., int->float) need care.
                    # Since it's not defined in the provided C signatures, we'll just warn.
                    # Check if ensure_zero_grad succeeds before attempting accumulation
                    if ff_lib.ff_tensor_ensure_zero_grad(self_ptr_cap) != 0:
                        print(f"Warning: Could not ensure zero grad for input of astype op.")
                        return

                    # --- Placeholder for actual C backward call ---
                    # Example: Assume identity gradient for compatible types for now
                    # Need a C function like ff_tensor_accumulate_grad_maybe_cast(target_grad_ptr, contrib_grad_ptr)
                    print(f"Warning: Autograd backward pass for 'astype' not fully implemented in C backend. Accumulating gradient directly (may be incorrect for some type casts).")
                    # Attempt direct accumulation (may fail if types differ significantly)
                    # This is likely incorrect if types are not compatible floats/doubles
                    if self_ptr_cap.contents.grad and grad_output_ptr:
                         # Need a C accumulate function! accumulate_grad(self_ptr_cap.contents.grad, grad_output_ptr)
                         # Since we don't have a C accumulate function exposed via ctypes,
                         # we cannot directly perform the accumulation here safely.
                         print("ERROR: Cannot perform C grad accumulation from Python for astype.")
                         pass # Cannot call C accumulate_grad helper directly
                    # --- End Placeholder ---

                    # Placeholder if ff_tensor_astype_backward existed:
                    # ret = ff_lib.ff_tensor_astype_backward(grad_output_ptr, self_ptr_cap)
                    # if ret != 0: print(f"Warning: C backward for astype failed ({ret}).")

            result_tensor._backward_fn = _backward_fn

        return result_tensor

    # --- Properties ---
    @property
    def _c_tensor(self):
        """Direct access to the underlying C FFTensor struct contents."""
        if not hasattr(self, '_ptr') or not self._ptr:
            # Or raise specific exception like InvalidTensorStateError
             raise RuntimeError("Accessing underlying C tensor of a released Python Tensor object.")
        return self._ptr.contents

    @property
    def shape(self):
        """Returns the shape of the tensor as a tuple."""
        c_tensor = self._c_tensor; ndim = c_tensor.ndim
        if ndim == 0: return () # Scalar tensor
        # Check if shape pointer is valid before accessing
        if not c_tensor.shape:
             # This indicates an internal inconsistency if ndim > 0
             print(f"Warning: Tensor has ndim={ndim} but NULL shape pointer.")
             return () # Or raise error
        # Access shape elements safely
        try:
            return tuple(c_tensor.shape[i] for i in range(ndim))
        except IndexError:
             # Should not happen if ndim is correct
             raise RuntimeError("Internal error accessing tensor shape elements.")


    @property
    def ndim(self):
        """Returns the number of dimensions of the tensor."""
        return self._c_tensor.ndim

    @property
    def dtype(self):
        """Returns the data type enum (int) of the tensor."""
        return self._c_tensor.dtype

    @property
    def device(self):
        """Returns the device enum (int) where the tensor resides."""
        return self._c_tensor.device

    @property
    def size(self):
        """Returns the total number of elements in the tensor."""
        return self._c_tensor.size

    @property
    def requires_grad(self):
        """Boolean indicating if the tensor requires gradient computation."""
        # Handle potential access after release
        return self._ptr.contents.requires_grad if hasattr(self, '_ptr') and self._ptr else False

    @requires_grad.setter
    def requires_grad(self, value):
        """Sets the requires_grad flag."""
        if not self._ptr: raise RuntimeError("Cannot operate on released Tensor.")
        if not isinstance(value, bool): raise TypeError("requires_grad must be a boolean value")

        # Check if it's a leaf tensor (no parents in autograd graph)
        is_leaf = not self._prev
        if value and not is_leaf:
             print("Warning: Setting requires_grad=True on a non-leaf tensor. "
                   "Gradients will not be computed for its history.")
            # In frameworks like PyTorch, this is generally disallowed or raises an error.
            # Depending on the desired behavior, you might want to raise RuntimeError here.

        # Update the C tensor's flag
        self._c_tensor.requires_grad = value

        # If requires_grad is turned off, nullify the gradient
        if not value:
            self.grad = None # Use the grad setter to handle C pointer release

    @property
    def grad(self):
        """
        Returns the gradient Tensor for this tensor.
        Returns None if requires_grad is False or no gradient has been computed.
        """
        if not self._ptr: return None # Released tensor has no grad

        c_tensor = self._c_tensor
        # Check if grad is needed and if the C grad pointer exists
        if not c_tensor.requires_grad or not c_tensor.grad:
            return None

        # IMPORTANT: The C grad tensor needs its ref count incremented
        # because we are creating a *new* Python Tensor wrapper for it.
        ff_lib.ff_tensor_retain(c_tensor.grad)

        # Wrap the C gradient pointer in a new Python Tensor object
        # The gradient tensor itself does not track graph history (_children/_op)
        return Tensor(c_tensor.grad)

    @grad.setter
    def grad(self, value):
        """Sets the gradient Tensor for this tensor."""
        if not self._ptr: raise RuntimeError("Cannot operate on released Tensor.")

        c_tensor = self._c_tensor

        if not c_tensor.requires_grad and value is not None:
            raise RuntimeError("Cannot set grad on a tensor that does not require gradients.")

        old_grad_ptr = c_tensor.grad # Get current C grad pointer
        new_grad_ptr = None # Initialize pointer for the new gradient

        if value is None:
            new_grad_ptr = None # Setting gradient to None
        elif isinstance(value, Tensor):
            if not value._ptr:
                raise ValueError("Cannot assign a released Tensor as gradient.")
            # Check shape and dtype compatibility
            if value.shape != self.shape or value.dtype != self.dtype:
                raise TypeError(f"Gradient shape/dtype mismatch: "
                                f"Got {value.shape}/{FFDataType.to_string(value.dtype)}, "
                                f"expected {self.shape}/{FFDataType.to_string(self.dtype)}.")
            # Get the C pointer from the assigned Tensor object
            new_grad_ptr = value._ptr
            # Increment the C ref count for the new gradient tensor, as self._c_tensor.grad
            # now holds a reference to it.
            ff_lib.ff_tensor_retain(new_grad_ptr)
        else:
            raise TypeError("Gradient must be a Tensor object or None.")

        # Update the C tensor's grad pointer
        c_tensor.grad = new_grad_ptr

        # Release the OLD C gradient tensor (if it existed)
        if old_grad_ptr:
            ff_lib.ff_tensor_release(old_grad_ptr)

    @property
    def T(self):
        """Returns the transpose of a 2D tensor."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not self._ptr: raise RuntimeError("Cannot operate on released Tensor.")
        if self.ndim != 2: raise ValueError(".T is only supported for 2D tensors currently.")

        result_ptr = ff_lib.ff_tensor_transpose(self._ptr)
        if not result_ptr: raise RuntimeError("C backend tensor transpose failed.")

        # Check if gradients are needed
        requires_grad = self.requires_grad and result_ptr.contents.requires_grad

        # Create the Python wrapper
        result_tensor = Tensor(result_ptr,
                               _children={self} if requires_grad else (),
                               _op='T' if requires_grad else '')
        if result_tensor._ptr: result_tensor._ptr.contents.requires_grad = requires_grad

        # Define backward function if needed
        if requires_grad:
            self_ptr_cap = self._ptr # Capture self pointer

            def _backward_fn():
                grad_output_ptr = result_tensor._c_tensor.grad
                if grad_output_ptr:
                    # Call the C backward function for transpose
                    # grad_input += transpose(grad_output)
                    # ensure_zero_grad is handled within C backward function ideally
                    ret = ff_lib.ff_tensor_transpose_backward(grad_output_ptr, self_ptr_cap)
                    if ret != 0: print(f"Warning: C backward function for T (transpose) reported error {ret}.")

            result_tensor._backward_fn = _backward_fn

        return result_tensor

    # --- Operator Overloading Wrappers (Autograd) ---
    def _op_wrapper(self, other, c_forward_func, c_backward_func, op_name,
                    c_fwd_scalar_func=None, c_bwd_scalar_func=None,
                    c_rfwd_scalar_func=None, c_rbwd_scalar_func=None,
                    bwd_needs_out=False, bwd_needs_in_a=True, bwd_needs_in_b=True):
        """Internal helper for binary/scalar ops with autograd."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not self._ptr: raise RuntimeError("Cannot operate on a released Tensor.")

        is_other_tensor = isinstance(other, Tensor)
        is_scalar = not is_other_tensor and isinstance(other, (int, float))
        scalar_val = float(other) if is_scalar else 0.0 # Ensure float for C

        self_ptr = self._ptr
        other_ptr = None
        result_ptr = None
        c_backward_to_use = None
        children = set()
        ctx = {} # Context for backward pass

        try:
            if is_other_tensor:
                if not other._ptr: raise RuntimeError("Operating with a released Tensor.")
                other_ptr = other._ptr
                # Call Tensor-Tensor forward function
                result_ptr = c_forward_func(self_ptr, other_ptr)
                c_backward_to_use = c_backward_func # Use Tensor-Tensor backward
                # Add dependencies for autograd
                if self.requires_grad: children.add(self)
                if other.requires_grad: children.add(other)
                # Save necessary pointers for backward pass
                if bwd_needs_in_a: ctx['in_a'] = self_ptr # Save C pointers
                if bwd_needs_in_b: ctx['in_b'] = other_ptr

            elif is_scalar:
                forward_func_to_call = None
                backward_func_to_call = None
                fwd_args = []
                saved_scalar = scalar_val

                # Handle reverse operations (like scalar / tensor)
                if op_name == 'rdiv':
                     if c_rfwd_scalar_func:
                        forward_func_to_call = c_rfwd_scalar_func
                        backward_func_to_call = c_rbwd_scalar_func
                        fwd_args = [scalar_val, self_ptr]
                        ctx = {'in_a': self_ptr, 'scalar': scalar_val} # Still save input tensor as 'in_a'
                        # bwd_needs_out=True # This flag is passed in, respect it
                     else: return NotImplemented # Reverse scalar op not implemented
                else: # Standard Tensor-Scalar op (add, mul, div, pow etc.)
                     if c_fwd_scalar_func:
                        forward_func_to_call = c_fwd_scalar_func
                        backward_func_to_call = c_bwd_scalar_func
                        fwd_args = [self_ptr, scalar_val]
                        ctx = {'in_a': self_ptr, 'scalar': scalar_val}
                     else: return NotImplemented # Forward scalar op not implemented

                # Call the selected C forward function
                if forward_func_to_call:
                    result_ptr = forward_func_to_call(*fwd_args)
                    c_backward_to_use = backward_func_to_call # Use scalar backward func
                    # Add dependency if input tensor requires grad
                    if self.requires_grad: children.add(self)
                # else case handled by NotImplemented returns above

            else: # Neither Tensor nor Scalar
                return NotImplemented

        except Exception as e:
            # Catch potential errors during C call or type checks
            raise RuntimeError(f"Error during forward C call for op '{op_name}': {e}") from e

        # Check if C function returned a valid pointer
        if not result_ptr:
            # Provide more context if possible (e.g., input shapes/types)
            op_details = f"self={repr(self)}"
            if is_other_tensor: op_details += f", other={repr(other)}"
            elif is_scalar: op_details += f", other={scalar_val}"
            raise RuntimeError(f"C backend for op '{op_name}' failed (returned NULL). Inputs: {op_details}")

        # Determine if the output tensor should require gradients
        # It requires grad if any child requires grad AND the operation supports it (C sets flag)
        result_req_grad_flag = result_ptr.contents.requires_grad # Read flag set by C
        output_requires_grad = bool(children) and result_req_grad_flag

        # Create the Python Tensor wrapper for the result
        result_tensor = Tensor(result_ptr,
                               _children=children if output_requires_grad else (),
                               _op=op_name if output_requires_grad else '')
        # Ensure requires_grad flag is consistent
        if result_tensor._ptr: result_tensor._ptr.contents.requires_grad = output_requires_grad


        # Define the backward function if needed
        if output_requires_grad and c_backward_to_use:
            # Save output tensor pointer if needed for backward calculation
            if bwd_needs_out: ctx['out'] = result_ptr # Save C pointer
            # Save context and capture necessary variables for the closure
            result_tensor._ctx = ctx
            # Capture C pointers from context immediately
            captured_ctx_ptrs = {k: v for k, v in ctx.items() if isinstance(v, FFTensor_p)}
            captured_scalar = ctx.get('scalar', None)
            # Capture the specific C backward function pointer
            captured_c_backward_func = c_backward_to_use

            def _backward_fn():
                # This function is called during result_tensor.backward()
                grad_output_ptr = result_tensor._c_tensor.grad
                if grad_output_ptr:
                    # Build arguments list for the C backward function dynamically
                    args = [grad_output_ptr] # First arg is always grad_output

                    # Restore context based on operation type
                    # Need to carefully match the C function signature
                    in_a_ptr = captured_ctx_ptrs.get('in_a', None)
                    in_b_ptr = captured_ctx_ptrs.get('in_b', None)
                    out_ptr = captured_ctx_ptrs.get('out', None)
                    scalar_arg = captured_scalar

                    # Determine args order based on C function signatures defined earlier
                    if is_other_tensor: # Tensor-Tensor op
                        # Standard: grad_out, input_a, input_b
                        if bwd_needs_in_a: args.append(in_a_ptr)
                        if bwd_needs_in_b: args.append(in_b_ptr)
                        # Some ops might need output too (rare for simple binary)
                        if bwd_needs_out: args.append(out_ptr)
                    elif is_scalar: # Tensor-Scalar op
                        if op_name == 'rdiv': # Special case: rdiv_scalar_backward(gO, iA, val, fO)
                           args.extend([in_a_ptr, scalar_arg, out_ptr])
                        elif op_name == 'pow': # Special case: pow_scalar_backward(gO, iA, exp)
                            args.extend([in_a_ptr, scalar_arg])
                        else: # Default scalar backward: func(gO, iA, scalar_val) or func(gO, iA)
                            if bwd_needs_in_a: args.append(in_a_ptr)
                            # Check if C function expects the scalar value
                            # Based on signatures: mul_s, div_s need it. add_s does not.
                            if op_name in ['mul', 'div']:
                                 args.append(scalar_arg)
                            # Handle cases like add_scalar_backward(gO, iA)
                            # Need to inspect c_bwd_scalar_func signature if more complex cases arise


                    try:
                        # Call the captured C backward function
                        ret = captured_c_backward_func(*args)
                    except Exception as e:
                        # Catch errors during the C call itself
                        print(f"Runtime Error calling C backward function for op '{op_name}': {e}")
                        print(f"Args passed: {[type(arg) for arg in args]}") # Log types
                        traceback.print_exc()
                        ret = -99 # Indicate failure

                    if ret != 0:
                         # C function indicated an error (e.g., memory allocation failed)
                         print(f"Warning: C backward function for op '{op_name}' reported error code {ret}.")

            result_tensor._backward_fn = _backward_fn

        return result_tensor


    def _unary_op_wrapper(self, c_forward_func, c_backward_func, op_name, bwd_needs_out=False, bwd_needs_in=False):
        """Internal helper for unary ops with autograd."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not self._ptr: raise RuntimeError("Cannot operate on a released Tensor.")

        # Call C forward function
        result_ptr = c_forward_func(self._ptr)
        if not result_ptr: raise RuntimeError(f"C backend for op '{op_name}' failed (returned NULL). Input: {repr(self)}")

        # Determine if output requires grad
        result_req_grad_flag = result_ptr.contents.requires_grad
        output_requires_grad = self.requires_grad and result_req_grad_flag

        # Create Python wrapper
        result_tensor = Tensor(result_ptr,
                               _children={self} if output_requires_grad else (),
                               _op=op_name if output_requires_grad else '')
        if result_tensor._ptr: result_tensor._ptr.contents.requires_grad = output_requires_grad

        # Define backward function if needed
        if output_requires_grad and c_backward_func:
            ctx_dict = {}
            self_ptr_cap = self._ptr # Capture input pointer

            if bwd_needs_in: ctx_dict['in_a'] = self_ptr_cap # Store captured C pointer
            if bwd_needs_out: ctx_dict['out'] = result_ptr # Store result C pointer
            result_tensor._ctx = ctx_dict # Attach context

            captured_ctx_ptrs = {k: v for k, v in ctx_dict.items()}
            captured_c_backward_func = c_backward_func

            def _backward_fn():
                grad_output_ptr = result_tensor._c_tensor.grad
                if grad_output_ptr:
                    args = [grad_output_ptr] # Always start with grad_output

                    input_a_ptr = captured_ctx_ptrs.get('in_a', None) # Use None if not needed/saved
                    output_ptr = captured_ctx_ptrs.get('out', None) # Use None if not needed/saved

                    # Construct args based on C function signature requirements
                    # Signatures: tanh/exp/sigmoid_backward(gO, iA, fO)
                    #             relu/abs/mean_backward(gO, iA)
                    #             transpose_backward(gO, iA)
                    if op_name in ['tanh', 'exp', 'sigmoid']:
                        if not input_a_ptr or not output_ptr: raise RuntimeError(f"Backward for {op_name} needs input and output context.")
                        args.extend([input_a_ptr, output_ptr])
                    elif op_name in ['relu', 'abs', 'mean', 'T']: # T uses transpose_backward
                        if not input_a_ptr: raise RuntimeError(f"Backward for {op_name} needs input context.")
                        args.append(input_a_ptr)
                    else: # Default assumption: backward(gO, iA)
                         if not input_a_ptr: raise RuntimeError(f"Backward for {op_name} needs input context.")
                         args.append(input_a_ptr)

                    try:
                        ret = captured_c_backward_func(*args)
                    except Exception as e:
                        print(f"Runtime Error calling C backward function for op '{op_name}': {e}")
                        print(f"Args passed: {[type(arg) for arg in args]}")
                        traceback.print_exc()
                        ret = -99
                    if ret != 0:
                         print(f"Warning: C backward function for op '{op_name}' reported error code {ret}.")

            result_tensor._backward_fn = _backward_fn

        return result_tensor

    def _scalar_arg_op_wrapper(self, c_forward_func, c_backward_func, op_name, scalar_args, bwd_needs_out=False, bwd_needs_in=True):
        """Internal helper for ops taking tensor and fixed scalar args (e.g., clip, pow)."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not self._ptr: raise RuntimeError("Cannot operate on a released Tensor.")

        # Prepare arguments for C forward function: tensor_ptr, scalar1, scalar2, ...
        fwd_args = [self._ptr] + [float(a) for a in scalar_args] # Ensure scalars are floats

        # Call C forward function
        result_ptr = c_forward_func(*fwd_args)
        if not result_ptr: raise RuntimeError(f"C backend for op '{op_name}' failed (returned NULL). Input: {repr(self)}, Args: {scalar_args}")

        # Determine if output requires grad
        result_req_grad_flag = result_ptr.contents.requires_grad
        output_requires_grad = self.requires_grad and result_req_grad_flag

        # Create Python wrapper
        result_tensor = Tensor(result_ptr,
                               _children={self} if output_requires_grad else (),
                               _op=op_name if output_requires_grad else '')
        if result_tensor._ptr: result_tensor._ptr.contents.requires_grad = output_requires_grad

        # Define backward function if needed
        if output_requires_grad and c_backward_func:
            ctx_dict = {'scalar_args': [float(a) for a in scalar_args]} # Store scalar args
            self_ptr_cap = self._ptr # Capture input tensor pointer

            if bwd_needs_in: ctx_dict['in_a'] = self_ptr_cap
            if bwd_needs_out: ctx_dict['out'] = result_ptr
            result_tensor._ctx = ctx_dict

            captured_ctx_ptrs = {k: v for k, v in ctx_dict.items() if isinstance(v, FFTensor_p)}
            captured_scalars = ctx_dict.get('scalar_args', [])
            captured_c_backward_func = c_backward_func

            def _backward_fn():
                grad_output_ptr = result_tensor._c_tensor.grad
                if grad_output_ptr:
                    args = [grad_output_ptr] # Start with grad_output

                    input_a_ptr = captured_ctx_ptrs.get('in_a', None)
                    output_ptr = captured_ctx_ptrs.get('out', None)
                    scalar_args_list = captured_scalars

                     # Construct args based on C backward signature
                     # Signatures: pow_scalar_backward(gO, iA, exponent)
                     #             clip_backward(gO, iA, min_val, max_val)
                    if bwd_needs_in:
                        if not input_a_ptr: raise RuntimeError(f"Backward for {op_name} needs input context.")
                        args.append(input_a_ptr)
                    if bwd_needs_out:
                         if not output_ptr: raise RuntimeError(f"Backward for {op_name} needs output context.")
                         args.append(output_ptr)

                    # Append scalar arguments
                    args.extend(scalar_args_list)

                    try:
                        ret = captured_c_backward_func(*args)
                    except Exception as e:
                        print(f"Runtime Error calling C backward function for op '{op_name}': {e}")
                        print(f"Args passed: {[type(arg) for arg in args]}")
                        traceback.print_exc()
                        ret = -99
                    if ret != 0:
                         print(f"Warning: C backward function for op '{op_name}' reported error code {ret}.")

            result_tensor._backward_fn = _backward_fn

        return result_tensor


    # --- Operator Methods ---
    # Each operator calls the appropriate wrapper with C function pointers

    def __add__(self, other):
        return self._op_wrapper(other, ff_lib.ff_tensor_add, ff_lib.ff_tensor_add_backward, "add",
                                c_fwd_scalar_func=ff_lib.ff_tensor_add_scalar,
                                c_bwd_scalar_func=ff_lib.ff_tensor_add_scalar_backward)
                                # bwd_needs_in_a/b default True is correct for add

    def __radd__(self, other): # other + self
        return self.__add__(other) # Addition is commutative

    def __sub__(self, other): # self - other
         # Currently assumes no scalar version defined in C for sub, only Tensor-Tensor
        return self._op_wrapper(other, ff_lib.ff_tensor_sub, ff_lib.ff_tensor_sub_backward, "sub")
                                # No scalar funcs passed

    def __rsub__(self, other): # other - self
        # Implement as other + (-1.0 * self)
        neg_self = self * -1.0 # Assumes __mul__ handles scalar multiplication
        return neg_self + other # Use __add__

    def __mul__(self, other): # self * other (element-wise)
         return self._op_wrapper(other, ff_lib.ff_tensor_mul, ff_lib.ff_tensor_mul_backward, "mul",
                                c_fwd_scalar_func=ff_lib.ff_tensor_mul_scalar,
                                c_bwd_scalar_func=ff_lib.ff_tensor_mul_scalar_backward)

    def __rmul__(self, other): # other * self
        return self.__mul__(other) # Multiplication is commutative

    def __matmul__(self, other): # self @ other
        # Assumes only Tensor-Tensor matmul
        return self._op_wrapper(other, ff_lib.ff_tensor_matmul, ff_lib.ff_tensor_matmul_backward, "matmul")

    def __truediv__(self, other): # self / other
        # Only handles Tensor / scalar currently
        return self._op_wrapper(other, None, None, "div", # No Tensor-Tensor div
                                c_fwd_scalar_func=ff_lib.ff_tensor_div_scalar,
                                c_bwd_scalar_func=ff_lib.ff_tensor_div_scalar_backward)

    def __rtruediv__(self, other): # other / self
         # Only handles scalar / Tensor currently
         return self._op_wrapper(other, None, None, "rdiv", # No Tensor-Tensor div
                                c_rfwd_scalar_func=ff_lib.ff_tensor_rdiv_scalar,
                                c_rbwd_scalar_func=ff_lib.ff_tensor_rdiv_scalar_backward,
                                bwd_needs_out=True) # rdiv backward needs forward output

    def __pow__(self, value): # self ** value
        # Only handles Tensor ** scalar currently
        if not isinstance(value, (int, float)): return NotImplemented
        # Uses scalar_arg_op_wrapper because pow takes tensor + scalar argument
        return self._scalar_arg_op_wrapper(ff_lib.ff_tensor_pow_scalar, # C forward func
                                           ff_lib.ff_tensor_pow_scalar_backward, # C backward func
                                           "pow", # op name
                                           scalar_args=[value], # List of scalar args
                                           bwd_needs_in=True) # pow backward needs input tensor

    def __gt__(self, value): # self > value (scalar only)
        if not isinstance(value,(int,float)): return NotImplemented
        if not _LIB_LOADED or not self._ptr: raise RuntimeError("Tensor/Backend invalid.")
        # Comparison ops generally don't support autograd
        res_ptr = ff_lib.ff_tensor_gt_scalar(self._ptr, float(value))
        if not res_ptr: raise RuntimeError("C gt_scalar failed.")
        # Result is bool, requires_grad=False
        return Tensor(res_ptr)

    def __lt__(self, value): # self < value (scalar only)
        if not isinstance(value,(int,float)): return NotImplemented
        if not _LIB_LOADED or not self._ptr: raise RuntimeError("Tensor/Backend invalid.")
        res_ptr = ff_lib.ff_tensor_lt_scalar(self._ptr, float(value))
        if not res_ptr: raise RuntimeError("C lt_scalar failed.")
        return Tensor(res_ptr)


    # --- Unary Methods ---
    def mean(self):
        return self._unary_op_wrapper(ff_lib.ff_tensor_mean, ff_lib.ff_tensor_mean_backward, "mean",
                                      bwd_needs_in=True) # mean backward needs input tensor

    def tanh(self):
        return self._unary_op_wrapper(ff_lib.ff_tensor_tanh, ff_lib.ff_tensor_tanh_backward, "tanh",
                                      bwd_needs_out=True, bwd_needs_in=True) # tanh backward needs input and output

    def exp(self):
        return self._unary_op_wrapper(ff_lib.ff_tensor_exp, ff_lib.ff_tensor_exp_backward, "exp",
                                      bwd_needs_out=True, bwd_needs_in=True) # exp backward needs input and output

    def sigmoid(self):
         return self._unary_op_wrapper(ff_lib.ff_tensor_sigmoid, ff_lib.ff_tensor_sigmoid_backward, "sigmoid",
                                       bwd_needs_out=True, bwd_needs_in=True) # sigmoid backward needs input and output

    def relu(self):
        return self._unary_op_wrapper(ff_lib.ff_tensor_relu, ff_lib.ff_tensor_relu_backward, "relu",
                                      bwd_needs_in=True) # relu backward needs input tensor

    def abs(self):
         return self._unary_op_wrapper(ff_lib.ff_tensor_abs, ff_lib.ff_tensor_abs_backward, "abs",
                                       bwd_needs_in=True) # abs backward needs input tensor (for sign)

    def clip(self, min_val, max_val):
         """Applies element-wise clipping: min(max(tensor, min_val), max_val)."""
         return self._scalar_arg_op_wrapper(ff_lib.ff_tensor_clip, ff_lib.ff_tensor_clip_backward, "clip",
                                            scalar_args=[min_val, max_val], # Pass min/max as list
                                            bwd_needs_in=True) # clip backward needs input tensor

    def outer(self, other):
        """Computes the outer product of two 1D tensors."""
        if not isinstance(other, Tensor): return NotImplemented
        if self.ndim != 1 or other.ndim != 1: raise ValueError("Outer product requires 1D tensors.")
        # Use the binary op wrapper
        return self._op_wrapper(other, ff_lib.ff_tensor_outer, ff_lib.ff_tensor_outer_backward, "outer",
                                bwd_needs_in_a=True, bwd_needs_in_b=True) # Outer backward needs both inputs


    # --- Autograd ---
    def backward(self):
        """Performs backpropagation starting from this Tensor (usually scalar loss)."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not self._ptr: raise RuntimeError("Cannot call backward on a released Tensor.")
        if not self.requires_grad:
            print("Warning: backward() called on a Tensor that does not require gradients.")
            return # No work to do

        # Sanity check: Usually called on a scalar loss tensor
        if self.size != 1:
            print("Warning: backward() called on non-scalar Tensor. "
                  "Implicitly using gradient of ones matching shape, which may not be intended.")
            # TODO: Optionally allow passing explicit gradient argument `backward(gradient=...)`
            # For now, we proceed assuming implicit grad of 1.0

        # --- Topological Sort (DFS) ---
        topo = []
        visited = set()
        # Use deque for efficiency, though list append/pop works too for moderate graphs
        # stack = deque([self]) # Alternative iterative DFS
        def build_topo(v_tensor):
            v_id = id(v_tensor) # Use object ID for visited set
            if v_id not in visited:
                visited.add(v_id)
                # Only explore parents if the node requires grad
                if hasattr(v_tensor, '_prev') and v_tensor.requires_grad:
                    # Iterate over copies of parent pointers for safety if needed
                    # Though modifying _prev during sort is not expected
                    parents_to_visit = list(v_tensor._prev)
                    for parent_tensor in parents_to_visit:
                        # Check if parent is valid and requires grad before recursing
                        if parent_tensor._ptr and parent_tensor.requires_grad:
                            build_topo(parent_tensor)
                        # Else: Stop recursion path if parent doesn't need grad
                topo.append(v_tensor) # Add node AFTER visiting children

        try:
            build_topo(self)
        except Exception as e:
            print(f"Error building topological sort during backward pass: {e}")
            traceback.print_exc()
            raise # Re-raise after logging

        # --- Initialize gradient for the starting tensor (self) to 1.0 ---
        # Ensure gradient tensor exists and is zeroed
        if ff_lib.ff_tensor_ensure_zero_grad(self._ptr) != 0:
            raise RuntimeError("Backward Error: Could not ensure gradient tensor for the starting node.")

        # Check if grad pointer is valid after ensure_zero_grad
        grad_ptr = self._c_tensor.grad
        if not grad_ptr:
             raise RuntimeError("Backward Error: Starting node gradient pointer is NULL after ensure_zero_grad.")

        # Fill the starting gradient tensor with 1.0
        if ff_lib.ff_tensor_fill(grad_ptr, 1.0) != 0:
            raise RuntimeError("Backward Error: Could not fill starting node gradient with 1.0.")


        # --- Backpropagate through the sorted graph ---
        # Iterate in reverse topological order (from output back to inputs)
        for node_tensor in reversed(topo):
            # Check if the node has a backward function defined (i.e., it resulted from an op)
            if hasattr(node_tensor, '_backward_fn') and callable(node_tensor._backward_fn):
                try:
                    # Execute the backward function (which calls the C kernel)
                    node_tensor._backward_fn()
                except Exception as e:
                    op_name = getattr(node_tensor, '_op', '<Unknown Op>')
                    print(f"Error executing backward function for node created by op '{op_name}': {e}")
                    # Print tensor details that caused the issue
                    print(f"Node Tensor: {repr(node_tensor)}")
                    if node_tensor._ctx: print(f"Node Context: {node_tensor._ctx}")
                    traceback.print_exc()
                    # Decide whether to continue or stop backward pass on error
                    # For debugging, maybe continue? For production, maybe stop?
                    # raise # Optionally re-raise to halt execution


    # --- Data Access & Representation ---
    def numpy(self):
        """
        Returns a NumPy array containing a copy of the tensor's data.
        Currently only supports CPU tensors.
        """
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not self._ptr: raise RuntimeError("Cannot get NumPy data from a released Tensor.")

        c_tensor=self._c_tensor
        np_dtype=FFDataType.to_numpy(c_tensor.dtype)
        shape=self.shape # Use property to get tuple

        if c_tensor.device != FFDevice.CPU:
            raise NotImplementedError("numpy() method currently only supports CPU tensors.")
        if np_dtype is None:
            raise TypeError(f"Cannot convert tensor with unsupported dtype {c_tensor.dtype} to NumPy.")

        # Handle empty tensor case
        if c_tensor.size == 0:
            return np.zeros(shape, dtype=np_dtype) # Return empty NumPy array with correct shape/dtype

        # Check data pointer validity
        c_data_ptr = c_tensor.data
        nbytes = c_tensor.nbytes
        if not c_data_ptr:
             # Should not happen if size > 0 and tensor created correctly
             raise RuntimeError("Internal Tensor state invalid: NULL data pointer for non-empty tensor.")

        # Create an empty NumPy array with the correct shape and dtype
        try:
            np_array = np.empty(shape, dtype=np_dtype)
        except ValueError as e:
            # Catch potential errors from invalid shape/dtype combinations
            raise ValueError(f"Failed to create NumPy array shape={shape} dtype={np_dtype}: {e}")

        # Copy data from C buffer to NumPy buffer
        # Requires C tensor data to be accessible (i.e., on CPU)
        try:
            ctypes.memmove(np_array.ctypes.data, c_data_ptr, nbytes)
        except Exception as e:
             # Catch potential memmove errors (though unlikely with preceding checks)
             raise RuntimeError(f"Memory copy failed during numpy() conversion: {e}")

        return np_array

    def item(self):
        """
        Returns the value of this tensor as a standard Python number.
        Only works for tensors with one element (scalars) residing on the CPU.
        """
        if not self._ptr: raise RuntimeError("Cannot get item from a released Tensor.")

        c_tensor = self._c_tensor # Access C struct contents

        # Check if it's a scalar tensor
        if c_tensor.size != 1:
            raise ValueError(f"Only tensors with one element can be converted to Python scalars using item(), but got size {c_tensor.size}")

        # Check if it's on the CPU
        if c_tensor.device != FFDevice.CPU:
            raise NotImplementedError("item() method currently only supports CPU tensors.")

        # Get the appropriate ctypes type
        c_dtype_ptr = FFDataType.to_ctype(c_tensor.dtype)
        if c_dtype_ptr is None:
            raise TypeError(f"Unsupported dtype ({FFDataType.to_string(c_tensor.dtype)}) for item()")

        # Check if data pointer is valid
        if c_tensor.data is None:
             raise ValueError("Cannot get item from tensor with NULL data pointer (even for size 1).")

        # Cast the void* data pointer to the correct ctypes pointer type
        value_ptr = ctypes.cast(c_tensor.data, ctypes.POINTER(c_dtype_ptr))

        # Dereference the pointer to get the value
        scalar_value = value_ptr[0]

        # Convert C types to standard Python types if needed (e.g., ctypes.c_float -> float)
        # This happens automatically for standard numeric ctypes
        return scalar_value


    def __repr__(self):
        """Provides a string representation of the Tensor."""
        # Handle case where tensor might be released or partially initialized
        if not hasattr(self, '_ptr') or not self._ptr: return "Tensor(<released>)"

        try:
            # Access properties safely
            c_tensor = self._c_tensor # Access C struct only once
            shape_str = str(self.shape) # Use shape property
            dtype_str = FFDataType.to_string(self.dtype)
            device_str = FFDevice.to_string(self.device)
            grad_str = ", requires_grad=True" if self.requires_grad else ""
            # Access C ref_count directly for debugging purposes
            ref_count_str = f", c_ref={c_tensor.ref_count}"

            data_str = ""
            # Try to show data preview for small CPU tensors
            if c_tensor.device == FFDevice.CPU and c_tensor.size > 0 and c_tensor.size <= 10:
                try:
                    # Use item() for scalars, numpy() for small arrays
                    if c_tensor.size == 1:
                         np_data_repr = repr(self.item())
                    else:
                         np_data_repr = repr(self.numpy())
                    data_str = f", data={np_data_repr}"
                except Exception as e:
                    # Avoid errors in repr if numpy/item fails
                    data_str = f", data=<preview failed: {type(e).__name__}>"
            elif c_tensor.size > 10:
                 data_str = ", data=<...>" # Indicate large tensor

            return (f"Tensor({shape_str}{data_str}, dtype={dtype_str}, "
                    f"device='{device_str}'{grad_str}{ref_count_str})")

        except Exception as e:
            # Catch any errors during property access (e.g., if _ptr becomes invalid)
            return f"Tensor(<error accessing properties: {e}>)"


# --- NN Components ---
# Based on typical PyTorch-like structure

class Parameter(Tensor):
    """
    A Tensor subclass that is automatically registered as a parameter
    when assigned as an attribute of a Module. Parameters require gradients by default.
    """
    def __init__(self, c_tensor_ptr):
        # Initialize as a Tensor first
        super().__init__(c_tensor_ptr, _children=(), _op='Parameter')
        # Ensure requires_grad is set to True after initialization
        if self._ptr and not self._c_tensor.requires_grad:
             self._c_tensor.requires_grad = True

    @classmethod
    def from_numpy(cls, np_array: np.ndarray, device=FFDevice.CPU):
        """Creates a Parameter from a NumPy array."""
        # Create a Tensor first, ensuring requires_grad=True
        t = Tensor.from_numpy(np_array, device=device, requires_grad=True)
        # Create Parameter using the Tensor's pointer
        p = cls(t._ptr)
        # Nullify the original Tensor's pointer to prevent double release in __del__
        t._ptr = None
        return p

    @classmethod
    def from_random(cls, shape, dtype=FFDataType.FLOAT32, device=FFDevice.CPU):
        """Creates a Parameter with Kaiming uniform initialization."""
        # Kaiming/He uniform initialization: sqrt(6 / fan_in)
        # Determine fan_in (usually input features)
        if len(shape) == 0: fan_in = 1 # Scalar
        elif len(shape) == 1: fan_in = shape[0] # Vector
        else: fan_in = shape[1] # Assume (out, in) or (N, C, H, W) -> use second dim

        limit = np.sqrt(6.0 / fan_in) if fan_in > 0 else 0.0
        np_dtype = FFDataType.to_numpy(dtype)
        if np_dtype is None: raise TypeError(f"Invalid dtype for random init: {dtype}")

        # Create random NumPy array
        np_data = np.random.uniform(-limit, limit, size=shape).astype(np_dtype)

        # Convert to Parameter
        return cls.from_numpy(np_data, device=device)

    @classmethod
    def zeros(cls, shape, dtype=FFDataType.FLOAT32, device=FFDevice.CPU):
        """Creates a Parameter initialized with zeros."""
        t = Tensor.zeros(shape, dtype=dtype, device=device, requires_grad=True)
        p = cls(t._ptr)
        t._ptr = None # Prevent double release
        return p

    def __repr__(self):
        # Modify the Tensor repr slightly
        base = super().__repr__()
        return base.replace("Tensor(", "Parameter(", 1)


class Module:
    """Base class for all neural network modules."""
    def __init__(self):
        # Use private names to avoid clashes with potential parameter/submodule names
        self._parameters = {}
        self._modules = {}
        self._initialized = False # Flag to control __setattr__ behavior during init

    def __setattr__(self, name, value):
        """
        Registers Parameters and Modules assigned as attributes.
        """
        # Allow setting internal attributes during __init__ before _initialized is True
        if not getattr(self, '_initialized', False):
             if name in ('_parameters', '_modules', '_initialized'):
                super().__setattr__(name, value)
                return
             # If setting other attributes during init, handle normally for now
             # Or raise error if only params/modules allowed before _complete_init?

        # Check if the value is a Parameter or Module after initialization
        param = self._parameters.get(name) # Check if name already exists as param/module
        module = self._modules.get(name)

        if isinstance(value, Parameter):
            if module is not None: del self._modules[name] # Remove if name was previously a module
            self._parameters[name] = value
        elif isinstance(value, Module):
            if param is not None: del self._parameters[name] # Remove if name was previously a param
            self._modules[name] = value
        else: # Assigning a non-Parameter/non-Module value
             # If the name previously held a param/module, remove it from tracking dicts
            if param is not None: del self._parameters[name]
            if module is not None: del self._modules[name]

        # Set the attribute on the object itself
        super().__setattr__(name, value)

    def _complete_init(self):
         """Call this at the end of subclass __init__ to enable attribute tracking."""
         self._initialized = True

    def __getattr__(self, name):
         """Retrieve parameters/modules if accessed directly."""
         # Check internal dicts first to allow access via self.param_name
         if '_parameters' in self.__dict__ and name in self._parameters:
             return self._parameters[name]
         if '_modules' in self.__dict__ and name in self._modules:
             return self._modules[name]
         # If not found, raise standard AttributeError
         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def parameters(self, recurse=True):
        """Returns an iterator over module parameters."""
        # Yield parameters directly held by this module
        for name, param in self._parameters.items():
            if param is not None: # Should always be Parameter, but check
                yield param
        # Recursively yield parameters from submodules if recurse=True
        if recurse:
            for name, module in self._modules.items():
                # Use 'yield from' to delegate iteration
                yield from module.parameters(recurse=True)

    def zero_grad(self):
        """Sets gradients of all parameters to None."""
        for p in self.parameters():
            # Only modify grad if it requires grad computation
            if p.requires_grad:
                 # Use the property setter which handles C pointer release
                p.grad = None

    def __call__(self, *args, **kwargs):
        """Allows calling the module instance like a function."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call. Should be overridden by subclasses."""
        raise NotImplementedError


class Linear(Module):
    """Applies a linear transformation: y = xA^T + b (or y = x @ W + b)."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight Parameter: shape (in_features, out_features) for x @ W
        self.weight = Parameter.from_random((in_features, out_features))

        # Initialize bias Parameter if requested: shape (out_features,)
        if bias:
            self.bias = Parameter.zeros((out_features,))
        else:
            self.bias = None # Register as None if no bias

        self._complete_init() # Mark initialization complete

    def forward(self, input_tensor: Tensor) -> Tensor:
        # Check input tensor dimensions if necessary (e.g., input_tensor.ndim >= 1)
        # Perform matrix multiplication: input @ weight
        # Input shape: (BatchSize, in_features) or (in_features,)
        # Weight shape: (in_features, out_features)
        # Output shape: (BatchSize, out_features) or (out_features,)
        try:
            output = input_tensor @ self.weight
        except RuntimeError as e:
             print(f"ERROR during Linear matmul: {e}")
             print(f"Input shape: {input_tensor.shape}")
             print(f"Weight shape: {self.weight.shape}")
             raise

        # Add bias if it exists (handles broadcasting)
        if self.bias is not None:
            try:
                 # Output (Batch, out) + Bias (out,) -> Broadcasting adds bias to each row
                output = output + self.bias
            except RuntimeError as e:
                print(f"ERROR during Linear bias addition: {e}")
                print(f"Output shape: {output.shape}")
                print(f"Bias shape: {self.bias.shape}")
                raise
        return output


class MSELoss(Module):
    """Computes the Mean Squared Error loss."""
    def __init__(self):
        super().__init__()
        self._complete_init()

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        # Ensure input and target shapes match? Or rely on subtraction op to check.
        diff = input_tensor - target_tensor
        # Use the power operator ** instead of .pow() method
        squared_diff = diff ** 2
        loss = squared_diff.mean() # Compute mean over all elements
        return loss


class SGD:
    """Implements Stochastic Gradient Descent optimizer."""
    def __init__(self, params, lr=0.01):
        # Ensure params is an iterable (e.g., list or generator)
        try:
            self.params = list(params) # Convert generator to list to reuse
        except TypeError:
            raise TypeError("Optimizer 'params' argument must be an iterable (e.g., a list or generator).")

        self.lr = lr
        if not self.params:
             print("Warning: SGD optimizer initialized with no parameters.")

    def step(self):
        """Performs a single optimization step."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")

        # Temporarily ignore potential invalid value warnings from NumPy (e.g., NaN grads)
        # Though C backend should ideally handle this.
        with np.errstate(invalid='ignore'):
            for p in self.params:
                # Only update parameters that require gradients and have a gradient computed
                if p.requires_grad and p.grad is not None:
                    grad_tensor = p.grad # Get the gradient Tensor object
                    # Check if the grad tensor itself is valid (pointer exists)
                    if not grad_tensor._ptr:
                        print(f"Warning: Parameter has requires_grad=True but its grad Tensor is invalid/released. Skipping update.")
                        continue

                    # Call the C SGD step function
                    # ff_optim_sgd_step(param_ptr, grad_ptr, learning_rate)
                    ret = ff_lib.ff_optim_sgd_step(p._ptr, grad_tensor._ptr, self.lr)

                    if ret != 0:
                        # C function reported an error
                        print(f"Warning: C backend SGD step failed for parameter {repr(p)} (error code: {ret}).")
                        # Optionally, inspect grad_tensor value here if needed
                        # print(f"Gradient value (might be large/NaN): {repr(grad_tensor)}")
                        # Decide if training should halt on optimizer errors.

    def zero_grad(self):
        """Sets the gradients of all optimized parameters to None."""
        # Use the Module's zero_grad logic by iterating through stored params
        for p in self.params:
            # Using p.grad = None correctly handles releasing the old C grad tensor
            p.grad = None


# --- Example Usage ---
if __name__ == "__main__":
    if not _LIB_LOADED: exit(1) # Exit if library failed to load

    print("\n--- FusionFlow Core (Final Fixes Applied) ---")
    print("Includes fixes for item(), broadcasting add, NameError, indentation, MSELoss pow.")
    print("Ready to run tests or neuralengine_ff.py.")

    # --- Autograd Test ---
    print("\n--- Running Autograd Test ---")
    a, b, c, ab, y = None, None, None, None, None # Initialize for finally block
    try:
        # Create tensors requiring gradients
        a = Tensor.from_numpy(np.array([2.], dtype=np.float32), requires_grad=True)
        b = Tensor.from_numpy(np.array([3.], dtype=np.float32), requires_grad=True)
        c = Tensor.from_numpy(np.array([1.], dtype=np.float32), requires_grad=True)

        # Perform operations to build computation graph: y = a * b + c
        ab = a * b
        y = ab + c
        print(f"y={repr(y)}")

        # Clear any previous gradients (important if re-running)
        a.grad = None; b.grad = None; c.grad = None

        # Perform backpropagation
        print("Calling y.backward()...");
        y.backward()
        print("Backward pass complete.")

        # Check gradients using item()
        a_grad = a.grad.item() if a.grad else None
        b_grad = b.grad.item() if b.grad else None
        c_grad = c.grad.item() if c.grad else None

        # Verify gradients (dy/da = b = 3, dy/db = a = 2, dy/dc = 1)
        print(f"dy/da (exp 3.0): {a_grad}")
        print(f"dy/db (exp 2.0): {b_grad}")
        print(f"dy/dc (exp 1.0): {c_grad}")

        # Add assertions for automated testing
        assert a_grad is not None and abs(a_grad - 3.0) < 1e-6, "Gradient check for 'a' failed!"
        assert b_grad is not None and abs(b_grad - 2.0) < 1e-6, "Gradient check for 'b' failed!"
        assert c_grad is not None and abs(c_grad - 1.0) < 1e-6, "Gradient check for 'c' failed!"
        print("Simple autograd test PASSED.")

    except Exception as e:
        print(f"Autograd Test Error: {type(e).__name__} - {e}")
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        # Explicitly delete tensors to trigger __del__ and release C memory
        print("\n--- Autograd Test Cleanup ---")
        del a, b, c, ab, y # Delete in reverse order of potential dependency? Not strictly necessary with GC.
        gc.collect() # Encourage garbage collection

    # --- Identity Training Test ---
    print("\n--- Running Identity Training Test (with Autograd) ---")
    # Initialize variables to ensure they exist for the finally block
    X, Y, model, criterion, optimizer = None, None, None, None, None
    final_outputs = None
    try:
        # --- Config ---
        DIM=32
        BATCH_SIZE=4
        EPOCHS=20 # Increased epochs slightly
        LR=0.001

        # --- Data ---
        print(f"\nPreparing data (Batch Size: {BATCH_SIZE}, Dim: {DIM})...")
        np_x = np.random.randn(BATCH_SIZE, DIM).astype(np.float32)
        # Target Y is the same as input X for identity mapping
        X = Tensor.from_numpy(np_x)
        Y = Tensor.from_numpy(np_x)
        # Input X does not require grad for this test, Y never requires grad

        # --- Model, Loss, Optimizer ---
        print("Initializing Model, Loss, Optimizer...")
        model = Linear(DIM, DIM)
        criterion = MSELoss()
        optimizer = SGD(model.parameters(), lr=LR) # Pass model params to optimizer

        print(f"Model Layer W Shape: {model.weight.shape}") # Check weight init shape
        print("-" * 30)
        print("Starting training loop...")

        initial_loss=-1.0
        final_loss=-1.0

        # --- Training Loop ---
        for epoch in range(EPOCHS):
            # 1. Forward pass
            outputs = model(X) # Equivalent to model.forward(X)
            loss_tensor = criterion(outputs, Y) # Compute loss
            loss_val = loss_tensor.item() # Get Python scalar value

            if epoch == 0: initial_loss = loss_val

            # 2. Backward pass (compute gradients)
            optimizer.zero_grad() # Zero gradients before backward pass
            loss_tensor.backward() # Compute gradients starting from loss

            # 3. Optimizer step (update weights)
            optimizer.step() # Update parameters based on computed gradients

            # --- Logging ---
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss_val:.6f}")

            if epoch == EPOCHS - 1: final_loss = loss_val # Store final loss

            # Check for NaN/Inf loss
            if np.isnan(loss_val) or np.isinf(loss_val):
                print("Error: Loss became NaN or Inf. Stopping training.")
                break

        print("-" * 30)
        print("Training loop finished.")

        # --- Evaluation ---
        if final_loss >= 0 and initial_loss >= 0:
            print(f"Initial Loss: {initial_loss:.6f}")
            print(f"Final Loss:   {final_loss:.6f}")
            # Check if loss decreased significantly (e.g., by 20%)
            if final_loss < initial_loss * 0.8:
                 print("PASSED: Loss successfully decreased.")
            else:
                 print("FAILED: Loss did not decrease significantly!")
        else:
             print("Error: Could not compare initial and final loss values.")

        # Optional: Check how close the final output is to the input
        final_outputs = model(X) # Get output after training
        np_final_outputs = final_outputs.numpy() # Convert to NumPy
        mean_abs_diff = np.mean(np.abs(np_x - np_final_outputs))
        print(f"\nMean Absolute Difference (Input vs Final Output): {mean_abs_diff:.6f}")
        print("-" * 30)

    except Exception as e:
        print(f"Training Test Error: {type(e).__name__} - {e}")
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        print("\n--- Training Test Cleanup ---")
        # Explicitly delete objects involved in the test
        del X, Y, model, criterion, optimizer, final_outputs
        gc.collect() # Encourage garbage collection
