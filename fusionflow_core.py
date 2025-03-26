# fusionflow_core.py
import ctypes
import os
import numpy as np
import atexit # To help ensure cleanup on exit, though __del__ is primary
import gc     # For potential explicit collection during debugging

print("--- Loading FusionFlow C Backend ---")

# --- 1. Define Constants (matching C enums) ---
class FFDataType:
    FLOAT32 = 0
    FLOAT64 = 1
    INT32 = 2
    INT64 = 3
    BOOL = 4
    # Add mappings for convenience
    _map = {
        FLOAT32: "float32", FLOAT64: "float64",
        INT32: "int32", INT64: "int64", BOOL: "bool"
    }
    _np_map = {
        FLOAT32: np.float32, FLOAT64: np.float64,
        INT32: np.int32, INT64: np.int64, BOOL: np.bool_
    }
    _ctype_map = {
        FLOAT32: ctypes.c_float, FLOAT64: ctypes.c_double,
        INT32: ctypes.c_int32, INT64: ctypes.c_int64, BOOL: ctypes.c_bool
    }
    @classmethod
    def to_string(cls, dtype_enum):
        return cls._map.get(dtype_enum, "unknown")
    @classmethod
    def to_numpy(cls, dtype_enum):
        return cls._np_map.get(dtype_enum)
    @classmethod
    def to_ctype(cls, dtype_enum):
        return cls._ctype_map.get(dtype_enum)

class FFDevice:
    CPU = 0
    GPU_CUDA = 1
    GPU_ROCM = 2
    # Add mappings
    _map = {CPU: "cpu", GPU_CUDA: "cuda", GPU_ROCM: "rocm"}
    @classmethod
    def to_string(cls, device_enum):
        return cls._map.get(device_enum, "unknown")

# --- 2. Load the Shared Library ---
_LIB_LOADED = False
ff_lib = None
try:
    script_dir = os.path.dirname(__file__) or '.' # Handle running directly
    lib_path = os.path.join(script_dir, "libfusionflow_backend.so")

    if not os.path.exists(lib_path):
        print(f"Warning: Library not found at {lib_path}. Trying system load path.")
        lib_path = "libfusionflow_backend.so"

    ff_lib = ctypes.CDLL(lib_path)
    print(f"Successfully loaded C backend: {lib_path}")
    _LIB_LOADED = True

except OSError as e:
    print(f"FATAL: Error loading shared library: {e}")
    print("Ensure 'libfusionflow_backend.so' is compiled and accessible.")
    # Exit or raise a more specific exception if the library is essential
    # raise ImportError(f"Could not load FusionFlow C backend: {e}")

# --- 3. Define C Structures and Pointers ---
if _LIB_LOADED:
    class FFTensorStruct(ctypes.Structure):
        pass # Forward declaration

    FFTensorStruct._fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_size_t)),
        ("strides", ctypes.POINTER(ctypes.c_size_t)),
        ("ndim", ctypes.c_size_t),
        ("dtype", ctypes.c_int),
        ("device", ctypes.c_int),
        ("size", ctypes.c_size_t),
        ("nbytes", ctypes.c_size_t),
        ("ref_count", ctypes.c_int),
        ("requires_grad", ctypes.c_bool),
        ("grad", ctypes.POINTER(FFTensorStruct)),
    ]
    FFTensor_p = ctypes.POINTER(FFTensorStruct)

# --- 4. Define C Function Signatures ---
if _LIB_LOADED:
    # Utility
    ff_lib.ff_dtype_size.argtypes = [ctypes.c_int]
    ff_lib.ff_dtype_size.restype = ctypes.c_size_t
    # Lifecycle
    ff_lib.ff_tensor_create.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
    ff_lib.ff_tensor_create.restype = FFTensor_p
    ff_lib.ff_tensor_retain.argtypes = [FFTensor_p]
    ff_lib.ff_tensor_retain.restype = None
    ff_lib.ff_tensor_release.argtypes = [FFTensor_p]
    ff_lib.ff_tensor_release.restype = None
    # Operations
    ff_lib.ff_tensor_add.argtypes = [FFTensor_p, FFTensor_p]
    ff_lib.ff_tensor_add.restype = FFTensor_p
    # Add more signatures as C functions are implemented...
    print("C function signatures defined.")
else:
    print("Skipping C function signature definition as library failed to load.")

print("-" * 30)

# --- 5. Python Tensor Wrapper Class ---
class Tensor:
    """
    Python wrapper for the FusionFlow C Tensor (FFTensor).
    Manages the lifecycle (reference counting) of the underlying C object.
    """
    def __init__(self, c_tensor_ptr):
        """
        Initialize with a pointer to a C FFTensor struct.
        Assumes the pointer is valid and ownership is transferred here
        (or handled by the caller via retain/release).
        Typically called by factory methods, not directly by user.
        """
        if not isinstance(c_tensor_ptr, FFTensor_p):
            raise TypeError(f"Expected FFTensor_p (ctypes pointer), got {type(c_tensor_ptr)}")
        if not c_tensor_ptr:
             raise ValueError("Cannot initialize Tensor with a NULL pointer.")

        self._ptr = c_tensor_ptr
        # DO NOT RETAIN here. The creator (e.g., factory function calling C create)
        # gives us the initial reference (ref_count=1).

    def __del__(self):
        """Calls the C release function when the Python object is garbage collected."""
        # print(f"DEBUG: Releasing tensor @ {hex(id(self))}, C ptr: {self._ptr}")
        if _LIB_LOADED and ff_lib and hasattr(self, '_ptr') and self._ptr:
            ff_lib.ff_tensor_release(self._ptr)
            self._ptr = None # Avoid dangling pointer access if __del__ is somehow called again

    # --- Factory Methods ---
    @classmethod
    def zeros(cls, shape, dtype=FFDataType.FLOAT32, device=FFDevice.CPU, requires_grad=False):
        """Creates a new tensor filled with zeros."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        ndim = len(shape)
        shape_array = (ctypes.c_size_t * ndim)(*shape)
        c_ptr = ff_lib.ff_tensor_create(shape_array, ndim, dtype, device, requires_grad)
        if not c_ptr:
            raise MemoryError("C backend failed to create tensor.")
        return cls(c_ptr) # Wrap the owned pointer (ref_count=1 from create)

    # TODO: Add factories like ones(), empty(), from_numpy(), etc.
    # from_numpy would need to:
    # 1. Create an empty C tensor of the right shape/dtype.
    # 2. Get the C data pointer.
    # 3. Get the NumPy data pointer.
    # 4. Use ctypes.memmove to copy data from NumPy to C buffer.

    # --- Properties ---
    @property
    def _c_tensor(self):
        """Provides safe access to the dereferenced C structure contents."""
        if not self._ptr:
            raise RuntimeError("Accessing properties of a released or invalid Tensor.")
        return self._ptr.contents

    @property
    def shape(self):
        c_tensor = self._c_tensor
        if c_tensor.ndim == 0:
            return ()
        return tuple(c_tensor.shape[i] for i in range(c_tensor.ndim))

    @property
    def ndim(self):
        return self._c_tensor.ndim

    @property
    def dtype(self):
        # Return the enum value, maybe map to string later if needed
        return self._c_tensor.dtype

    @property
    def device(self):
        # Return the enum value
        return self._c_tensor.device

    @property
    def size(self):
        return self._c_tensor.size

    @property
    def requires_grad(self):
        return self._c_tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        if not isinstance(value, bool):
            raise TypeError("requires_grad must be boolean")
        # TODO: Add check: Can only set requires_grad on leaf tensors? (Like PyTorch)
        # Requires autograd graph tracking info. For now, allow setting.
        self._c_tensor.requires_grad = value
        # If setting requires_grad=True, potentially allocate grad tensor if needed?
        # if value and not self._c_tensor.grad:
        #    # Allocate gradient tensor (needs a C function)
        #    pass

    @property
    def grad(self):
        """
        Returns the gradient Tensor.
        Manages reference counting for the gradient.
        """
        c_tensor = self._c_tensor
        if not c_tensor.requires_grad:
            return None # Or raise error? PyTorch returns None.
        if not c_tensor.grad:
            return None # Gradient not computed or allocated yet.

        # IMPORTANT: We need to retain the C gradient tensor before wrapping it,
        # because the wrapper's __del__ will release it later.
        ff_lib.ff_tensor_retain(c_tensor.grad)
        return Tensor(c_tensor.grad) # Return a new Python wrapper for the grad tensor

    @grad.setter
    def grad(self, value):
        """Sets the gradient Tensor."""
        c_tensor = self._c_tensor
        if not c_tensor.requires_grad:
            raise RuntimeError("Cannot set grad on Tensor that does not require gradients.")

        # Release the old gradient if it exists
        if c_tensor.grad:
            ff_lib.ff_tensor_release(c_tensor.grad)
            c_tensor.grad = None

        if value is None:
            return # Just cleared the gradient

        if not isinstance(value, Tensor):
            raise TypeError("Gradient must be a Tensor instance or None.")

        # Check shape/dtype compatibility (optional but recommended)
        if value.shape != self.shape or value.dtype != self.dtype:
             print(f"Warning: Setting gradient with incompatible shape or dtype."
                   f" Expected {self.shape} / {FFDataType.to_string(self.dtype)},"
                   f" got {value.shape} / {FFDataType.to_string(value.dtype)}.")
             # raise TypeError("Gradient shape or dtype mismatch") # Stricter

        # Retain the new gradient C tensor before assigning its pointer
        ff_lib.ff_tensor_retain(value._ptr)
        c_tensor.grad = value._ptr


    # --- Operator Overloading ---
    def __add__(self, other):
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        if not isinstance(other, Tensor):
            # TODO: Handle scalar addition (create scalar tensor or specific C op)
            return NotImplemented

        result_ptr = ff_lib.ff_tensor_add(self._ptr, other._ptr)
        if not result_ptr:
            raise RuntimeError("C backend tensor addition failed.")
        # Wrap the new owned C tensor pointer
        return Tensor(result_ptr)

    # TODO: Implement other operators (__sub__, __mul__, __matmul__, etc.)
    # __matmul__ will require implementing ff_tensor_matmul in C first.

    # --- Data Access ---
    def numpy(self):
        """Copies the tensor data to a NumPy array."""
        if not _LIB_LOADED: raise RuntimeError("FusionFlow C backend not loaded.")
        c_tensor = self._c_tensor
        if c_tensor.device != FFDevice.CPU:
            raise NotImplementedError("Cannot copy non-CPU tensor data to NumPy yet.")

        np_dtype = FFDataType.to_numpy(c_tensor.dtype)
        if np_dtype is None:
            raise TypeError(f"Unsupported dtype {c_tensor.dtype} for NumPy conversion.")

        # Create an empty numpy array with the correct shape and type
        np_array = np.empty(self.shape, dtype=np_dtype)

        # Get C data pointer and number of bytes
        c_data_ptr = c_tensor.data
        nbytes = c_tensor.nbytes

        # Copy data from C buffer to NumPy buffer using ctypes
        # Assuming C tensor is contiguous for simplicity here. Needs stride handling for non-contiguous.
        # TODO: Add check for contiguous strides before using simple memmove
        ctypes.memmove(np_array.ctypes.data, c_data_ptr, nbytes)

        return np_array

    # --- Representation ---
    def __repr__(self):
        if not self._ptr:
            return "Tensor(<released>)"
        # Basic representation, could be enhanced to show some data
        shape_str = str(self.shape)
        dtype_str = FFDataType.to_string(self.dtype)
        device_str = FFDevice.to_string(self.device)
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({shape_str}, dtype={dtype_str}, device='{device_str}'{grad_str})"

    # --- Autograd (Placeholder) ---
    def backward(self):
        """Initiates backward pass to compute gradients."""
        # This logic will be complex.
        # 1. Check if tensor is scalar and requires_grad.
        # 2. Call a C function `ff_tensor_backward(self._ptr)` OR
        # 3. Implement the backward pass traversal logic here in Python,
        #    calling the respective _backward functions stored (perhaps via ctx?)
        #    or calling C gradient computation functions.
        # The current C backend has no backward logic yet.
        raise NotImplementedError("Backward pass not implemented yet.")


# --- Cleanup Hook ---
# Attempt to clean up any remaining tensors on exit, though __del__ is primary
_managed_tensors = [] # Keep track? Or rely solely on GC? Relying on GC is cleaner.

# def cleanup_all():
#     print("INFO: interpreter exit cleanup hook called.")
#     # Force garbage collection to try and trigger __del__
#     gc.collect()
# atexit.register(cleanup_all)

# --- Example Usage (Modified test_backend.py) ---
if __name__ == "__main__":
    if not _LIB_LOADED:
        print("Cannot run example usage as C library failed to load.")
        exit(1)

    print("--- Running Python Wrapper Tests ---")
    tensor_a = None
    tensor_b = None
    tensor_c = None

    try:
        # Test tensor creation using factory
        print("Creating tensor A: shape=(2, 3), dtype=float32")
        shape_a = (2, 3)
        tensor_a = Tensor.zeros(shape_a, dtype=FFDataType.FLOAT32, requires_grad=True)
        print(repr(tensor_a))
        # Access properties
        print(f"  Shape: {tensor_a.shape}, Ndim: {tensor_a.ndim}, DType: {tensor_a.dtype}")
        print(f"  Ref Count (C): {tensor_a._c_tensor.ref_count}") # Should be 1

        print("\nCreating tensor B: shape=(2, 3), dtype=float32")
        shape_b = (2, 3)
        tensor_b = Tensor.zeros(shape_b, dtype=FFDataType.FLOAT32, requires_grad=False)
        print(repr(tensor_b))
        print(f"  Ref Count (C): {tensor_b._c_tensor.ref_count}") # Should be 1

        # Modify data using numpy() - Create, modify, then maybe copy back?
        # OR add a fill method, or from_numpy factory
        print("\nModifying tensor B using numpy()...")
        np_b = tensor_b.numpy()
        np_b[0, 1] = 5.0
        print(f"  NumPy array for B (modified):\n{np_b}")
        # TODO: Need a way to copy data *back* into the C tensor if needed
        # e.g., tensor_b.copy_from_numpy(np_b)
        # For now, tensor_b's C data remains zeros. Let's test addition with that.

        # Test tensor addition using overloaded operator
        print("\nAdding tensor A and tensor B (A is zeros, B is zeros in C)...")
        tensor_c = tensor_a + tensor_b # Calls __add__
        print("Result tensor C:")
        print(repr(tensor_c))
        print(f"  Ref Count (C): {tensor_c._c_tensor.ref_count}") # Should be 1

        # Verify addition result using numpy()
        np_c = tensor_c.numpy()
        print(f"  Tensor C data via numpy():\n{np_c}")
        expected_c = np.zeros_like(np_c) # Since both A and B's C data were 0
        assert np.allclose(np_c, expected_c), "Addition result mismatch!"
        print("  Addition result verified.")

        # Test requires_grad property
        print(f"\nTensor A requires_grad: {tensor_a.requires_grad}")
        tensor_a.requires_grad = False
        print(f"Tensor A requires_grad (after setting False): {tensor_a.requires_grad}")
        tensor_a.requires_grad = True # Set back for potential grad testing later

        # Test grad property (will be None initially)
        print(f"Tensor A grad: {tensor_a.grad}")

        # Manually delete a tensor to test __del__ (optional)
        print("\nDeleting tensor_c explicitly...")
        c_ptr_addr = ctypes.addressof(tensor_c._ptr.contents) if tensor_c._ptr else 0
        del tensor_c
        tensor_c = None
        # Force garbage collection to make __del__ more likely to run immediately (for debug)
        gc.collect()
        print(f"Tensor C (originally @ {hex(c_ptr_addr)}) should have been released by C backend.")


    except (MemoryError, ValueError, RuntimeError, OSError, AttributeError, NotImplementedError) as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- Cleanup ---
        # No explicit ff_tensor_release calls needed here!
        # Python's garbage collector will call __del__ on tensor_a, tensor_b
        # when they go out of scope (or during interpreter shutdown).
        print("\n--- Python Script Finished ---")
        # We rely on GC and __del__ for cleanup of remaining tensors (tensor_a, tensor_b).
        # Optional: Trigger GC manually at exit to observe __del__ calls if needed for debugging.
        # print("Triggering final GC...")
        # gc.collect()
        pass
