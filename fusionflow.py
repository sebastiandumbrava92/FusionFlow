# filename: fusionflow_prototype.py
# Description: A conceptual prototype for a deep learning library 'FusionFlow',
#              blending ideas from PyTorch, JAX, and TensorFlow/Keras.
#              NOTE: This is a *highly simplified simulation* for API design
#                    demonstration only. It uses NumPy for backend and has a
#                    basic autograd engine. Not performant or feature-complete.
# Version: Includes fix for Tensor transpose (.T attribute).

import numpy as np
import time

print(f"FusionFlow Prototype Initializing...")
# Adding current date/time as requested by user preferences
print(f"Current Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 30)

# --- 1. Core Tensor Class ---
# Wraps NumPy array, tracks computation graph for autograd

class Tensor:
    """A wrapper around a NumPy array that supports automatic differentiation."""
    def __init__(self, data, _children=(), _op='', requires_grad=False, dtype=None):
        if isinstance(data, (int, float)):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Input data must be a NumPy array, int, or float, got {type(data)}")

        self.data = data.astype(dtype if dtype is not None else data.dtype)
        self.requires_grad = requires_grad
        self.grad = None
        if self.requires_grad:
            self.zero_grad() # Initialize gradient array

        # --- Autograd internals ---
        # _children: tuple of Tensors that produced this Tensor
        # _op: string name of the operation that produced this Tensor
        # _backward: function to compute gradients w.r.t. children
        self._backward = lambda: None
        self._prev = set(_children) # Dependencies in the graph
        self._ctx = None # Optional context saved by the forward op for backward

    def zero_grad(self):
        """Resets the gradient of this Tensor to zero."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float64) # Use float64 for grads

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, shape={self.shape})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, _children=(self, other), _op='+', requires_grad=requires_grad)

        def _backward():
            # Gradient flows back scaled by 1. Need to handle broadcasting.
            if self.requires_grad:
                grad_self = out.grad
                # Handle broadcasting (summing gradients over broadcasted dimensions)
                if self.ndim < out.ndim:
                    sum_axes = tuple(range(out.ndim - self.ndim))
                    grad_self = grad_self.sum(axis=sum_axes)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)
                self.grad += grad_self

            if other.requires_grad:
                grad_other = out.grad
                # Handle broadcasting
                if other.ndim < out.ndim:
                    sum_axes = tuple(range(out.ndim - other.ndim))
                    grad_other = grad_other.sum(axis=sum_axes)
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, _children=(self, other), _op='*', requires_grad=requires_grad)

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            if self.requires_grad:
                grad_self = other.data * out.grad
                # Handle broadcasting
                if self.ndim < out.ndim:
                    sum_axes = tuple(range(out.ndim - self.ndim))
                    grad_self = grad_self.sum(axis=sum_axes)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)
                self.grad += grad_self

            if other.requires_grad:
                grad_other = self.data * out.grad
                 # Handle broadcasting
                if other.ndim < out.ndim:
                    sum_axes = tuple(range(out.ndim - other.ndim))
                    grad_other = grad_other.sum(axis=sum_axes)
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@', requires_grad=requires_grad)

        # --- !!! CORRECTED MATMUL BACKWARD !!! ---
        def _backward():
            # Using numpy's matmul rules for gradient calculation shapes
            # dL/dA = dL/dOut @ B.T
            if self.requires_grad:
                # Calculate gradient contribution using numpy directly on data
                # Use other.T.data which relies on the .T property we added
                grad_a = out.grad @ other.data.T
                # Ensure gradient shape matches self.data shape (handle broadcasting)
                if grad_a.shape != self.shape:
                     # Simplified broadcasting handler: sum over extra leading dims if needed
                     if grad_a.ndim > self.ndim:
                         diff_ndim = grad_a.ndim - self.ndim
                         grad_a = grad_a.sum(axis=tuple(range(diff_ndim)))
                     # This doesn't cover all broadcasting cases, but is a start
                self.grad += grad_a

            # dL/dB = A.T @ dL/dOut
            if other.requires_grad:
                # Calculate gradient contribution using numpy directly on data
                # Use self.T.data which relies on the .T property we added
                grad_b = self.data.T @ out.grad
                # Ensure gradient shape matches other.data shape (handle broadcasting)
                if grad_b.shape != other.shape:
                     if grad_b.ndim > other.ndim:
                         diff_ndim = grad_b.ndim - other.ndim
                         grad_b = grad_b.sum(axis=tuple(range(diff_ndim)))
                other.grad += grad_b
        # --- !!! END CORRECTED MATMUL BACKWARD !!! ---

        out._backward = _backward
        return out

    def relu(self):
        requires_grad = self.requires_grad
        out = Tensor(np.maximum(0, self.data), _children=(self,), _op='ReLU', requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                # Gradient is 1 where input > 0, else 0
                self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        # Simple sum to scalar
        requires_grad = self.requires_grad
        out = Tensor(np.sum(self.data), _children=(self,), _op='sum', requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                # Gradient is 1 for all elements, scaled by output grad
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    # Simplified log_softmax for NLLLoss
    def log_softmax(self, axis=-1):
        # shift max for numerical stability
        max_val = self.data.max(axis=axis, keepdims=True)
        shifted_exp = np.exp(self.data - max_val)
        sum_exp = shifted_exp.sum(axis=axis, keepdims=True)
        log_probs = (self.data - max_val) - np.log(sum_exp)

        out = Tensor(log_probs, _children=(self,), _op='LogSoftmax', requires_grad=self.requires_grad)
        out._ctx = {'probs': np.exp(log_probs)} # Save probs for backward

        def _backward():
            if self.requires_grad:
                probs = out._ctx['probs']
                # Jacobian of log_softmax is complex, simplified view for NLLLoss:
                # If dL/d(log_softmax)_i is 'g_i', then dL/d(input)_j = sum_i(g_i * (delta_ij - prob_j))
                # For NLLLoss, gradient dL/d(log_softmax)_i is typically -1 for the target class, 0 otherwise.
                # Let grad be the incoming grad (dL/d(log_softmax))
                dL_dInput = out.grad - probs * out.grad.sum(axis=axis, keepdims=True)
                self.grad += dL_dInput

        out._backward = _backward
        return out

    # --- !!! ADDED TRANSPOSE FUNCTIONALITY !!! ---
    @property
    def T(self):
        """Returns a new Tensor with the axes transposed, like numpy.ndarray.T"""
        requires_grad = self.requires_grad
        # Note: np.transpose(self.data) or self.data.T handles multi-dimensional arrays correctly
        out = Tensor(self.data.T, _children=(self,), _op='T', requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                # Gradient of transpose is transpose of gradient
                self.grad += out.grad.T

        out._backward = _backward
        return out

    # Also useful to add a specific transpose method for more control
    def transpose(self, *axes):
        """
        Returns a view of the array with axes transposed.
        Matches numpy.transpose behavior.
        """
        requires_grad = self.requires_grad
        new_data = np.transpose(self.data, axes if axes else None)
        out = Tensor(new_data, _children=(self,), _op='transpose', requires_grad=requires_grad)

        # Store original axes order to reverse in backward pass
        if not axes:
            axes = tuple(range(self.ndim)[::-1]) # Default reverses all axes
        inv_axes = np.argsort(axes) # axes to return to original shape
        out._ctx = {'inv_axes': inv_axes}

        def _backward():
            if self.requires_grad:
                inv_axes = out._ctx['inv_axes']
                # Transpose the incoming gradient back to original axes order
                self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        return out
    # --- !!! END ADDED TRANSPOSE FUNCTIONALITY !!! ---

    def backward(self):
        """Performs backpropagation starting from this Tensor (usually a scalar loss)."""
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on Tensor that does not require grad.")
        if self.data.size != 1:
            # In PyTorch, backward() can only be called on scalar outputs or
            # with a gradient argument. We'll simplify and require scalar loss.
            print("Warning: Calling backward() on non-scalar Tensor. Implicitly assuming gradient is 1.0.")
            # raise ValueError("backward can only be called on scalar Tensors (e.g., loss)")

        # Build topological order (simple reverse order traversal for this prototype)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                # Ensure children requiring gradients are processed
                for child in v._prev:
                    if child.requires_grad:
                        build_topo(child)
                topo.append(v)

        build_topo(self)

        # Perform backpropagation
        self.grad = np.ones_like(self.data) # Gradient of the loss w.r.t. itself is 1
        for node in reversed(topo):
            node._backward() # Calls the specific backward function stored by the op

    # NumPy compatibility properties
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    # Need __radd__, __rmul__, etc. for completeness (e.g., 5 * tensor)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    # ... add more operators as needed: sub, div, pow, neg, etc.

# --- 2. Parameters ---
class Parameter(Tensor):
    """A Tensor that is considered a model parameter and requires gradients by default."""
    def __init__(self, data, dtype=np.float32): # Default to float32 for params
        super().__init__(data, requires_grad=True, dtype=dtype)

# --- 3. Module System (PyTorch-like) ---
class Module:
    """Base class for all neural network modules."""
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        """Registers Parameters and Modules automatically."""
        if isinstance(value, Parameter):
            if '_parameters' not in self.__dict__: # Handle case during init
                 raise AttributeError("Cannot assign parameters before Module.__init__() call")
            # Detach parameter from any previous graph connection during assignment?
            # Simple prototype: assume parameters are leaves when assigned.
            value._prev = set()
            value._op = 'Parameter'
            self._parameters[name] = value
        elif isinstance(value, Module):
             if '_modules' not in self.__dict__:
                 raise AttributeError("Cannot assign submodules before Module.__init__() call")
             self._modules[name] = value
        super().__setattr__(name, value) # Default behavior

    def __getattr__(self, name):
        """Retrieve parameters/modules if not found directly."""
        if '_parameters' in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        if '_modules' in self.__dict__ and name in self._modules:
            return self._modules[name]
        # Fallback to default getattr to avoid infinite recursion on missing attributes
        # during initialization or other internal access.
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object or its modules have no attribute '{name}'")


    def parameters(self):
        """Returns an iterator over module parameters, yielding both the name and the parameter itself."""
        for name, param in self._parameters.items():
            yield name, param
        for name, module in self._modules.items():
            # Recursively yield parameters from submodules
            for sub_name, param in module.parameters():
                yield f"{name}.{sub_name}", param

    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for _, param in self.parameters():
            param.zero_grad()

    def __call__(self, *args, **kwargs):
        """Alias for forward pass."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call."""
        raise NotImplementedError

# --- 4. Basic Layers ---
class Linear(Module):
    """Applies a linear transformation: y = xA^T + b"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Kaiming uniform initialization (simplified)
        limit = np.sqrt(6.0 / in_features)
        # Use Parameter class for weights and biases
        self.weight = Parameter(np.random.uniform(-limit, limit, (out_features, in_features)))
        if bias:
            # Bias init (simplified zero)
            self.bias = Parameter(np.zeros(out_features))
        else:
            # Still register bias as None for consistent attribute access
            self.register_parameter('bias', None) # PyTorch style registration needed? No, handled by setattr

    def forward(self, input_tensor):
        # Use self.weight (which is a Parameter, subclass of Tensor) directly
        # The matmul operation will connect it to the graph
        output = input_tensor @ self.weight.T # Use the .T property we added to Tensor

        if self.bias is not None:
            # Use self.bias (Parameter) directly
            output = output + self.bias
        return output

class ReLU(Module):
    """Applies the Rectified Linear Unit function element-wise."""
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.relu()

# --- 5. Loss Function ---
class NLLLoss(Module):
    """Negative Log Likelihood Loss."""
    def __init__(self):
        super().__init__()

    def forward(self, log_probs_input, target):
        """
        Args:
            log_probs_input (Tensor): Input tensor of shape (N, C) containing log-probabilities.
            target (Tensor): Ground truth tensor of shape (N,) containing class indices (0 to C-1).
                               Assumed to be integer type, not requiring gradients.
        """
        N = log_probs_input.shape[0]
        C = log_probs_input.shape[1]

        # --- More robust NLLLoss backward ---
        # Create a one-hot encoding of the target (using numpy for indexing)
        # We need this to calculate the gradient dL/d(log_probs) = -1/N for target class, 0 otherwise
        target_one_hot = np.zeros((N, C), dtype=np.float64) # Use float64 for grads
        target_data = target.data.astype(int) # Ensure target data is integer for indexing
        target_one_hot[np.arange(N), target_data] = 1.0

        # Gather the log-probabilities of the target classes using the one-hot mask
        # Compute loss using numpy first
        log_probs_target = log_probs_input.data * target_one_hot
        loss_val = -np.sum(log_probs_target) / N

        # Wrap loss value in Tensor, connect to graph
        out = Tensor(loss_val, _children=(log_probs_input,), _op='NLLLoss', requires_grad=log_probs_input.requires_grad)
        # Store context needed for backward: the one-hot target mask
        out._ctx = {'target_one_hot': target_one_hot, 'N': N}

        def _backward():
            if log_probs_input.requires_grad:
                ctx = out._ctx
                target_one_hot = ctx['target_one_hot']
                N = ctx['N']
                # Gradient of NLLLoss w.r.t log_probs[i,j] is -1/N if j is the target class for sample i, else 0
                # This is exactly -(target_one_hot / N)
                grad_log_probs = -target_one_hot / N
                # Multiply by the gradient flowing into this node (out.grad, usually 1 for the final loss)
                log_probs_input.grad += grad_log_probs * out.grad

        out._backward = _backward
        return out


# --- 6. Optimizer ---
class SGD:
    """Implements stochastic gradient descent."""
    def __init__(self, params, lr=0.01):
        # Ensure params is an iterable of Parameter objects
        self.params = list(params) # Get list of Parameter tensors
        self.lr = lr

    def step(self):
        """Performs a single optimization step."""
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                # Basic SGD update rule
                p.data -= self.lr * p.grad
            elif p.requires_grad and p.grad is None:
                print(f"Warning: Parameter requires grad but grad is None during optimizer step.")


    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            # Only zero grad if it requires grad, otherwise grad is always None
            if p.requires_grad:
                p.zero_grad()


# --- 7. JIT and Grad Simulation (Interfaces) ---

def jit(func):
    """
    Simulated JIT decorator. In a real library, this would trigger
    tracing and compilation (e.g., via XLA). Here, it just runs the function.
    """
    print(f"[jit] Compiling {func.__name__} (simulation)...")
    def wrapper(*args, **kwargs):
        # print(f"[jit] Running compiled {func.__name__} (simulation)...")
        return func(*args, **kwargs)
    return wrapper

# grad function is harder to simulate cleanly alongside the Module backward() path
# without a more robust autograd system. We'll omit its use in the example.
# def grad(func, argnums=0):
#    """ Simulates JAX-like grad transformation """
#    def grad_func(*args, **kwargs):
#       # 1. Run func to build graph / get output
#       # 2. Call backward() on output
#       # 3. Extract gradient for specified argnums
#       raise NotImplementedError("grad simulation needs more complex autograd")
#    return grad_func


# --- 8. Example Usage: Simple MLP ---

# Define Model using Module system
class SimpleMLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.relu1 = ReLU()
        self.fc2 = Linear(hidden_size, output_size)

    # Uncomment the line below to simulate applying JIT
    # @jit
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Hyperparameters
input_size = 10
hidden_size = 5
output_size = 2
lr = 0.1
epochs = 25 # Increased epochs slightly

# Create model, loss, optimizer
model = SimpleMLP(input_size, hidden_size, output_size)
criterion = NLLLoss()
# Pass model parameters to optimizer - Use the generator directly
optimizer = SGD( (p for _, p in model.parameters()), lr=lr )

print("\nModel Architecture:")
# Basic print representation for Module (can be improved)
print(model)
print("\nOptimizer:")
print(optimizer)
print(f"Learning Rate: {optimizer.lr}")
print("-" * 30)

print("Starting Training Simulation...")

# Dummy Data (Batch size = 4)
# Ensure consistent types (e.g., float32 for inputs)
X_train = Tensor(np.random.rand(4, input_size).astype(np.float32))
# Target classes (0 or 1) - Ensure target is integer type
Y_train = Tensor(np.random.randint(0, output_size, size=(4,)).astype(np.int32))


# Training Loop
for epoch in range(epochs):
    # --- Forward Pass ---
    # Apply model
    logits = model(X_train)

    # Compute log-probabilities (input to NLLLoss)
    log_probs = logits.log_softmax(axis=1)

    # Compute loss
    loss = criterion(log_probs, Y_train)

    # --- Backward Pass ---
    # 1. Zero gradients before backward pass
    optimizer.zero_grad() # Or model.zero_grad() - same effect if optimizer holds all params

    # 2. Compute gradients
    loss.backward()

    # --- Optimizer Step ---
    # 3. Update weights
    optimizer.step()

    if (epoch + 1) % 5 == 0:
         # Check if loss.data is available (it should be a numpy scalar)
         loss_val = loss.data.item() if hasattr(loss.data, 'item') else loss.data
         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_val:.4f}")

print("-" * 30)
print("Training Finished.")

# --- Verification (optional) ---
print("\nSample Gradients after training (fc1 weight mean absolute):")
# Access parameter directly
fc1_weight_param = model.fc1.weight
if fc1_weight_param.grad is not None:
     # Calculate mean absolute gradient to see magnitude
     print(f"Mean Abs Grad: {np.mean(np.abs(fc1_weight_param.grad)):.6f}, Std Dev Grad: {fc1_weight_param.grad.std():.6f}")
else:
     print("Gradients not computed or cleared.")

print("\nSample Weights after training (fc1 weight mean):")
print(f"Mean Weight: {fc1_weight_param.data.mean():.4f}")
print("-" * 30)
