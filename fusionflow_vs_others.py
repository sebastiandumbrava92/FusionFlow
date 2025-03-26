# fusionflow_vs_others.py (previously fusionflow_vs_others_rigorous.py)
"""
Compares FusionFlow, PyTorch, and TensorFlow on training a simple
linear layer to learn the identity function using SGD and MSE loss.

Includes multiple trials and controlled initialization for a more
rigorous comparison. Imports FusionFlow from the top-level module.
"""

import numpy as np
import time
import gc
import traceback
import sys
import os
import importlib
import argparse
from collections import defaultdict
import statistics # For mean/stdev

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Rigorous comparison of FusionFlow, PyTorch, TensorFlow for Identity Task.")
parser.add_argument('--num_trials', type=int, default=5, help='Number of training trials per framework.')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.05, help='Learning rate for SGD.')
parser.add_argument('--dim', type=int, default=32, help='Dimension of the identity function input/output.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training data.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
parser.add_argument('--skip_ff', action='store_true', help='Skip FusionFlow test.')
parser.add_argument('--skip_torch', action='store_true', help='Skip PyTorch test.')
parser.add_argument('--skip_tf', action='store_true', help='Skip TensorFlow test.')
args = parser.parse_args()

# --- Configuration ---
NUM_TRIALS = args.num_trials
EPOCHS = args.epochs
LR = args.lr
DIM = args.dim
BATCH_SIZE = args.batch_size
SEED = args.seed

print("--- Configuration ---")
print(f"Number of Trials: {NUM_TRIALS}")
print(f"Epochs per Trial: {EPOCHS}")
print(f"Learning Rate:    {LR}")
print(f"Dimension:        {DIM}")
print(f"Batch Size:       {BATCH_SIZE}")
print(f"Random Seed:      {SEED}")
print("-" * 20)


# --- Framework Import & Setup ---
# Set seeds *before* any framework initialization
np.random.seed(SEED)

FF_LOADED = False
TORCH_LOADED = False
TF_LOADED = False

# 1. FusionFlow
# Import from the new top-level fusionflow module
if not args.skip_ff:
    print("--- Importing FusionFlow ---")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path: sys.path.insert(0, script_dir)
        # Import directly from the new top-level module
        from fusionflow import Tensor as FFTensor # Use aliases for clarity if preferred
        from fusionflow import Parameter as FFParameter
        from fusionflow import Linear as FFLinear
        from fusionflow import MSELoss as FFMSELoss
        from fusionflow import SGD as FFSGD
        # If import succeeds, assume core loading worked
        print("FusionFlow components imported successfully.")
        FF_LOADED = True
    except ImportError as e: FF_LOADED = False; print(f"INFO: Could not import FusionFlow: {e}")
    except Exception as e: FF_LOADED = False; print(f"ERROR importing FusionFlow: {e}"); traceback.print_exc()
else: print("--- Skipping FusionFlow ---")


# 2. PyTorch
if not args.skip_torch:
    print("\n--- Importing PyTorch ---")
    try:
        torch = importlib.import_module('torch')
        torch.nn = importlib.import_module('torch.nn')
        torch.optim = importlib.import_module('torch.optim')
        torch.manual_seed(SEED)
        # Force CPU for comparison
        torch_device = torch.device("cpu")
        print(f"PyTorch imported successfully. Using device: {torch_device}")
        TORCH_LOADED = True
    except ImportError: TORCH_LOADED = False; print("INFO: PyTorch not found.")
    except Exception as e: TORCH_LOADED = False; print(f"ERROR importing PyTorch: {e}"); traceback.print_exc()
else: print("--- Skipping PyTorch ---")


# 3. TensorFlow
if not args.skip_tf:
    print("\n--- Importing TensorFlow ---")
    try:
        tf = importlib.import_module('tensorflow')
        # Set seed *early*
        tf.random.set_seed(SEED)
        # Suppress TF logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        # Attempt to force CPU - might need restart if GPU was already initialized
        try:
             tf.config.set_visible_devices([], 'GPU')
             print("INFO: TensorFlow GPU visibility disabled (forcing CPU).")
        except RuntimeError as e:
             print(f"INFO: Could not set TF device visibility (might need restart?): {e}")
             print("INFO: Proceeding, TF might use GPU if available and already initialized.")
        physical_devices_cpu = tf.config.list_physical_devices('CPU')
        visible_devices = tf.config.get_visible_devices()
        print(f"TensorFlow imported successfully. Visible devices: {visible_devices}")
        TF_LOADED = True
    except ImportError: TF_LOADED = False; print("INFO: TensorFlow not found.")
    except Exception as e: TF_LOADED = False; print(f"ERROR importing TensorFlow: {e}"); traceback.print_exc()
else: print("--- Skipping TensorFlow ---")


# --- Data and Initial Weights Generation ---
# Generate ONCE and reuse for all trials and frameworks
print("\n--- Generating Shared Data and Initial Weights ---")
X_np = np.random.randn(BATCH_SIZE, DIM).astype(np.float32)
Y_np = X_np.copy()

# Generate initial weights compatible with FusionFlow/TF Dense kernel (in, out)
lim = np.sqrt(6.0 / DIM)
W_init_np = np.random.uniform(-lim, lim, size=(DIM, DIM)).astype(np.float32)
b_init_np = np.zeros((DIM,), dtype=np.float32)

print(f"Generated data X_np: {X_np.shape}, Y_np: {Y_np.shape}")
print(f"Generated weights W_init_np: {W_init_np.shape}, b_init_np: {b_init_np.shape}")


# --- Training Function Definitions ---

def train_fusionflow(x_data, y_data, w_init, b_init, dim, epochs, lr):
    """Trains FusionFlow model with given initial weights, returns dict of results or None on error."""
    if not FF_LOADED: return None
    model, criterion, optimizer = None, None, None
    X_ff, Y_ff, loss_tensor, outputs, final_outputs_ff = None, None, None, None, None
    try:
        X_ff = FFTensor.from_numpy(x_data) # Using alias FFTensor
        Y_ff = FFTensor.from_numpy(y_data)
        model = FFLinear(dim, dim) # Using alias FFLinear

        if not isinstance(model.weight, FFParameter) or not isinstance(model.bias, FFParameter):
             raise TypeError("Model 'weight' or 'bias' is not a FusionFlow Parameter.")
        model.weight.copy_from_numpy(w_init)
        if model.bias is not None: model.bias.copy_from_numpy(b_init)

        criterion = FFMSELoss() # Using alias FFMSELoss
        optimizer = FFSGD(model.parameters(), lr=lr) # Using alias FFSGD

        initial_loss = -1.0
        final_loss = -1.0
        print_every = max(1, epochs // 5)

        for epoch in range(epochs):
            outputs = model(X_ff)
            loss_tensor = criterion(outputs, Y_ff)
            loss_val = loss_tensor.item()
            if epoch == 0: initial_loss = loss_val
            optimizer.zero_grad(); loss_tensor.backward(); optimizer.step()
            if epoch == 0 or (epoch + 1) % print_every == 0 or epoch == epochs - 1: print(f"    FF Epoch [{epoch+1:>{len(str(epochs))}}/{epochs}], Loss: {loss_val:.6f}")
            if np.isnan(loss_val) or np.isinf(loss_val): raise RuntimeError("Loss divergence")
        final_loss = loss_val

        with np.errstate(invalid='ignore'):
            final_outputs_ff = model(X_ff)
            final_mae = np.mean(np.abs(y_data - final_outputs_ff.numpy()))

        print(f"    FF Final -> Loss: {final_loss:.6f}, MAE: {final_mae:.6f}")
        return {"loss": final_loss, "mae": final_mae}

    except Exception as e:
        print(f"  ERROR during FusionFlow trial: {type(e).__name__} - {e}")
        traceback.print_exc()
        return None
    finally:
        del X_ff, Y_ff, loss_tensor, outputs, final_outputs_ff, model, criterion, optimizer
        gc.collect()


def train_pytorch(x_data, y_data, w_init, b_init, dim, epochs, lr):
    """Trains PyTorch model with given initial weights, returns dict of results or None on error."""
    if not TORCH_LOADED: return None
    model, criterion, optimizer = None, None, None
    X_pt, Y_pt, loss, outputs, final_outputs_pt = None, None, None, None, None
    try:
        X_pt = torch.from_numpy(x_data).to(torch_device)
        Y_pt = torch.from_numpy(y_data).to(torch_device)
        model = torch.nn.Linear(dim, dim, bias=True).to(torch_device)

        with torch.no_grad():
            if model.weight.shape != w_init.T.shape:
                 raise ValueError(f"PyTorch weight shape mismatch: Expected {w_init.T.shape}, got {model.weight.shape}")
            model.weight.copy_(torch.from_numpy(w_init.T))
            if model.bias is not None:
                 if model.bias.shape != b_init.shape:
                     raise ValueError(f"PyTorch bias shape mismatch: Expected {b_init.shape}, got {model.bias.shape}")
                 model.bias.copy_(torch.from_numpy(b_init))

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

        initial_loss = -1.0
        final_loss = -1.0
        print_every = max(1, epochs // 5)

        model.train()
        for epoch in range(epochs):
            outputs = model(X_pt); loss = criterion(outputs, Y_pt); loss_val = loss.item()
            if epoch == 0: initial_loss = loss_val
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if epoch == 0 or (epoch + 1) % print_every == 0 or epoch == epochs - 1: print(f"    PT Epoch [{epoch+1:>{len(str(epochs))}}/{epochs}], Loss: {loss_val:.6f}")
            if np.isnan(loss_val) or np.isinf(loss_val): raise RuntimeError("Loss divergence")
        final_loss = loss_val

        model.eval()
        with torch.no_grad():
            final_outputs_pt = model(X_pt)
            final_mae = np.mean(np.abs(y_data - final_outputs_pt.cpu().numpy()))

        print(f"    PT Final -> Loss: {final_loss:.6f}, MAE: {final_mae:.6f}")
        return {"loss": final_loss, "mae": final_mae}

    except Exception as e:
        print(f"  ERROR during PyTorch trial: {type(e).__name__} - {e}")
        traceback.print_exc()
        return None
    finally:
        # Use simpler finally block from previous correction
        if TORCH_LOADED and 'torch' in sys.modules and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # Rely on GC for intermediate vars like model, criterion etc.
        del X_pt, Y_pt, loss, outputs, final_outputs_pt # Can still del tensors


def train_tensorflow(x_data, y_data, w_init, b_init, dim, epochs, lr):
    """Trains TensorFlow/Keras model with given initial weights, returns dict of results or None on error."""
    if not TF_LOADED: return None
    model, loss_fn, optimizer = None, None, None
    X_tf, Y_tf, loss, outputs, final_outputs_tf = None, None, None, None, None
    layer_weights_list = [] # Define outside try for finally
    # Define inputs/dense_layer outside try for finally
    inputs, dense_layer = None, None
    try:
        X_tf = tf.constant(x_data, dtype=tf.float32)
        Y_tf = tf.constant(y_data, dtype=tf.float32)

        inputs = tf.keras.Input(shape=(dim,), dtype=tf.float32)
        dense_layer = tf.keras.layers.Dense(dim, activation=None, use_bias=True,
                                           kernel_initializer='zeros',
                                           bias_initializer='zeros'
                                           )
        outputs = dense_layer(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        layer_weights_list = dense_layer.get_weights()
        if layer_weights_list[0].shape != w_init.shape:
            raise ValueError(f"TF weight shape mismatch: Expected {w_init.shape}, got {layer_weights_list[0].shape}")
        layer_weights_list[0] = w_init
        if dense_layer.use_bias:
            if len(layer_weights_list) < 2 or layer_weights_list[1].shape != b_init.shape:
                 raise ValueError(f"TF bias shape mismatch: Expected {b_init.shape}, got {layer_weights_list[1].shape if len(layer_weights_list)>1 else 'None'}")
            layer_weights_list[1] = b_init
        dense_layer.set_weights(layer_weights_list)

        loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0)

        initial_loss = -1.0
        final_loss = -1.0
        print_every = max(1, epochs // 5)

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss_val = loss_fn(y, predictions) # Renamed 'loss' to 'loss_val'
            gradients = tape.gradient(loss_val, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss_val

        for epoch in range(epochs):
            loss = train_step(X_tf, Y_tf) # 'loss' here is the TF tensor result
            loss_val = loss.numpy() # Get numpy value

            if epoch == 0: initial_loss = loss_val

            if epoch == 0 or (epoch + 1) % print_every == 0 or epoch == epochs - 1:
                print(f"    TF Epoch [{epoch+1:>{len(str(epochs))}}/{epochs}], Loss: {loss_val:.6f}")
            if np.isnan(loss_val) or np.isinf(loss_val): raise RuntimeError("Loss divergence")
        final_loss = loss_val

        final_outputs_tf = model(X_tf, training=False)
        final_mae = np.mean(np.abs(y_data - final_outputs_tf.numpy()))

        print(f"    TF Final -> Loss: {final_loss:.6f}, MAE: {final_mae:.6f}")
        return {"loss": final_loss, "mae": final_mae}

    except Exception as e:
        print(f"  ERROR during TensorFlow trial: {type(e).__name__} - {e}")
        traceback.print_exc()
        return None
    finally:
        # Explicitly delete TF/Keras objects in reverse order of creation
        del final_outputs_tf, loss, outputs, dense_layer, model, inputs, loss_fn, optimizer, X_tf, Y_tf, layer_weights_list
        if TF_LOADED and 'tf' in sys.modules:
             tf.keras.backend.clear_session()
        gc.collect()


# --- Main Comparison Execution ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print(f" Starting Rigorous Comparison: Identity Task")
    print(f" Config: DIM={DIM}, BS={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}, SEED={SEED}, TRIALS={NUM_TRIALS}")
    print("="*70)

    all_results = defaultdict(lambda: defaultdict(list))

    framework_functions = {}
    if FF_LOADED: framework_functions["FusionFlow"] = train_fusionflow
    if TORCH_LOADED: framework_functions["PyTorch"] = train_pytorch
    if TF_LOADED: framework_functions["TensorFlow"] = train_tensorflow

    if not framework_functions:
        print("No frameworks available to test. Exiting.")
        sys.exit(1)

    overall_start_time = time.time()

    print(f"\n--- Running {NUM_TRIALS} Trials ---")

    for i in range(NUM_TRIALS):
        print(f"\n--- Trial {i+1}/{NUM_TRIALS} ---")

        for name, func in framework_functions.items():
            print(f"\n  Running {name}...")
            start_time_trial = time.time()
            result = func(X_np, Y_np, W_init_np, b_init_np, DIM, EPOCHS, LR)
            end_time_trial = time.time()
            duration_trial = end_time_trial - start_time_trial

            if result is not None:
                all_results[name]["loss"].append(result["loss"])
                all_results[name]["mae"].append(result["mae"])
                all_results[name]["duration"].append(duration_trial)
                print(f"  {name} Trial Duration: {duration_trial:.4f}s")
            else:
                 all_results[name]["loss"].append(np.nan)
                 all_results[name]["mae"].append(np.nan)
                 all_results[name]["duration"].append(np.nan)
                 print(f"  {name} Trial FAILED.")
        gc.collect()


    # --- Calculate and Print Summary Statistics ---
    print("\n" + "="*70)
    print(f" Rigorous Comparison Summary ({NUM_TRIALS} Trials, {EPOCHS} Epochs, LR={LR})")
    print("="*70)
    print(f"{'Framework':<15} | {'Metric':<10} | {'Mean':<15} | {'Std Dev':<15} | {'# Valid Trials':<15}")
    print("-" * 70)

    framework_order = ["FusionFlow", "PyTorch", "TensorFlow"]

    for name in framework_order:
        if name not in framework_functions: continue

        print(f"{name:<15} | {'-'*53}")

        for metric in ["loss", "mae", "duration"]:
            data = all_results[name][metric]
            valid_data = [x for x in data if not np.isnan(x)]
            num_valid = len(valid_data)
            num_total = len(data)

            if num_valid > 0:
                mean_val = statistics.mean(valid_data)
                stdev_val = statistics.stdev(valid_data) if num_valid > 1 else 0.0
                mean_str = f"{mean_val:.6f}"
                stdev_str = f"{stdev_val:.6f}"
            else:
                mean_str = "ERROR"
                stdev_str = "ERROR"

            valid_trial_str = f"{num_valid}/{num_total}"

            print(f"{'':<15} | {metric:<10} | {mean_str:<15} | {stdev_str:<15} | {valid_trial_str:<15}")
    print("="*70)

    overall_end_time = time.time()
    print(f"Total Execution Time: {overall_end_time - overall_start_time:.2f} seconds")
    print("="*70)
