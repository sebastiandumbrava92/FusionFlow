# fusionflow_test.py
"""
Test suite for the FusionFlow library core components.
Imports necessary classes from the top-level fusionflow module and runs standard tests.
"""

import numpy as np
import gc
import traceback
import sys
import os
import time # For potential timing if needed

# --- Import Core Components ---
# Ensure the script can find fusionflow.py (and fusionflow_core.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

print("--- Importing FusionFlow Core ---")
FF_LOADED = False # Flag to check if import was successful
try:
    # Import directly from the new top-level module 'fusionflow'
    from fusionflow import (
        Tensor,
        Parameter,
        Module,
        Linear,
        MSELoss,
        SGD,
        FFDataType,
        FFDevice
    )
    # If this import succeeds, the underlying C library loading in fusionflow_core likely worked.
    print("Successfully imported components from fusionflow.")
    FF_LOADED = True
except ImportError as e:
    print(f"INFO: Could not import FusionFlow components: {e}")
    # FF_LOADED remains False
except Exception as e:
     print(f"ERROR during FusionFlow import: {e}")
     traceback.print_exc()
     # FF_LOADED remains False

# --- Test Function Definitions ---

def run_autograd_test():
    """Runs a simple autograd test (y = a*b + c)."""
    print("\n--- Running Autograd Test ---")
    a, b, c, ab, y = None, None, None, None, None # Initialize for finally block
    passed = False
    try:
        # Create tensors requiring gradients (Using imported classes)
        a = Tensor.from_numpy(np.array([2.], dtype=np.float32), requires_grad=True)
        b = Tensor.from_numpy(np.array([3.], dtype=np.float32), requires_grad=True)
        c = Tensor.from_numpy(np.array([1.], dtype=np.float32), requires_grad=True)

        ab = a * b
        y = ab + c
        print(f"y={repr(y)}")
        a.grad = None; b.grad = None; c.grad = None
        print("Calling y.backward()...");
        y.backward()
        print("Backward pass complete.")
        a_grad = a.grad.item() if a.grad else None
        b_grad = b.grad.item() if b.grad else None
        c_grad = c.grad.item() if c.grad else None
        print(f"dy/da (exp 3.0): {a_grad}")
        print(f"dy/db (exp 2.0): {b_grad}")
        print(f"dy/dc (exp 1.0): {c_grad}")
        assert a_grad is not None and abs(a_grad - 3.0) < 1e-6, "Gradient check for 'a' failed!"
        assert b_grad is not None and abs(b_grad - 2.0) < 1e-6, "Gradient check for 'b' failed!"
        assert c_grad is not None and abs(c_grad - 1.0) < 1e-6, "Gradient check for 'c' failed!"
        print("Simple autograd test PASSED.")
        passed = True
    except Exception as e:
        print(f"Autograd Test Error: {type(e).__name__} - {e}")
        traceback.print_exc()
        passed = False
    finally:
        print("\n--- Autograd Test Cleanup ---")
        del a, b, c, ab, y
        gc.collect()
        return passed


def run_training_test(epochs=20, lr=0.001, batch_size=4, dim=32):
    """
    Runs the identity mapping training test with specified hyperparameters.
    Returns True if the loss decreases by at least 20%, False otherwise.
    """
    test_label = f"(Epochs: {epochs}, LR: {lr}, BS: {batch_size}, Dim: {dim})"
    print(f"\n--- Running Identity Training Test {test_label} ---")
    X, Y, model, criterion, optimizer = None, None, None, None, None
    final_outputs, loss_tensor = None, None # Ensure defined for finally
    passed = False
    try:
        DIM, BATCH_SIZE, EPOCHS, LR = dim, batch_size, epochs, lr

        # --- Data ---
        np_x = np.random.randn(BATCH_SIZE, DIM).astype(np.float32)
        X = Tensor.from_numpy(np_x)
        Y = Tensor.from_numpy(np_x)

        # --- Model, Loss, Optimizer ---
        model = Linear(DIM, DIM) # Using imported classes
        criterion = MSELoss()
        optimizer = SGD(model.parameters(), lr=LR)

        print("-" * 30)
        print(f"  Starting training loop...")

        initial_loss=-1.0
        final_loss=-1.0
        print_every = max(1, epochs // 10)

        # --- Training Loop ---
        for epoch in range(EPOCHS):
            outputs = model(X)
            loss_tensor = criterion(outputs, Y)
            loss_val = loss_tensor.item()

            if epoch == 0: initial_loss = loss_val

            optimizer.zero_grad()
            loss_tensor.backward()
            optimizer.step()

            if epoch == 0 or (epoch + 1) % print_every == 0 or epoch == EPOCHS - 1:
                 print(f"    Epoch [{epoch+1:>{len(str(EPOCHS))}}/{EPOCHS}], Loss: {loss_val:.6f}")

            if epoch == EPOCHS - 1: final_loss = loss_val

            if np.isnan(loss_val) or np.isinf(loss_val):
                print("    Error: Loss became NaN or Inf. Stopping training.")
                raise RuntimeError("Loss divergence (NaN/Inf)")

        print("-" * 30)
        print("  Training loop finished.")

        # --- Evaluation ---
        if final_loss >= 0 and initial_loss >= 0:
            print(f"  Initial Loss: {initial_loss:.6f}")
            print(f"  Final Loss:   {final_loss:.6f}")
            loss_threshold = initial_loss * 0.8
            if final_loss < loss_threshold:
                 print(f"  PASSED: Loss decreased significantly (below {loss_threshold:.6f}).")
                 passed = True
            else:
                 print(f"  FAILED: Loss did not decrease significantly (threshold: {loss_threshold:.6f})!")
                 passed = False
        else:
             print("  Error: Could not compare initial and final loss values.")
             passed = False

        final_outputs = model(X)
        np_final_outputs = final_outputs.numpy()
        mean_abs_diff = np.mean(np.abs(np_x - np_final_outputs))
        print(f"  Mean Absolute Difference: {mean_abs_diff:.6f}")

    except Exception as e:
        print(f"  Training Test Error: {type(e).__name__} - {e}")
        traceback.print_exc()
        passed = False
    finally:
        print(f"--- Training Test {test_label} Finished ---")
        # Cleanup
        del X, Y, model, criterion, optimizer, final_outputs, loss_tensor
        gc.collect()
        return passed


# --- Main Execution Block ---
if __name__ == "__main__":
    overall_start_time = time.time()
    print("="*50)
    print(" Starting FusionFlow Test Suite")
    print("="*50)

    # Check if the FusionFlow components were imported successfully
    # This implicitly checks if the C backend loaded via fusionflow_core
    if not FF_LOADED:
        print("FATAL: FusionFlow components failed to import.")
        print("Ensure fusionflow.py and fusionflow_core.py are present and C library compiled/loadable.")
        sys.exit(1)
    else:
         print("FusionFlow components imported successfully.")

    results = {}
    # Run baseline autograd test
    results["autograd"] = run_autograd_test()
    print("="*50) # Separator

    # --- Define Training Configurations ---
    training_configs = [
        {"label": "lr0.001_e20",  "lr": 0.001, "epochs": 20},
        {"label": "lr0.01_e100",  "lr": 0.01,  "epochs": 100},
        {"label": "lr0.05_e100",  "lr": 0.05,  "epochs": 100},
        {"label": "lr0.05_e500",  "lr": 0.05,  "epochs": 500},
        {"label": "lr0.1_e100",   "lr": 0.1,   "epochs": 100},
        {"label": "lr0.01_e1000", "lr": 0.01,  "epochs": 1000},
    ]

    print("\n--- Running Multiple Training Configurations ---")
    for config in training_configs:
        start_time = time.time()
        test_name = f"training_{config['label']}"
        status = run_training_test(epochs=config["epochs"], lr=config["lr"])
        results[test_name] = status
        end_time = time.time()
        print(f"  Duration: {end_time - start_time:.2f} seconds")
        print("-" * 35)

    # --- Final Summary ---
    overall_end_time = time.time()
    print("\n" + "="*50)
    print(" FusionFlow Test Suite Summary")
    print("="*50)
    all_tests_passed = True
    for test_name, status in results.items():
        print(f"Test '{test_name}': {'PASSED' if status else 'FAILED'}")
        if not status:
            all_tests_passed = False

    print("-" * 50)
    total_duration = overall_end_time - overall_start_time
    print(f"Total Execution Time: {total_duration:.2f} seconds")
    if all_tests_passed:
        print("Overall Result: All tests PASSED")
        exit_code = 0
    else:
        print("Overall Result: Some tests FAILED")
        exit_code = 1
    print("="*50)

    sys.exit(exit_code)
