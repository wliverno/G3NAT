#!/usr/bin/env python3
"""
Test script to verify the refactored checkpoint functionality.
"""

import os
import subprocess
import time
import sys

def test_refactored_checkpoint():
    """Test the refactored checkpoint functionality."""
    
    print("=== Testing Refactored Checkpoint Functionality ===")
    
    # Test parameters - use smaller checkpoint frequency for faster testing
    test_args = [
        "--num_samples", "50",   # Very small dataset for quick testing
        "--seq_length", "4",     # Short sequences
        "--num_epochs", "15",    # Short training
        "--checkpoint_frequency", "2",  # Save every 2 epochs for faster testing
        "--checkpoint_dir", "./test_refactored_checkpoints",
        "--output_dir", "./test_refactored_outputs"
    ]
    
    # Clean up any existing test files
    if os.path.exists("./test_refactored_checkpoints"):
        import shutil
        shutil.rmtree("./test_refactored_checkpoints")
    if os.path.exists("./test_refactored_outputs"):
        import shutil
        shutil.rmtree("./test_refactored_outputs")
    
    print("1. Starting initial training with refactored code...")
    print(f"Command: python main.py {' '.join(test_args)}")
    
    # Start training process
    process = subprocess.Popen(
        ["python", "main.py"] + test_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Let it run longer to ensure it creates a checkpoint
    print("   Waiting for training to create checkpoint...")
    time.sleep(25)  # Increased wait time
    
    print("2. Interrupting training...")
    process.terminate()
    
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("Force killing process...")
        process.kill()
        process.wait()
    
    # Check if there were any errors during training
    stdout, stderr = process.communicate()
    if stderr.strip():
        print(f"Training stderr output: {stderr}")
    
    print("3. Checking for checkpoint files...")
    
    checkpoint_dir = "./test_refactored_checkpoints"
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        print(f"Checkpoint directory contents: {files}")
        
        if "checkpoint_latest.pth" in files:
            print("✓ Checkpoint file found!")
        else:
            print("✗ No checkpoint file found")
            print("   This might be because training was interrupted too early")
            return False
            
        if "training_progress.json" in files:
            print("✓ Progress file found!")
        else:
            print("✗ No progress file found")
            return False
            
        if "training_status.txt" in files:
            print("✓ Status file found!")
            with open(os.path.join(checkpoint_dir, "training_status.txt"), "r") as f:
                status = f.read()
                print(f"Status: {status.strip()}")
        else:
            print("✗ No status file found")
            return False
    else:
        print("✗ Checkpoint directory not found")
        print("   This suggests training failed to start or checkpoint directory creation failed")
        return False
    
    print("4. Resuming training from checkpoint...")
    resume_args = test_args + ["--resume_from", "./test_refactored_checkpoints/checkpoint_latest.pth"]
    print(f"Command: python main.py {' '.join(resume_args)}")
    
    # Resume training
    process = subprocess.Popen(
        ["python", "main.py"] + resume_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Let it complete
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        print("✓ Training resumed and completed successfully!")
        print("5. Checking final results...")
        
        if os.path.exists("./test_refactored_outputs"):
            output_files = os.listdir("./test_refactored_outputs")
            print(f"Output files: {output_files}")
            
            if "training_curves.png" in output_files:
                print("✓ Training curves generated!")
            if "sample_predictions.png" in output_files:
                print("✓ Sample predictions generated!")
            if any(f.endswith(".pth") for f in output_files):
                print("✓ Final model saved!")
                
        return True
    else:
        print("✗ Training failed to resume")
        print(f"stdout: {stdout}")
        print(f"stderr: {stderr}")
        return False

def test_backward_compatibility():
    """Test that the refactored code maintains backward compatibility."""
    
    print("\n=== Testing Backward Compatibility ===")
    
    # Test parameters without checkpoint arguments
    test_args = [
        "--num_samples", "20",   # Very small dataset
        "--seq_length", "4",     # Short sequences
        "--num_epochs", "5",     # Very short training
        "--output_dir", "./test_backward_compat_outputs"
    ]
    
    # Clean up any existing test files
    if os.path.exists("./test_backward_compat_outputs"):
        import shutil
        shutil.rmtree("./test_backward_compat_outputs")
    
    print("1. Testing training without checkpoint arguments...")
    print(f"Command: python main.py {' '.join(test_args)}")
    
    # Run training without checkpoint arguments
    process = subprocess.Popen(
        ["python", "main.py"] + test_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Let it complete
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        print("✓ Backward compatibility test passed!")
        return True
    else:
        print("✗ Backward compatibility test failed")
        print(f"stdout: {stdout}")
        print(f"stderr: {stderr}")
        return False

if __name__ == "__main__":
    # Test refactored checkpoint functionality
    success1 = test_refactored_checkpoint()
    
    # Test backward compatibility
    success2 = test_backward_compatibility()
    
    if success1 and success2:
        print("\n=== All Tests PASSED ===")
        print("Refactored checkpoint functionality is working correctly!")
        print("Backward compatibility is maintained!")
    else:
        print("\n=== Some Tests FAILED ===")
        if not success1:
            print("- Refactored checkpoint functionality has issues")
        if not success2:
            print("- Backward compatibility has issues")
        sys.exit(1) 