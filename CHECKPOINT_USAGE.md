# Checkpoint Functionality for DNA Transport GNN Training

This document explains how to use the checkpoint functionality in `main.py` for training interruption and resumption on supercomputing clusters.

## Overview

The checkpoint system allows you to:
- Interrupt training at any time (Ctrl+C, job timeout, etc.)
- Resume training from exactly where you left off
- Monitor training progress through lightweight files
- Maintain all existing functionality

## Command Line Arguments

### New Checkpoint Arguments

- `--checkpoint_dir`: Directory to save checkpoints (default: `./checkpoints`)
- `--resume_from`: Path to specific checkpoint file to resume from
- `--checkpoint_frequency`: Save checkpoint every N epochs (default: 10)

### Example Usage

```bash
# Start fresh training with checkpoints
python main.py --num_epochs 100 --checkpoint_dir ./my_checkpoints

# Resume from latest checkpoint
python main.py --num_epochs 100 --checkpoint_dir ./my_checkpoints

# Resume from specific checkpoint
python main.py --num_epochs 100 --resume_from ./my_checkpoints/checkpoint_latest.pth

# Custom checkpoint frequency
python main.py --num_epochs 100 --checkpoint_frequency 5 --checkpoint_dir ./my_checkpoints
```

## File Structure

When using checkpoints, the following files are created:

```
checkpoint_dir/
├── checkpoint_latest.pth     # Latest checkpoint (always overwritten)
├── training_progress.json    # Detailed progress tracking
└── training_status.txt       # Simple status for quick monitoring
```

### File Contents

**checkpoint_latest.pth**: Complete training state
- Model state dictionary
- Optimizer state dictionary
- Current epoch number
- Training and validation loss history
- Training arguments
- Energy grid data
- Timestamp

**training_progress.json**: Detailed progress tracking
```json
{
  "epoch": 45,
  "train_loss": 0.1234,
  "val_loss": 0.1456,
  "timestamp": 1703123456.789,
  "args": {...}
}
```

**training_status.txt**: Simple status file
```
Epoch: 45
Train Loss: 0.1234
Val Loss: 0.1456
Last Update: 2023-12-21 14:30:45
```

## How It Works

### Starting Fresh Training
1. Creates checkpoint directory if it doesn't exist
2. Initializes model and training state
3. Saves checkpoint every N epochs (default: 10)
4. Updates progress files every epoch

### Resuming Training
1. Automatically detects existing checkpoint in `checkpoint_dir`
2. Loads model state, optimizer state, and loss history
3. Resumes from the next epoch
4. Continues saving checkpoints and progress files

### Interruption Handling
- **Graceful interruption**: Training can be stopped with Ctrl+C
- **Job timeout**: Checkpoint is saved periodically, so you can resume
- **System crash**: Latest checkpoint is preserved for resumption

## Best Practices

### For Supercomputing Clusters
1. **Set appropriate checkpoint frequency**: 
   - Use `--checkpoint_frequency 5` for short jobs
   - Use `--checkpoint_frequency 10` for longer jobs

2. **Monitor progress**:
   - Check `training_status.txt` for quick status
   - Use `training_progress.json` for detailed analysis

3. **Resume efficiently**:
   - Simply run the same command to resume from latest checkpoint
   - Use `--resume_from` for specific checkpoints

### Example Cluster Job Script
```bash
#!/bin/bash
#SBATCH --job-name=dna_training
#SBATCH --time=24:00:00
#SBATCH --mem=32G

# Start/resume training
python main.py \
    --num_epochs 1000 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_frequency 10 \
    --output_dir ./outputs
```

## Testing

Run the test script to verify checkpoint functionality:

```bash
python test_checkpoint.py
```

This will:
1. Start a short training session
2. Interrupt it after creating a checkpoint
3. Resume from the checkpoint
4. Verify all files are created correctly

## Backward Compatibility

The checkpoint functionality is completely backward compatible:
- If no checkpoint arguments are provided, training works exactly as before
- Existing scripts will continue to work without modification
- Checkpoint features are opt-in only

## Troubleshooting

### Common Issues

1. **"Checkpoint not found"**: 
   - Ensure checkpoint directory exists
   - Check file permissions

2. **"Failed to load checkpoint"**:
   - Checkpoint file may be corrupted
   - Try resuming from a different checkpoint
   - Start fresh training if needed

3. **Model compatibility issues**:
   - Ensure same model type and parameters
   - Check that arguments match between runs

### Debug Information

The script provides detailed output about checkpoint operations:
- Checkpoint save/load attempts
- Resume information
- Error messages for troubleshooting 