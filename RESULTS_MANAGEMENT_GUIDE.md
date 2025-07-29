# Enhanced Results Management System

This system provides automatic hyperparameter tracking and results merging for your SGLD experiments.

## Key Features

### 🔄 **Automatic Results Merging**
- When you run experiments with identical hyperparameters, new results are automatically merged with existing ones
- No more overwriting previous results or managing multiple files manually
- Perfect for incrementally building up statistics with more repetitions

### 📊 **Hyperparameter Tracking** 
- All hyperparameters are saved with results in a structured JSON format
- Unique hash generated for each hyperparameter combination
- Easy comparison between different experimental setups

### 🔍 **Results Exploration**
- Command-line tools to explore and compare saved results
- Detailed information about each experiment
- Find compatible results that can be merged

## How It Works

### 1. **Automatic Detection**
When you run an experiment, the system:
1. Generates a unique hash from your hyperparameters
2. Checks if a results file with matching hyperparameters already exists
3. If found, merges your new results with the existing ones
4. If not found, creates a new results file

### 2. **Hyperparameter Matching**
Two experiments are considered compatible for merging if they have identical:
- Beta values
- Learning rates (a0)
- Number of epochs
- Dataset type and classes
- All other training parameters

### 3. **Results Merging**
When results are merged:
- Raw experimental data is combined
- Statistics (mean, variance, std) are recalculated
- Plots show data from all merged experiments
- Original individual results are preserved

## Usage Examples

### Running Experiments
```python
# Your first experiment - 5 repetitions
python main.py  # Creates new results file

# Later, run more repetitions with identical settings
python main.py  # Automatically merges with previous results
```

### Exploring Results
```bash
# List all saved results
python explore_results.py list

# Show detailed information about a specific file
python explore_results.py show sgld_mnist_5clsv5cls_h1bd712c9.json

# Compare hyperparameters between two files
python explore_results.py compare file1.json file2.json

# Find files with compatible hyperparameters
python explore_results.py compatible target_file.json
```

## File Structure

### Enhanced JSON Format
```json
{
  "metadata": {
    "created_timestamp": "2025-07-29T16:03:59.112734",
    "hyperparameter_hash": "1bd712c99b3c",
    "total_experiments": 15,
    "version": "1.0"
  },
  "hyperparameters": {
    "beta_values": [100, 1000, 5000],
    "num_repetitions": 5,
    "dataset_type": "mnist",
    "mnist_classes": [[0,1,2,3,4], [5,6,7,8,9]],
    // ... all other parameters
  },
  "results": {
    "100": {
      "train_bce_mean": 0.1234,
      "raw_train_bce": [0.12, 0.13, 0.11, ...],
      // ... all statistics and raw data
    }
  }
}
```

### Filename Convention
Files are automatically named with descriptive information:
- `sgld_mnist_0v1_h5bcb2592.json` - MNIST digits 0 vs 1
- `sgld_mnist_5clsv5cls_h1bd712c9.json` - MNIST 5 classes vs 5 classes
- `sgld_synth_h7a3f8e12.json` - Synthetic dataset

## Benefits

### 🚀 **Efficiency**
- No need to manage multiple result files manually
- Automatic accumulation of experimental data
- Consistent naming and organization

### 🔒 **Safety** 
- Never lose previous results due to accidental overwriting
- All raw data is preserved in merged results
- Backward compatibility with legacy format

### 📈 **Analysis**
- Easy to build up large statistical samples over time
- Compare different hyperparameter configurations
- Track experimental progress and parameter sensitivity

### 🎯 **Reproducibility**
- Complete hyperparameter tracking ensures reproducibility
- Easy to identify which results came from which settings
- Hash-based identification prevents parameter drift

## Workflow Example

```bash
# Day 1: Initial experiment with 5 repetitions
python main.py
# → Creates: sgld_mnist_evenvodd_h5bcb2592.json (5 reps per beta)

# Day 2: Add 10 more repetitions with same settings  
python main.py
# → Updates: sgld_mnist_evenvodd_h5bcb2592.json (15 reps per beta)

# Day 3: Try different learning rate
# (modify a0 in main.py)
python main.py  
# → Creates: sgld_mnist_evenvodd_h7a3f8e12.json (new file, different hyperparams)

# Explore results
python explore_results.py list
python explore_results.py show sgld_mnist_evenvodd_h5bcb2592.json
```

## Technical Details

### Hyperparameter Hashing
- Uses SHA256 hash of sorted hyperparameters
- Ignores minor variations in dataset sizes
- Accounts for complex nested parameters (like grouped MNIST classes)

### Results Merging Algorithm
1. Load existing results
2. Combine raw experimental data arrays
3. Recalculate all statistics (mean, variance, standard deviation)
4. Preserve individual experiment tracking
5. Update metadata with new totals

### Backward Compatibility
- Legacy `.txt` format still generated for compatibility
- Existing analysis scripts continue to work
- Gradual migration to enhanced format

This system makes it easy to build up comprehensive experimental results over time while maintaining full traceability and preventing data loss.
