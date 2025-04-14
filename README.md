# Stylish Progress

A stylish progress bar library that provides a beautiful visual interface similar to tqdm.

## Installation

### Method 1: Direct Import (Recommended)
Simply clone the repository and import directly from the cloned directory:

```bash
# Clone the repository
git clone https://github.com/h4xryu/Custom-progress-bar.git

# Move to your project directory
cd your-project-directory

# Copy the stylish_progress directory to your project
cp -r ../stylish-progress/stylish_progress ./
```

Then in your Python code:
```python
from stylish_progress import Bar
```

### Method 2: Python Path
Add the directory to your Python path:

#### Linux/macOS
```bash
# Temporary (current terminal session only)
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export PYTHONPATH=$PYTHONPATH:$(pwd)' >> ~/.bashrc
source ~/.bashrc
```

#### Windows
```cmd
# Temporary (current terminal session only)
set PYTHONPATH=%PYTHONPATH%;%CD%

# Permanent (System Properties > Environment Variables)
# Add new System Variable:
# Variable name: PYTHONPATH
# Variable value: %CD%
```

### Method 3: Virtual Environment
Create a virtual environment and install the package:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Copy the package to site-packages
cp -r ../stylish-progress/stylish_progress venv/lib/python3.x/site-packages/
```

## Usage

### Basic Usage

```python
from stylish_progress import Bar

# For general sequences like lists or tuples
for item in Bar(range(100), desc="Processing"):
    # Do something
    pass

# For iterables with unknown length
iterator = some_iterator()
for item in Bar(iterator, total=1000, desc="Processing"):
    # Do something
    pass
```

### Using in Deep Learning Training

```python
from stylish_progress import Bar, Writer

# Training loop
writer = Writer('./logs/experiment1')
train_bar = Bar(train_loader, desc="Training")

for epoch in range(num_epochs):
    for batch in train_bar:
        # Training step
        loss = model.train_step(batch)
        # Display loss in progress bar
        train_bar.update_loss(loss.item())
```

## Features

- Support for general sequences and iterables
- Loss display feature for deep learning training
- Color output support
- ETA (Estimated Time of Arrival) display
- Compact mode support
