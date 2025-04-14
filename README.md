# Stylish Progress

A stylish progress bar library that provides a beautiful visual interface similar to tqdm.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stylish-progress.git

# Add the directory to your Python path
export PYTHONPATH=$PYTHONPATH:/path/to/stylish-progress
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

## License

MIT License 