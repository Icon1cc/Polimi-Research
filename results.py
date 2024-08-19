import os
import json
import matplotlib.pyplot as plt
import csv

# Directory where checkpoints are saved
checkpoint_dir = './results'

# List all checkpoint directories
checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint')]

# Sort checkpoints by global step
checkpoints.sort(key=lambda x: int(x.split('-')[-1]))

# Lists to store steps and losses
steps = []
losses = []

print(f"Found checkpoints: {checkpoints}")

# Iterate through each checkpoint and extract the loss
for checkpoint in checkpoints:
    state_file = os.path.join(checkpoint, 'trainer_state.json')
    print(f"Processing checkpoint: {checkpoint}")
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            trainer_state = json.load(f)
            # Extract the global step and loss
            step = trainer_state['global_step']
            print(f"Global step: {step}")
            loss = None
            for entry in trainer_state['log_history']:
                if 'loss' in entry:
                    loss = entry['loss']
                    print(f"Loss at step {step}: {loss}")
            if loss is not None:
                steps.append(step)
                losses.append(loss)

# Check if losses were collected
if not steps or not losses:
    print("No loss data found in checkpoints.")
else:
    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='o')
    plt.xlabel('Global Step')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.show()

    # Save the loss values to a CSV file
    csv_file = "training_losses.csv"
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Step", "Loss"])
        writer.writerows(zip(steps, losses))
    print(f"Loss data saved to {csv_file}")
