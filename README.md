# Navigation Project: Deep Q-Learning Agent

## Project Overview

This project implements a Deep Q-Learning agent to navigate and collect bananas in a Unity environment. The agent learns
to collect yellow bananas (+1 reward) while avoiding blue ones (-1 reward) in a large, square world.

### Environment Details

- **State Space**: 37 dimensions (including velocity and ray-based perception)
- **Action Space**: 4 discrete actions
    - 0: move forward
    - 1: move backward
    - 2: turn left
    - 3: turn right
- **Goal**: Achieve an average score of +13 over 100 consecutive episodes

## Dependencies and Installation

### Required Packages

```bash
pip install -r requirements.txt
```

Key dependencies include:

- Python 3.10
- PyTorch
- Pillow==9.4.0
- matplotlib==3.7.2
- jupyter==1.0.0
- pandas==2.0.3
- scipy==1.11.2
- protobuf>=3.20.3
- grpcio>=1.63.0

### Unity Environment Setup

Due to compatibility issues with Apple M2 machines, there are two recommended approaches:

1. **Udacity Workspace (Recommended)**:

    - Use the provided Udacity workspace which has the Unity environment pre-configured
    - Download the trained model weights and results locally

2. **Local Setup (Alternative)**:
    - Download the appropriate Unity environment for your OS:
        - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
        - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
        - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
        - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    - Place the unzipped file in the project root directory

## Running the Project

1. **Setup Environment**:

    ```bash
    # Create and activate conda environment
    conda create --name drlnd python=3.10
    conda activate drlnd

    # Install dependencies
    pip install -r requirements.txt
    ```

2. **Training the Agent**:
    - Open `Navigation.ipynb` in Jupyter Notebook
    - Follow the notebook cells sequentially to:
        - Initialize the environment
        - Train the agent
        - Visualize the results

Note: If using Apple M2 hardware, it's recommended to:

1. Train the model in Udacity's workspace
2. Download the trained weights and results
3. Run visualization and analysis locally

The project includes detailed implementation in `Navigation.ipynb` and training results in the `Report.md` file.
