# CSCE-642-RL-Project

A project demonstrating the use of Q-Learning Reinforcement Learning (RL) to train an agent in a grid-based environment derived from real-world images.

---

## Setup Instructions

### Prerequisites

- Python 3.9.10 is required.
- Ensure you have all necessary dependencies installed.

### Installation

1. Clone the repository.
2. Navigate to the project root directory.
3. Run the following command to install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage Instructions

#### How to Run YOLO Inference

Run YOLO inference on an image using the following command:
```bash
   python yolo_inference.py --image path/to/image --model path/to/model
```
Example: 
```bash
   python yolo_inference.py --image ./dataset/RGBD/images/single_image/20240226_083200_png.rf.ba6bcc3c50c82775159757e0a0850e80_sn.jpg --model ./best.pt
```
---
#### How to Train the RL Agent
Train the RL agent using the following command:
```bash
   python train_agent.py
```

##### Important: Inside the `train_agent.py` file, modify the `input_path` variable to specify the folder containing the images you want to train on.
---
#### How to Train and Evaluate the RL Agent
1. Training: Use `train_agent.py` to train the RL agent on your chosen dataset.
2. Evaluation: Use `evaluate_agent.py` to assess the performance of the trained RL agent.
---
#### Excpected Output
- A plotted graph showing the training progress and results.
- An animated visualization (agent_animation.gif) of the agent navigating the grid environment.
- Text output summarizing the agent's success rate, average path length, and standard deviation.
---

#### Known Issues or Limitations
- For our images, we used real-world images of chickens in a chicken coop, which can sometimes limit the training ability of the Q-Learning RL agent.
- If there is no clear path to the egg (goal), the agent may receive a poor reward and fail to train effectively.




