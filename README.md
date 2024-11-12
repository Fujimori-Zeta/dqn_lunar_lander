# Deep Q-Learning for Lunar Lander

This project implements Deep Q-Learning (DQN) to solve the Lunar Lander environment using the Gymnasium API.

## How to Run

1. Clone the repository:
      git clone https://github.com/Fujimori-Zeta/dqn_lunar_lander.git
   cd dqn_lunar_lander

2. Install the dependencies

      `pip install -r requirements.txt`
3. Run the training:
     `python dqn_lunar_lander.py`


4. Visualize
    `python -c "from dqn_lunar_lander import show_video_of_model; show_video_of_model(agent, 'LunarLander-v2')"`
