# DQN on Space Invaders
Implementation of a Deep Q Network using keras on Space Invaders using the OpenAI Gym. Any atari game can be substituted with this code (change to the action space may be needed).

## Setup
Create a virtual environment using virtualenv (make sure to use a python version >= 3.10) and run the following commands after activating your virtual env.

```sh
pip install -r requirements.txt
pip install -e .
```

### If using vscode and your editor shows is not able to find the project dependencies
Follow the instructions [here](https://code.visualstudio.com/docs/python/environments#_working-with-python-interpreters) to change the python interpreter used by the editor. Make sure you select the interpreter of the virtual environment you created when installing the project.

## Run
```sh
python run.py --render_mode=human
```
## Training command
python run.py --train=true --episodes=# --epsilon=1 --epsilon_decay=10000

## Optional commands
```
--obs_type         TYPE        Environment returns the RGB image that is displayed to human players as an observation.
--weights          FILE        Load another tested model's weights.
--save_on_cancel   T/F         Save the model weights when exiting training.
--buffer_capacity  INT         Memory size of stored environment variables. Larger buffer sizes can increase training time.
--epochs           INT         How many steps of gradient descent are wanted.
--batch_size       INT         Determines how many training samples are choosen from the buffer.
--learning_rate    FLOAT       Determines the size of the steps the neural network takes during the optimization process.
--discount_factor  FLOAT       Determines the importance of future rewards in the agent's decision making process.
```

## Before you push changes to the remote repo make sure 

1. You are not adding unnecessary dependencies to the project. 
2. `requirements.txt` file is up to date.
3. Everything works.
4. Verify you did not unintentionally over-write the model weights.

## Best Run
https://github.com/abun100/rl_space_invaders/assets/114605559/78d9a7d7-76fa-498f-9a26-414b0be58599


