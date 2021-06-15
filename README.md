# AI Toy Games

Collection of very simple toy problems in the form of games, solved using traditional AI techniques (A*, backtrack, local search).

All games and algorithms are entirely implemented in Python (with numpy dependency) therefore speed isn't really the focus here as I did this merely for fun (aka educational purposes). Nonetheless, there are a few cool tricks for optimizing stuff (e.g. numpy `stride_trick` for checking sudoku blocks).

Most if not all games here can be formalized as CSP problems, so most solutions presented here explicitely refer to CSP as presented in the book `Artificial Intelligence: A Modern Approach`, off of which most implementations are inspired from.

Install with `pip install -e .` after cloning into repo.

## Snake (with Gym snake environment), DQN
___
A notable exception is that of `Snake`, which I (not really) ""solved"" using Deep Reinforcement Learning (simple DQN).

The agent is trained entirely from raw pixels starting from a random policy using a frame-stack of size 4 (as the game graphics is so poor you can't even tell where's the head.. :D ); you can find the hyperparameters and the settings I used for training in the script `toy_games/reinforcement_learning/train_rl_baseline.py`.

I also provide a Gym Environment implementation for Snake to train your agent on with configurable grid size, rendering options.. 

Snake pre-trained weights can be loaded using stable baseline 3 utils directly from `snake.zip` (PyTorch). 
<p align="center">
<img src="assets/snake.gif" width="500" height="500">
</p>

Although the agent is far from super-human performance, it still exhibits some cool strategies, maybe one day it will surpass its master..

# Run games
All games are playable and you can run them by executing the corresponding file/script as in `python3 toy_games/games/snake.py` for playing snake.

You can then test out how different configuration play out (e.g. size of grid world).
# Examples
### Sliding Block Puzzle, A* search
___
<p align="center">
<img src="assets/sbp.gif" width="400" height="400">
</p>

### N Queens, backtrack search
___
<p align="center">
<img src="assets/nqueens.gif" width="400" height="400">
</p>

### Sudoku, generating solvable sudokus through backtrack search
___
<p align="center">
    <img src="assets/sudoku.gif" width="400" height="400">
</p>
