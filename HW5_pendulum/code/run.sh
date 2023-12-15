#!/bin/bash

# python main.py --env-name "cartpole" --seed=1 --no-baseline
# python main.py --env-name "cartpole" --seed=2 --no-baseline
# python main.py --env-name "cartpole" --seed=3 --no-baseline

# python main.py --env-name "cartpole" --seed=1 --baseline
# python main.py --env-name "cartpole" --seed=2 --baseline
# python main.py --env-name "cartpole" --seed=3 --baseline

# python main.py --env-name "cartpole" --seed=1 --ppo
# python main.py --env-name "cartpole" --seed=2 --ppo
# python main.py --env-name "cartpole" --seed=3 --ppo

# python main.py --env-name "pendulum" --seed=1 --no-baseline
# python main.py --env-name "pendulum" --seed=2 --no-baseline
# python main.py --env-name "pendulum" --seed=3 --no-baseline

# python main.py --env-name "pendulum" --seed=1 --baseline
# python main.py --env-name "pendulum" --seed=2 --baseline
# python main.py --env-name "pendulum" --seed=3 --baseline

# python main.py --env-name "pendulum" --seed=1 --ppo
# python main.py --env-name "pendulum" --seed=2 --ppo
# python main.py --env-name "pendulum" --seed=3 --ppo

python main.py --env-name "cheetah" --seed=1 --no-baseline
python main.py --env-name "cheetah" --seed=1 --baseline
python main.py --env-name "cheetah" --seed=1 --ppo