# IAEML - Assignment

This repository was specifically designed to practice and implement the concepts covered in IAEML course at Innopolis University. [Vehicular automation](https://en.wikipedia.org/wiki/Vehicular_automation) is a notable challenging domain where these concepts can be applied.

Specifically, in this repository we have a simple 2D environment with an agent, destination and obstacles to avoid. The obstacles can be either static or moving. For simplicity, all objects are modeled as circles with radius $r=1$.

`tasks.py` comes in handy when we want to train something like meta-learning.

`base.py` defines an environment which models core elements required in self-driving cars:

- Perception is modeled as $M$ evenly fanned rays casted from the center of the agent.
- Navigation is solved using a dynamic programming algorithm fully executed on a device.

The environment has the following interface

- `init_params()` prepares `env_params` and `init_state`.
- `reset()` can be used to set `env_state` back to `env_state`.
- `step()` executes one step of the simulation.
- `get_observation()` returns the "point-of-view" of the agent.

Notably, `BaseEnv` is just a struct with functions that operates on `BaseEnvState` and takes `BaseEnvParams` as static arguments. Static arguments are "baked" into a function using `jax.jit()` decorator.

One limitation at the moment is that sampling a new task will always result in recompiling the environment functions since it changes the content of `env_params` and shapes in `env_state`. The latter can be solved by forcing all maps to be of the same shape.

`base.py` does not implement a model of vehicle dynamics.

You can play it

```bash
python src/main.py
```

There are some issues with JAX on OSX. This solves it

```bash
JAX_PLATFORM_NAME=cpu python src/main.py
```

...

---

Tested with python 3.11, Ubuntu 24.04 adm64 on Oct 20, 2025.

## Robotaxi environment (Stage 1)

- Implemented in `src/env/robotaxi.py` with a bicycle-like model driven by acceleration and steering actions, friction, and speed limits.
- Observation includes path following cues, goal direction, speed, heading, and collision rays.
- Reward encourages progress to goal, penalises collisions, large steering/accel, and comfort-speed violations.

Run the interactive renderer (with throttle/steer controls) using the new env:

```bash
python src/main.py --map-id 1 --fps 60 --num-ray-sensors 32
```

## PPO training pipeline (Stage 2)

- Minimal PPO reference implementation in `src/train_robotaxi_ppo.py` (Flax + Optax, vectorised envs via AutoResetWrapper).
- Default config: 4 envs, horizon 128, 10 updates for a quick sanity run.

Example quick run (CPU):

```bash
python src/train_robotaxi_ppo.py --num-envs 4 --horizon 128 --learning-rate 3e-4 --seed 0
```

Artifacts: the script prints update-level rewards/losses; extend the update loop for longer training if desired.
