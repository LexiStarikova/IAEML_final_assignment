import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization

from env.robotaxi import RobotaxiEnv
from utils.autoreset import AutoResetWrapper
from utils.renderer import PygameFrontend
from train_robotaxi_ppo import ActorCritic, scale_actions, obs_to_vec


def parse_args():
    p = argparse.ArgumentParser(description="Play/record a trained Robotaxi PPO policy")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .msgpack checkpoint")
    p.add_argument("--map-id", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--path-length", type=int, default=200)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--perception-radius", type=float, default=6.0)
    p.add_argument("--num-ray-sensors", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record", type=str, default=None, help="Output .mp4/.gif or a folder for PNG frames")
    p.add_argument("--record-fps", type=int, default=60)
    p.add_argument("--frames", type=int, default=800, help="Max frames to run/record")
    return p.parse_args()


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)

    env_params, init_state = RobotaxiEnv.init_params(
        key=key,
        map_id=args.map_id,
        max_steps=args.max_steps,
        path_length=args.path_length,
        fps=args.fps,
        perception_radius=args.perception_radius,
        num_ray_sensors=args.num_ray_sensors,
    )
    env = AutoResetWrapper(RobotaxiEnv, env_params, init_state)

    # Create model and restore params using correct tree structure.
    # Need an example observation vector to initialize shapes.
    obs0, st0 = env.reset(key)
    obs_vec0 = obs0.as_vector()[None, :]

    model = ActorCritic(action_dim=2)
    init_vars = model.init(key, obs_vec0)
    params = serialization.from_bytes(init_vars, Path(args.checkpoint).read_bytes())

    def agent_fn(state, obs):
        obs_vec = obs.as_vector()[None, :]
        mean, log_std, value = model.apply(params, obs_vec)
        del log_std, value
        tanh_action = jnp.tanh(mean[0])
        return scale_actions(tanh_action, env_params)

    frontend = PygameFrontend(
        env,
        env_params,
        init_state,
        eval_mode=True,
        agent_fn=agent_fn,
        record_path=args.record,
        record_fps=args.record_fps,
        max_frames=args.frames,
    )
    frontend.run()


if __name__ == "__main__":
    main()


