import argparse

import jax
from jax import numpy as jnp

from env.robotaxi import RobotaxiEnv
from utils.renderer import PygameFrontend
from utils.autoreset import AutoResetWrapper


def build_args():
    parser = argparse.ArgumentParser(description="Robotaxi environment viewer")
    parser.add_argument("--map-id", type=int, default=1, help="Map layout id from env.tasks")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--path-length", type=int, default=150)
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--perception-radius", type=float, default=5.0)
    parser.add_argument("--num-ray-sensors", type=int, default=32)
    parser.add_argument("--fov", type=float, default=jnp.pi)
    parser.add_argument("--discretization-scale", type=int, default=1)
    parser.add_argument("--wheelbase", type=float, default=2.5)
    parser.add_argument("--max-accel", type=float, default=3.0)
    parser.add_argument("--max-brake", type=float, default=4.0)
    parser.add_argument("--max-speed", type=float, default=6.0)
    parser.add_argument("--max-steer-deg", type=float, default=35.0)
    parser.add_argument("--friction", type=float, default=0.15)
    parser.add_argument("--steer-gain", type=float, default=3.0)
    parser.add_argument("--turn-min-speed", type=float, default=2.0)
    parser.add_argument("--yaw-rate-gain", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-mode", action="store_true", help="Use agent_fn if provided")
    return parser.parse_args()


def main():
    args = build_args()
    key = jax.random.PRNGKey(args.seed)
    env_params, init_state = RobotaxiEnv.init_params(
        key=key,
        map_id=args.map_id,
        max_steps=args.max_steps,
        path_length=args.path_length,
        fps=args.fps,
        perception_radius=args.perception_radius,
        num_ray_sensors=args.num_ray_sensors,
        fov=args.fov,
        discretization_scale=args.discretization_scale,
        wheelbase=args.wheelbase,
        max_accel=args.max_accel,
        max_brake=args.max_brake,
        max_speed=args.max_speed,
        max_steer=jnp.deg2rad(args.max_steer_deg),
        friction=args.friction,
        steer_gain=args.steer_gain,
        turn_min_speed=args.turn_min_speed,
        yaw_rate_gain=args.yaw_rate_gain,
    )

    env = AutoResetWrapper(RobotaxiEnv, env_params, init_state)
    frontend = PygameFrontend(env, env_params, init_state, eval_mode=args.eval_mode)
    frontend.run()


if __name__ == "__main__":
    main()
