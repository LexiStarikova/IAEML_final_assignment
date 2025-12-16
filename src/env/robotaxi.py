import jax
from jax import numpy as jnp
import equinox
import chex

from functools import partial
from typing import Tuple, Dict, Any

from env.base import (
    BaseEnv,
    BaseEnvParams,
    BaseEnvState,
    BaseEnvObservation,
    RADIUS,
    NUM_RAY_SENSORS,
    FOV,
)
from env.tasks import BaseTaskSampler
from utils import utils


# Simple dynamic bicycle-like model with first-order velocity.
DEFAULT_WHEELBASE = 2.0
DEFAULT_MAX_ACCEL = 3.0
DEFAULT_MAX_BRAKE = 4.0
DEFAULT_MAX_SPEED = 6.0
DEFAULT_MAX_STEER = jnp.deg2rad(75.0)
DEFAULT_FRICTION = 0.15
DEFAULT_STEER_GAIN = 3.0
DEFAULT_TURN_MIN_SPEED = 2.0
DEFAULT_YAW_RATE_GAIN = 3.0  # easier turning for keyboard control

# Reward weights (dense shaping)
# Key idea: reward SMALL progress too, especially in mazes where Euclidean goal distance can increase during detours.
DEFAULT_REWARD_GOAL = 30.0
DEFAULT_REWARD_COLLISION = 30.0
DEFAULT_REWARD_PROGRESS = 3.0
DEFAULT_REWARD_STEP = 0.001
DEFAULT_REWARD_SPEED_PENALTY = 0.01
DEFAULT_REWARD_ACTION_PENALTY = 0.001
DEFAULT_REWARD_DISTANCE_SHAPING = 0.005  # mild dense term: closer => higher reward
DEFAULT_REWARD_PATH_PROGRESS = 2.0  # reward reducing distance-to-path (potential diff)
DEFAULT_REWARD_PATH_DIST = 0.2  # penalty for being far from path
DEFAULT_REWARD_PATH_ALONG = 20.0  # reward for moving along the planned path direction
DEFAULT_COMFORT_SPEED = 4.0


class RobotaxiEnvParams(BaseEnvParams):
    wheelbase: float
    max_accel: float
    max_brake: float
    max_speed: float
    max_steer: float
    friction: float
    steer_gain: float
    turn_min_speed: float
    yaw_rate_gain: float
    reward_goal: float
    reward_collision: float
    reward_progress_scale: float
    reward_step_penalty: float
    reward_speed_penalty: float
    reward_action_penalty: float
    reward_distance_shaping: float
    reward_path_progress: float
    reward_path_dist: float
    reward_path_along: float
    comfort_speed: float


class RobotaxiEnvState(BaseEnvState):
    speed: chex.Scalar
    heading: chex.Scalar


class RobotaxiObservation(BaseEnvObservation):
    speed: chex.Scalar
    heading: chex.Scalar
    goal_direction: chex.Array
    goal_distance: chex.Scalar

    def as_vector(self) -> chex.Array:
        """Flatten the observation for policy consumption."""
        rays_flat = self.collision_rays.reshape(-1)
        return jnp.concatenate(
            [
                jnp.atleast_1d(self.distance_to_path),
                self.direction_of_path,
                self.goal_direction,
                jnp.atleast_1d(self.goal_distance),
                jnp.atleast_1d(self.speed),
                jnp.atleast_1d(self.heading),
                rays_flat,
            ],
            axis=0,
        )


class RobotaxiEnv(BaseEnv):
    @staticmethod
    def _wrap_angle(theta: chex.Array) -> chex.Array:
        return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    @staticmethod
    def init_params(
        key: chex.PRNGKey,
        map_id: int,
        max_steps: int,
        path_length: int,
        discretization_scale: int = 1,
        perception_radius: float = 5.0,
        num_ray_sensors: float = None,
        fov: float = None,
        fps: float = 60.0,
        wheelbase: float = DEFAULT_WHEELBASE,
        max_accel: float = DEFAULT_MAX_ACCEL,
        max_brake: float = DEFAULT_MAX_BRAKE,
        max_speed: float = DEFAULT_MAX_SPEED,
        max_steer: float = DEFAULT_MAX_STEER,
        steer_gain: float = DEFAULT_STEER_GAIN,
        turn_min_speed: float = DEFAULT_TURN_MIN_SPEED,
        yaw_rate_gain: float = DEFAULT_YAW_RATE_GAIN,
        friction: float = DEFAULT_FRICTION,
    ) -> Tuple[RobotaxiEnvParams, RobotaxiEnvState]:
        if discretization_scale != 1:
            raise Exception("discretization_scale != 1 is not implemented!")

        task = BaseTaskSampler(map_id)
        num_ray_sensors = num_ray_sensors or NUM_RAY_SENSORS
        fov = float(fov or FOV)
        wheelbase = float(wheelbase)
        max_accel = float(max_accel)
        max_brake = float(max_brake)
        max_speed = float(max_speed)
        max_steer = float(max_steer)
        steer_gain = float(steer_gain)
        turn_min_speed = float(turn_min_speed)
        yaw_rate_gain = float(yaw_rate_gain)
        friction = float(friction)

        env_params = RobotaxiEnvParams(
            max_steps_in_episode=max_steps,
            fps=fps,
            step_size=1 / fps,
            map_height_width=(task.height, task.width),
            nav_grid_shape_dtype=(
                (task.height * discretization_scale, task.width * discretization_scale),
                jnp.float32,
            ),
            path_array_shape_dtype=((path_length, 2), jnp.float32),
            discretization_scale=discretization_scale,
            perception_radius=perception_radius,
            agent_FOV=fov,
            num_ray_sensors=num_ray_sensors,
            wheelbase=wheelbase,
            max_accel=max_accel,
            max_brake=max_brake,
            max_speed=max_speed,
            max_steer=max_steer,
            friction=friction,
            steer_gain=steer_gain,
            turn_min_speed=turn_min_speed,
            yaw_rate_gain=yaw_rate_gain,
            reward_goal=DEFAULT_REWARD_GOAL,
            reward_collision=DEFAULT_REWARD_COLLISION,
            reward_progress_scale=DEFAULT_REWARD_PROGRESS,
            reward_step_penalty=DEFAULT_REWARD_STEP,
            reward_speed_penalty=DEFAULT_REWARD_SPEED_PENALTY,
            reward_action_penalty=DEFAULT_REWARD_ACTION_PENALTY,
            reward_distance_shaping=DEFAULT_REWARD_DISTANCE_SHAPING,
            reward_path_progress=DEFAULT_REWARD_PATH_PROGRESS,
            reward_path_dist=DEFAULT_REWARD_PATH_DIST,
            reward_path_along=DEFAULT_REWARD_PATH_ALONG,
            comfort_speed=DEFAULT_COMFORT_SPEED,
        )

        shape, dtype = env_params.path_array_shape_dtype

        convert_to_world_f = partial(
            utils.convert_to_world_view, map_shape=env_params.map_height_width
        )
        convert_to_world_vmap = jax.vmap(convert_to_world_f, in_axes=(0))

        heading = jnp.arctan2(task.agent_forward_dir[1], task.agent_forward_dir[0])

        init_state = RobotaxiEnvState(
            time=jnp.asarray(0),
            goal_pos=convert_to_world_f(task.goal_pos).astype(jnp.float32),
            agent_pos=convert_to_world_f(task.agent_pos).astype(jnp.float32),
            agent_forward_dir=task.agent_forward_dir.astype(jnp.float32),
            static_obstacles=convert_to_world_vmap(task.static_obstacles).astype(
                jnp.float32
            ),
            kinematic_obstacles=convert_to_world_vmap(task.kinematic_obstacles).astype(
                jnp.float32
            ),
            kinematic_obst_velocities=task.kinematic_obst_velocities.astype(
                jnp.float32
            ),
            path_array=jnp.zeros(shape=shape, dtype=dtype),
            speed=jnp.asarray(0.0, dtype=jnp.float32),
            heading=heading.astype(jnp.float32),
        )

        obstacles = jnp.concatenate(
            [init_state.static_obstacles, init_state.kinematic_obstacles], axis=0
        )
        path_array = BaseEnv._find_path(
            init_state.agent_pos, init_state.goal_pos, obstacles, env_params
        )
        init_state = equinox.tree_at(lambda t: t.path_array, init_state, path_array)

        return env_params, init_state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: chex.PRNGKey, env_params: RobotaxiEnvParams, init_state: RobotaxiEnvState
    ) -> Tuple[RobotaxiObservation, RobotaxiEnvState]:
        del key  # unused
        obs = RobotaxiEnv.get_observation(init_state, env_params)
        return obs, init_state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: RobotaxiEnvState,
        action: chex.Array,
        env_params: RobotaxiEnvParams,
    ) -> Tuple[
        RobotaxiObservation,
        RobotaxiEnvState,
        chex.Scalar | chex.Array,
        chex.Array,
        Dict[Any, Any],
    ]:
        del key  # deterministic transition

        # clamp and split actions
        accel = jnp.clip(action[0], -env_params.max_brake, env_params.max_accel)
        steer = jnp.clip(action[1], -env_params.max_steer, env_params.max_steer)

        dt = env_params.step_size
        friction_term = env_params.friction * env_state.speed
        speed = env_state.speed + dt * (accel - friction_term)
        # allow reverse
        speed = jnp.clip(speed, -env_params.max_speed, env_params.max_speed)

        eff_speed = jnp.maximum(jnp.abs(speed), env_params.turn_min_speed)
        # Easier turning for keyboard driving:
        # - always allow turning even at low speeds via turn_min_speed
        # - add a direct yaw term from steer so you can rotate without building forward velocity
        heading_rate = (
            env_params.steer_gain * eff_speed / env_params.wheelbase * jnp.tan(steer)
            + env_params.yaw_rate_gain * steer
        )
        heading = RobotaxiEnv._wrap_angle(env_state.heading + heading_rate * dt)
        forward_dir = jnp.array([jnp.cos(heading), jnp.sin(heading)], dtype=jnp.float32)

        new_agent_pos = env_state.agent_pos + forward_dir * speed * dt
        # World coordinates are (x, y) where x in [0, w) and y in [0, h)
        # env_params.map_height_width is (h, w), so we must flip for world-space clipping.
        map_limits = (
            jnp.array(
                [env_params.map_height_width[1], env_params.map_height_width[0]],
                dtype=jnp.float32,
            )
            - 1e-3
        )
        new_agent_pos = jnp.clip(new_agent_pos, jnp.zeros_like(map_limits), map_limits)

        # move obstacles
        kinematic_obstacles = BaseEnv._move_kinematic_obstacles(env_state, env_params)
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, kinematic_obstacles], axis=0
        )

        # termination checks
        goal_done = BaseEnv._check_goal(
            new_agent_pos, env_state.goal_pos, circle_radius=RADIUS
        )
        collision_done = BaseEnv._check_collisions(
            new_agent_pos, obstacles, circle_radius=RADIUS
        )
        time_done = env_state.time >= env_params.max_steps_in_episode
        done = jnp.logical_or(goal_done, jnp.logical_or(collision_done, time_done))

        # reward shaping
        prev_goal_dist = jnp.linalg.norm(env_state.agent_pos - env_state.goal_pos)
        new_goal_dist = jnp.linalg.norm(new_agent_pos - env_state.goal_pos)
        # Dense shaping:
        # - progress_reward rewards reducing distance (difference of potential)
        # - distance_reward gives an additional smooth gradient toward the goal
        progress_reward = (
            prev_goal_dist - new_goal_dist
        ) * env_params.reward_progress_scale
        distance_reward = -env_params.reward_distance_shaping * new_goal_dist

        # Path-based shaping (critical when the optimal route requires detours).
        # 1) Potential shaping via distance-to-path (prev - new)
        # 2) Dense directional reward for moving along the next path segment (works even for tiny steps)
        prev_path_dist = jnp.min(
            jnp.linalg.norm(env_state.path_array - env_state.agent_pos, axis=1)
        )
        new_path_dist = jnp.min(
            jnp.linalg.norm(env_state.path_array - new_agent_pos, axis=1)
        )
        path_progress_reward = (prev_path_dist - new_path_dist) * env_params.reward_path_progress
        path_dist_penalty = env_params.reward_path_dist * new_path_dist

        # Path-direction reward: dot(delta_pos, path_dir) > 0 is good even if goal distance doesn't change much.
        dists = jnp.linalg.norm(env_state.path_array - env_state.agent_pos, axis=1)
        closest_idx = jnp.argmin(dists)
        path_vec = (
            env_state.path_array.at[closest_idx + 1].get()
            - env_state.path_array.at[closest_idx].get()
        )
        path_dir = path_vec / (jnp.linalg.norm(path_vec) + 1e-8)
        delta_pos = new_agent_pos - env_state.agent_pos
        path_along_reward = env_params.reward_path_along * jnp.dot(delta_pos, path_dir)

        speed_penalty = (
            env_params.reward_speed_penalty
            * jnp.maximum(0.0, speed - env_params.comfort_speed) ** 2
        )
        action_penalty = env_params.reward_action_penalty * jnp.sum(action**2)
        reward = (
            progress_reward
            + distance_reward
            + path_progress_reward
            + path_along_reward
            - path_dist_penalty
            - env_params.reward_step_penalty
            - speed_penalty
            - action_penalty
        )
        reward = reward + env_params.reward_goal * goal_done
        reward = reward - env_params.reward_collision * collision_done

        # update path every second to save compute
        new_time = env_state.time + 1
        recompute_path = jnp.mod(new_time, env_params.fps)
        path_array = jax.lax.cond(
            recompute_path,
            lambda _: BaseEnv._find_path(
                new_agent_pos, env_state.goal_pos, obstacles, env_params
            ),
            lambda _: env_state.path_array,
            None,
        )

        new_state = RobotaxiEnvState(
            time=new_time,
            goal_pos=env_state.goal_pos,
            agent_pos=new_agent_pos,
            agent_forward_dir=forward_dir,
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=kinematic_obstacles,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=path_array,
            speed=speed,
            heading=heading,
        )

        obs = RobotaxiEnv.get_observation(new_state, env_params)
        info = {
            "time": new_time,
            "speed": speed,
            "heading": heading,
            "distance_to_path": obs.distance_to_path,
            "goal_distance": obs.goal_distance,
        }

        return obs, new_state, reward, done, info

    @staticmethod
    def get_observation(
        env_state: RobotaxiEnvState, env_params: RobotaxiEnvParams
    ) -> RobotaxiObservation:
        # find closest waypoint and path direction
        dists = jnp.linalg.norm(env_state.path_array - env_state.agent_pos, axis=1)
        closest_idx = jnp.argmin(dists)
        path_vec = (
            env_state.path_array.at[closest_idx + 1].get()
            - env_state.path_array.at[closest_idx].get()
        )
        path_dir = path_vec / (jnp.linalg.norm(path_vec) + 1e-8)

        anchor = env_state.path_array.at[closest_idx].get()
        rel_vec = env_state.agent_pos - anchor
        proj_len = jnp.dot(rel_vec, path_dir)
        proj_point = anchor + proj_len * path_dir
        distance_to_path = jnp.linalg.norm(env_state.agent_pos - proj_point)

        # rays perception
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )
        rays, ray_perceptions = BaseEnv._collision_ray_intersections(
            env_state.agent_pos, env_state.agent_forward_dir, obstacles, env_params
        )
        rays = rays * ray_perceptions[:, None]

        goal_vec = env_state.goal_pos - env_state.agent_pos
        goal_distance = jnp.linalg.norm(goal_vec)
        goal_direction = goal_vec / (goal_distance + 1e-8)

        return RobotaxiObservation(
            distance_to_path=distance_to_path,
            direction_of_path=path_dir,
            collision_rays=rays,
            speed=env_state.speed,
            heading=env_state.heading,
            goal_direction=goal_direction,
            goal_distance=goal_distance,
        )
