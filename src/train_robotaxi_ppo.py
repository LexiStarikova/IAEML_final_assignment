import argparse
from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax import serialization
from flax.training.train_state import TrainState

from env.robotaxi import RobotaxiEnv
from utils.autoreset import AutoResetWrapper


@dataclass
class PPOConfig:
    num_envs: int = 8
    horizon: int = 256
    updates: int = 1000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    minibatch_size: int = 1024
    update_epochs: int = 4
    seed: int = 0
    map_id: int = 1
    max_steps: int = 1000
    path_length: int = 200
    fps: int = 60
    perception_radius: float = 6.0
    num_ray_sensors: int = 32
    # Behavior cloning warmstart (strongly improves convergence on maze-like maps)
    pretrain_iters: int = 2000
    pretrain_batch: int = 256
    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 100
    run_name: str = "robotaxi_ppo"
    save_best: bool = True
    best_threshold: float = 1.0


def _cfg_to_json(cfg: PPOConfig) -> dict:
    return cfg.__dict__.copy()


def save_checkpoint(save_dir: str, run_name: str, step: int, params, cfg: PPOConfig) -> Path:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{run_name}_step{step:06d}.msgpack"
    meta_path = out_dir / f"{run_name}_step{step:06d}.json"
    ckpt_path.write_bytes(serialization.to_bytes(params))
    meta_path.write_text(json.dumps({"step": step, "config": _cfg_to_json(cfg)}, indent=2))
    return ckpt_path


def load_checkpoint(ckpt_path: str, model: nn.Module, obs_example: jnp.ndarray):
    """
    Returns params restored from msgpack using model.init structure.
    """
    init_vars = model.init(jax.random.PRNGKey(0), obs_example)
    params = serialization.from_bytes(init_vars, Path(ckpt_path).read_bytes())
    return params


class ActorCritic(nn.Module):
    action_dim: int
    hidden_sizes: Tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        for h in self.hidden_sizes:
            x = nn.tanh(nn.Dense(h)(x))
        mean = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        log_std = self.param(
            "log_std", lambda rng, shape: jnp.zeros(shape), (self.action_dim,)
        )
        return mean, log_std, value


def gaussian_log_prob(pre_actions, mean, log_std):
    var = jnp.exp(2.0 * log_std)
    log_scale = log_std
    return -0.5 * jnp.sum(
        ((pre_actions - mean) ** 2) / var + 2 * log_scale + jnp.log(2 * jnp.pi),
        axis=-1,
    )


def tanh_gaussian_sample(mean, log_std, key):
    eps = jax.random.normal(key, mean.shape)
    pre_actions = mean + jnp.exp(log_std) * eps
    tanh_actions = jnp.tanh(pre_actions)
    log_prob = gaussian_log_prob(pre_actions, mean, log_std)
    log_prob -= jnp.sum(jnp.log(1 - tanh_actions**2 + 1e-6), axis=-1)
    return tanh_actions, log_prob


def tanh_gaussian_log_prob(mean, log_std, tanh_actions):
    atanh = jnp.arctanh(jnp.clip(tanh_actions, -0.999, 0.999))
    log_prob = gaussian_log_prob(atanh, mean, log_std)
    log_prob -= jnp.sum(jnp.log(1 - tanh_actions**2 + 1e-6), axis=-1)
    return log_prob


def gaussian_entropy(log_std: jnp.ndarray) -> jnp.ndarray:
    """Entropy of a diagonal Gaussian (pre-tanh). Shape: (action_dim,) -> scalar."""
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))


def scale_actions(tanh_actions: jnp.ndarray, env_params) -> jnp.ndarray:
    accel = jnp.where(
        tanh_actions[..., 0] >= 0,
        tanh_actions[..., 0] * env_params.max_accel,
        tanh_actions[..., 0] * env_params.max_brake,
    )
    steer = tanh_actions[..., 1] * env_params.max_steer
    return jnp.stack([accel, steer], axis=-1)


def unscale_actions(env_actions: jnp.ndarray, env_params) -> jnp.ndarray:
    """Inverse of scale_actions: map env action to tanh-space in [-1, 1]."""
    accel = env_actions[..., 0]
    steer = env_actions[..., 1]
    accel_tanh = jnp.where(
        accel >= 0,
        accel / (env_params.max_accel + 1e-6),
        accel / (env_params.max_brake + 1e-6),
    )
    steer_tanh = steer / (env_params.max_steer + 1e-6)
    return jnp.clip(jnp.stack([accel_tanh, steer_tanh], axis=-1), -0.999, 0.999)


def teacher_action(env_state, env_params) -> jnp.ndarray:
    """
    Simple expert: follow DP path direction with a cross-track steering controller.
    Returns env-space action: [accel, steer].
    """
    # Path direction from closest waypoint segment
    dists = jnp.linalg.norm(env_state.path_array - env_state.agent_pos, axis=1)
    closest_idx = jnp.argmin(dists)
    path_vec = (
        env_state.path_array.at[closest_idx + 1].get()
        - env_state.path_array.at[closest_idx].get()
    )
    path_dir = path_vec / (jnp.linalg.norm(path_vec) + 1e-8)

    forward = env_state.agent_forward_dir / (jnp.linalg.norm(env_state.agent_forward_dir) + 1e-8)
    # 2D cross product scalar (z-component)
    cross = forward[0] * path_dir[1] - forward[1] * path_dir[0]
    steer = jnp.clip(2.5 * cross, -env_params.max_steer, env_params.max_steer)

    goal_dist = jnp.linalg.norm(env_state.goal_pos - env_state.agent_pos)
    # accelerate until close to goal, then ease off
    accel = jnp.where(goal_dist > 2.0, env_params.max_accel, 0.5 * env_params.max_accel)
    return jnp.array([accel, steer], dtype=jnp.float32)


def obs_to_vec(obs_batch):
    return jax.vmap(lambda o: o.as_vector())(obs_batch)


def compute_gae(rewards, values, dones, last_values, gamma, lam):
    T, B = rewards.shape
    advantages = jnp.zeros((T, B))
    gae = jnp.zeros((B,))
    next_values = jnp.concatenate([values[1:], last_values[None, :]], axis=0)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages = advantages.at[t].set(gae)

    returns = advantages + values
    return advantages, returns


def rollout(env, model, params, env_state, obs_vec, key, env_params, cfg: PPOConfig):
    keys = jax.random.split(key, cfg.horizon * cfg.num_envs + 1)
    work_key = keys[0]
    step_keys = keys[1:]
    step_keys = step_keys.reshape(cfg.horizon, cfg.num_envs, 2)

    obs_buf = []
    actions_tanh_buf = []
    actions_env_buf = []
    logp_buf = []
    rewards_buf = []
    dones_buf = []
    values_buf = []
    goal_dist_buf = []

    state = env_state
    obs_v = obs_vec

    for t in range(cfg.horizon):
        mean, log_std, value = model.apply(params, obs_v)
        values_buf.append(value)

        ks = step_keys[t, :, :]
        tanh_actions, logp = jax.vmap(tanh_gaussian_sample, in_axes=(0, None, 0))(
            mean, log_std, ks
        )
        env_actions = scale_actions(tanh_actions, env_params)

        obs_buf.append(obs_v)
        actions_tanh_buf.append(tanh_actions)
        actions_env_buf.append(env_actions)
        logp_buf.append(logp)

        obs_next, state, reward, done, info = jax.vmap(env.step)(ks, state, env_actions)
        rewards_buf.append(reward)
        dones_buf.append(done.astype(jnp.float32))
        goal_dist_buf.append(info.get("goal_distance", jnp.nan))

        obs_v = obs_to_vec(obs_next)

    # last value for bootstrap
    last_mean, last_log_std, last_value = model.apply(params, obs_v)
    values_buf.append(last_value)

    traj = {
        "obs": jnp.stack(obs_buf),
        "actions_tanh": jnp.stack(actions_tanh_buf),
        "actions_env": jnp.stack(actions_env_buf),
        "logp": jnp.stack(logp_buf),
        "rewards": jnp.stack(rewards_buf),
        "dones": jnp.stack(dones_buf),
        "values": jnp.stack(values_buf[:-1]),
        "next_values": jnp.stack(values_buf[1:]),
        "goal_distance": jnp.stack(goal_dist_buf),
        "last_obs_vec": obs_v,
        "last_state": state,
    }
    return traj, last_value, state, obs_v, work_key


def flatten_batch(data: Dict[str, jnp.ndarray]):
    return {k: v.reshape((-1,) + v.shape[2:]) for k, v in data.items()}


def train(cfg: PPOConfig):
    key = jax.random.PRNGKey(cfg.seed)
    env_params, init_state = RobotaxiEnv.init_params(
        key=key,
        map_id=cfg.map_id,
        max_steps=cfg.max_steps,
        path_length=cfg.path_length,
        fps=cfg.fps,
        perception_radius=cfg.perception_radius,
        num_ray_sensors=cfg.num_ray_sensors,
    )
    env = AutoResetWrapper(RobotaxiEnv, env_params, init_state)

    reset_keys = jax.random.split(key, cfg.num_envs)
    obs0, state0 = jax.vmap(env.reset)(reset_keys)
    obs_vec = obs_to_vec(obs0)

    obs_dim = obs_vec.shape[-1]
    action_dim = 2

    model = ActorCritic(action_dim=action_dim)
    params = model.init(key, obs_vec)

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.learning_rate)
    )
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # ---------------------------
    # Behavior cloning warmstart
    # ---------------------------
    # Collect teacher rollouts with the current dynamics/reward. This produces a policy that can
    # reach the goal, then PPO can improve.
    bc_tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.learning_rate))
    bc_state = TrainState.create(apply_fn=model.apply, params=train_state.params, tx=bc_tx)

    def bc_loss(params, obs_vec_batch, target_tanh):
        mean, log_std, _ = model.apply(params, obs_vec_batch)
        pred = jnp.tanh(mean)
        loss = jnp.mean((pred - target_tanh) ** 2)
        return loss

    bc_step = jax.jit(lambda st, o, a: st.apply_gradients(grads=jax.grad(bc_loss)(st.params, o, a)))

    # rollout teacher in parallel envs
    bc_key = key
    bc_reset_keys = jax.random.split(bc_key, cfg.num_envs)
    bc_obs, bc_env_state = jax.vmap(env.reset)(bc_reset_keys)
    bc_obs_vec = obs_to_vec(bc_obs)

    for i in range(cfg.pretrain_iters):
        # teacher action -> tanh space target
        env_act = jax.vmap(teacher_action, in_axes=(0, None))(bc_env_state, env_params)
        target_tanh = unscale_actions(env_act, env_params)
        bc_state = bc_step(bc_state, bc_obs_vec, target_tanh)

        # advance env with teacher
        step_keys = jax.random.split(bc_key, cfg.num_envs)
        bc_obs, bc_env_state, _, _, _ = jax.vmap(env.step)(step_keys, bc_env_state, env_act)
        bc_obs_vec = obs_to_vec(bc_obs)
        bc_key = jax.random.split(bc_key, 2)[0]

        if (i + 1) % 500 == 0:
            print(f"pretrain={i+1:05d} bc_loss={float(bc_loss(bc_state.params, bc_obs_vec, target_tanh)):.6f}")

    # start PPO from warmstarted params
    train_state = train_state.replace(params=bc_state.params)
    save_checkpoint(cfg.save_dir, cfg.run_name, step=0, params=train_state.params, cfg=cfg)

    best_min_goal_dist = float("inf")

    def evaluate_mean_policy(params, steps: int = 1000):
        """Deterministic eval: action = tanh(mean), single env from init_state."""
        st = init_state
        obs, st = RobotaxiEnv.reset(key, env_params, st)
        obs_v = obs.as_vector()
        min_goal_dist = 1e9
        for _ in range(steps):
            mean, log_std, value = model.apply(params, obs_v)
            del log_std, value
            tanh_action = jnp.tanh(mean)
            env_action = scale_actions(tanh_action, env_params)
            obs, st, r, done, info = RobotaxiEnv.step(key, st, env_action, env_params)
            obs_v = obs.as_vector()
            gd = float(info.get("goal_distance", 1e9))
            if gd < min_goal_dist:
                min_goal_dist = gd
            if bool(done):
                break
        success = min_goal_dist < 1.0
        return success, min_goal_dist

    def loss_fn(params, batch, adv, returns):
        mean, log_std, value = model.apply(params, batch["obs"])
        new_logp = tanh_gaussian_log_prob(mean, log_std, batch["actions_tanh"])
        ratio = jnp.exp(new_logp - batch["logp"])
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        clipped = jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
        policy_loss = -jnp.mean(jnp.minimum(ratio * adv_norm, clipped * adv_norm))
        value_loss = jnp.mean((returns - value) ** 2)
        entropy = jnp.mean(jax.vmap(gaussian_entropy)(jnp.broadcast_to(log_std, mean.shape)))
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
        return loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }

    def update_step(train_state, batch, adv, returns):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            train_state.params, batch, adv, returns
        )
        new_state = train_state.apply_gradients(grads=grads)
        return new_state, metrics, loss

    # JIT the update step for speed/stability (still driven by a python loop).
    update_step_jit = jax.jit(update_step)

    for update in range(1, cfg.updates + 1):
        key, rollout_key = jax.random.split(key)
        traj, last_value, state0, obs_vec, key = rollout(
            env, model, train_state.params, state0, obs_vec, rollout_key, env_params, cfg
        )

        advantages, returns = compute_gae(
            traj["rewards"],
            traj["values"],
            traj["dones"],
            last_value,
            cfg.gamma,
            cfg.gae_lambda,
        )

        flat_data = flatten_batch(
            {
                "obs": traj["obs"],
                "actions_tanh": traj["actions_tanh"],
                "logp": traj["logp"],
            }
        )
        flat_adv = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        total_steps = flat_adv.shape[0]
        batch_size = cfg.minibatch_size

        for epoch in range(cfg.update_epochs):
            perm_key, key = jax.random.split(key)
            idx = jax.random.permutation(perm_key, total_steps)
            for start in range(0, total_steps, batch_size):
                mb_idx = idx[start : start + batch_size]
                mbatch = {k: v[mb_idx] for k, v in flat_data.items()}
                mb_adv = flat_adv[mb_idx]
                mb_ret = flat_returns[mb_idx]
                train_state, metrics, loss = update_step_jit(train_state, mbatch, mb_adv, mb_ret)

        # simple logging
        mean_reward = jnp.mean(traj["rewards"])
        min_goal_dist = jnp.min(traj["goal_distance"])
        done_rate = jnp.mean(traj["dones"])
        print(
            f"update={update:04d} reward={float(mean_reward):.3f} "
            f"min_goal_dist={float(min_goal_dist):.3f} done_rate={float(done_rate):.3f} "
            f"loss={float(loss):.4f} policy={float(metrics['policy_loss']):.4f} value={float(metrics['value_loss']):.4f}"
        )

        # Save best checkpoint when we get inside the goal region (or any chosen threshold).
        if cfg.save_best:
            cur_min_gd = float(min_goal_dist)
            if cur_min_gd < cfg.best_threshold and cur_min_gd < best_min_goal_dist:
                best_min_goal_dist = cur_min_gd
                save_checkpoint(
                    cfg.save_dir,
                    f"{cfg.run_name}_best",
                    step=update,
                    params=train_state.params,
                    cfg=cfg,
                )
                print(f"  saved best checkpoint at update={update:04d} min_goal_dist={best_min_goal_dist:.3f}")

        if update % 50 == 0:
            success, eval_min_gd = evaluate_mean_policy(train_state.params, steps=cfg.max_steps)
            print(f"  eval@{update:04d}: success={success} eval_min_goal_dist={eval_min_gd:.3f}")

        if cfg.save_every > 0 and update % cfg.save_every == 0:
            save_checkpoint(cfg.save_dir, cfg.run_name, step=update, params=train_state.params, cfg=cfg)

    return train_state


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Robotaxi")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-id", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--path-length", type=int, default=200)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--perception-radius", type=float, default=6.0)
    parser.add_argument("--num-ray-sensors", type=int, default=32)
    parser.add_argument("--pretrain-iters", type=int, default=2000)
    parser.add_argument("--pretrain-batch", type=int, default=256)  # reserved for future minibatching
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--run-name", type=str, default="robotaxi_ppo")
    parser.add_argument("--save-best", action="store_true", default=True)
    parser.add_argument("--no-save-best", action="store_false", dest="save_best")
    parser.add_argument("--best-threshold", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PPOConfig(
        num_envs=args.num_envs,
        horizon=args.horizon,
        updates=args.updates,
        learning_rate=args.learning_rate,
        seed=args.seed,
        map_id=args.map_id,
        max_steps=args.max_steps,
        path_length=args.path_length,
        fps=args.fps,
        perception_radius=args.perception_radius,
        num_ray_sensors=args.num_ray_sensors,
        pretrain_iters=args.pretrain_iters,
        pretrain_batch=args.pretrain_batch,
        save_dir=args.save_dir,
        save_every=args.save_every,
        run_name=args.run_name,
        save_best=args.save_best,
        best_threshold=args.best_threshold,
    )
    train(cfg)


if __name__ == "__main__":
    main()
