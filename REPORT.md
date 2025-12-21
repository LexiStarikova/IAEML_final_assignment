# IAEML Assignment Report

## Stage 1: Environment Design and Implementation

### 1. Dynamics Model Description and Justification

The implemented vehicle dynamics model is a **simplified bicycle-like model** that captures the essential kinematics and dynamics of a wheeled vehicle. This model extends the base environment (`BaseEnv`) with realistic vehicle motion.

#### State Space

The state space extends `BaseEnvState` with two additional continuous state variables:
- **Speed** (`speed`): Linear velocity in the forward direction, allowing both forward and reverse motion (range: `[-max_speed, max_speed]`)
- **Heading** (`heading`): Orientation angle in radians, wrapped to `[-π, π]`

#### Action Space

The action space is 2-dimensional:
- **Acceleration** (`action[0]`): Clamped to `[-max_brake, max_accel]` (default: `[-4.0, 3.0]`)
- **Steering angle** (`action[1]`): Clamped to `[-max_steer, max_steer]` (default: `±75°`)

#### Dynamics Equations

The dynamics are implemented as follows:

**Velocity update with friction:**
```
speed(t+1) = clip(speed(t) + dt * (accel - friction * speed(t)), -max_speed, max_speed)
```

Where `friction = 0.15` provides linear velocity damping, modeling air resistance and rolling friction.

**Heading rate computation:**
The heading rate combines two terms:
1. **Bicycle model term**: `steer_gain * eff_speed / wheelbase * tan(steer)`
   - This models the kinematic bicycle model where turning radius depends on wheelbase and steering angle
   - Uses `eff_speed = max(|speed|, turn_min_speed)` to ensure turning is possible even at low speeds

2. **Direct yaw rate term**: `yaw_rate_gain * steer`
   - Provides additional rotation capability, making the model more responsive for keyboard control
   - Allows rotation even without forward velocity

**Heading and position update:**
```
heading(t+1) = wrap_angle(heading(t) + heading_rate * dt)
forward_dir = [cos(heading), sin(heading)]
position(t+1) = position(t) + forward_dir * speed * dt
```

#### Design Choices and Justification

1. **Bicycle model over point-mass**: The bicycle model captures realistic steering behavior where turning radius depends on velocity, essential for navigation tasks.

2. **First-order velocity dynamics**: Using acceleration as input with friction provides smooth, realistic motion without the complexity of second-order dynamics (which would require modeling mass and inertia).

3. **Separate brake/accel limits**: Different limits for acceleration (3.0) and braking (4.0) reflect that vehicles can brake more aggressively than they can accelerate.

4. **Friction term**: Linear friction (`friction * speed`) provides velocity damping, preventing unrealistic perpetual motion and making control more stable.

5. **Turn minimum speed**: The `turn_min_speed` parameter (default: 2.0) ensures the agent can always turn, addressing the kinematic bicycle model limitation where turning at zero speed requires infinite steering.

6. **Yaw rate gain**: The additional direct yaw term makes the vehicle more maneuverable, which is beneficial for both human control and learning, as it reduces the need for complex speed-steering coordination.

**Default Parameters:**
- Wheelbase: 2.0 (typical for small vehicles)
- Max acceleration: 3.0 m/s²
- Max braking: 4.0 m/s²
- Max speed: 6.0 m/s
- Max steering: ±75° (allows sharp turns)
- Friction coefficient: 0.15
- Steer gain: 3.0
- Yaw rate gain: 3.0

---

### 2. Reward Function Description and Justification

The reward function uses **dense reward shaping** to guide the agent efficiently toward the goal while encouraging safe and smooth behavior. This is critical because sparse rewards (only on goal/collision) make learning extremely difficult in maze-like environments.

#### Reward Components

The total reward combines multiple terms:

```
reward = progress_reward
       + distance_reward
       + path_progress_reward
       + path_along_reward
       - path_dist_penalty
       - reward_step_penalty
       - speed_penalty
       - action_penalty
       + reward_goal * goal_done
       - reward_collision * collision_done
```

**1. Progress toward goal:**
```
progress_reward = (prev_goal_dist - new_goal_dist) * reward_progress_scale
```
- Rewards reducing Euclidean distance to goal (potential difference)
- Default weight: 3.0
- **Justification**: Provides gradient toward goal, but insufficient alone in mazes where detours increase Euclidean distance

**2. Distance shaping:**
```
distance_reward = -reward_distance_shaping * new_goal_dist
```
- Mild negative reward proportional to goal distance
- Default weight: 0.005
- **Justification**: Adds smooth gradient, but kept small to avoid conflict with path-based rewards

**3. Path-based shaping (critical for mazes):**

The environment provides a dynamically computed path from current position to goal using a pathfinding algorithm. Three path-related rewards guide the agent:

**a) Path distance progress:**
```
path_progress_reward = (prev_path_dist - new_path_dist) * reward_path_progress
```
- Rewards reducing distance to the planned path
- Default weight: 2.0
- **Justification**: Encourages staying on/near the optimal route even when Euclidean distance to goal increases

**b) Path distance penalty:**
```
path_dist_penalty = reward_path_dist * new_path_dist
```
- Penalizes being far from the path
- Default weight: 0.2
- **Justification**: Prevents excessive deviation from the planned route

**c) Path direction reward:**
```
path_along_reward = reward_path_along * dot(delta_pos, path_dir)
```
- Rewards moving along the direction of the next path segment
- Default weight: 20.0 (highest dense reward component)
- **Justification**: Provides dense signal even when path distance doesn't change much, critical for maze navigation where small steps along the path should be rewarded

**4. Penalties:**

**Speed penalty:**
```
speed_penalty = reward_speed_penalty * max(0, speed - comfort_speed)²
```
- Penalizes speeds above comfort speed (default: 4.0 m/s)
- Default weight: 0.01
- **Justification**: Encourages moderate speeds for safety and stability

**Action penalty:**
```
action_penalty = reward_action_penalty * sum(action²)
```
- L2 penalty on actions
- Default weight: 0.001
- **Justification**: Encourages smooth control and prevents excessive steering/acceleration

**Step penalty:**
```
reward_step_penalty = 0.001 (constant per step)
```
- Small negative reward per step
- **Justification**: Encourages efficient completion

**5. Terminal rewards:**

- **Goal reward**: +30.0 on reaching goal
- **Collision penalty**: -30.0 on collision

#### Design Rationale

The reward function balances multiple objectives:

1. **Dense guidance**: Multiple dense reward terms ensure the agent receives useful signal at every step, essential for learning in complex environments.

2. **Path-aware shaping**: Using the dynamically computed path is crucial because:
   - In mazes, the shortest path often requires moving away from the goal initially
   - Euclidean distance rewards can be misleading (reward decreasing while taking detours)
   - Path-based rewards provide correct guidance throughout navigation

3. **Relative importance**: The weights were tuned through experimentation:
   - Path-along reward (20.0) is highest, providing strong guidance
   - Goal/collision rewards (30.0) dominate terminal states
   - Progress rewards (3.0) provide moderate guidance
   - Penalties are small to avoid overwhelming the shaping rewards

4. **Balance**: The combination of positive shaping rewards and small penalties encourages goal-directed, smooth, and safe behavior.

---

### 3. Observation Space

The observation extends `BaseEnvObservation` with additional vehicle-specific information:

**Components:**
1. **Path following cues** (inherited from base):
   - `distance_to_path`: Perpendicular distance to the planned path
   - `direction_of_path`: Normalized direction vector of the next path segment

2. **Goal information**:
   - `goal_direction`: Normalized vector from agent to goal
   - `goal_distance`: Euclidean distance to goal

3. **Vehicle state**:
   - `speed`: Current linear velocity
   - `heading`: Current orientation angle

4. **Perception** (inherited from base):
   - `collision_rays`: Array of ray intersection distances (shape: `[num_ray_sensors, 2]`), with default 32 rays in 360° FOV

**Vector representation:**
The observation is flattened to a vector for neural network consumption:
```
[distance_to_path, direction_of_path (2D), goal_direction (2D),
 goal_distance, speed, heading, collision_rays (flattened)]
```
Total dimension: 1 + 2 + 2 + 1 + 1 + 1 + (32 * 2) = 72

This provides comprehensive information about the environment, vehicle state, and obstacles, enabling informed decision-making.

---

### 4. Difficulties Encountered and Resolution

#### Difficulty 1: Path-aware reward shaping in mazes

**Problem**: Initial reward function used only Euclidean distance to goal. In maze-like environments, the agent would receive negative rewards when taking necessary detours, making learning very difficult.

**Solution**: Implemented path-based reward shaping that uses the dynamically computed optimal path. The agent receives rewards for:
- Reducing distance to the path (staying on route)
- Moving along the path direction (making progress even on detours)

This provides correct guidance even when Euclidean distance to goal temporarily increases.

#### Difficulty 2: Kinematic bicycle model at low speeds

**Problem**: The standard bicycle model requires forward velocity to turn, but in tight spaces the agent needs to turn even at low speeds. At zero speed, the model cannot turn, causing the agent to get stuck.

**Solution**:
1. Introduced `turn_min_speed` parameter that ensures effective speed for turning is always ≥ 2.0 m/s
2. Added a direct yaw rate term (`yaw_rate_gain * steer`) that allows rotation even without forward velocity

This combination enables maneuverability at all speeds while maintaining realistic high-speed behavior.

#### Difficulty 3: Reward weight tuning

**Problem**: Initial reward weights led to suboptimal behavior: either the agent would ignore the path and go straight (too much goal-distance reward) or get stuck making tiny movements (too much path-following reward).

**Solution**: Systematic tuning through experimentation:
- Increased path-along reward (20.0) to provide strong directional guidance
- Balanced goal progress (3.0) to provide motivation without conflicting with path
- Kept penalties small (0.001-0.01) to avoid overwhelming shaping rewards

#### Difficulty 4: JAX compilation and static arguments

**Problem**: JAX's `jax.jit` requires static arguments to be specified. The environment parameters need to be marked as static, but some calculations depend on dynamic state.

**Solution**: Used `@partial(jax.jit, static_argnames=("env_params",))` to mark `env_params` as static, allowing JAX to compile the step function efficiently while keeping state dynamic. The environment parameters are "baked in" during compilation, but state updates remain fast.

---

### 5. Results and Reproduction

#### Results

The environment successfully models realistic vehicle dynamics with:
- Smooth acceleration and braking with friction
- Realistic steering behavior via bicycle model
- Dense reward shaping that guides learning in complex environments
- Comprehensive observation space for decision-making

A video demonstration is available at `videos/robotaxi.mp4` showing the agent navigating the environment.

#### Reproduction Instructions

**Prerequisites:**
```bash
pip install -r requirements.txt
```

**Test the environment interactively:**
```bash
python src/main.py --map-id 1 --fps 60 --num-ray-sensors 32
```

This opens an interactive renderer where you can control the vehicle using keyboard:
- Arrow keys/WASD for acceleration and steering
- The environment shows the agent, goal, obstacles, planned path, and collision rays

**Run a deterministic episode:**
```bash
python src/main.py --map-id 1
```

The environment is fully deterministic and JIT-compiled for efficiency. The dynamics, rewards, and observations are computed using JAX for fast execution.

---

## Stage 2: Reinforcement Learning Experiments

### 1. Experimental Setup and RL Model

#### Algorithm: Proximal Policy Optimization (PPO)

We implemented PPO, a state-of-the-art policy gradient algorithm that:
- Uses clipped surrogate objective to prevent large policy updates
- Employs Generalized Advantage Estimation (GAE) for stable advantage estimation
- Maintains separate actor (policy) and critic (value) networks with shared feature extraction

#### Network Architecture

**Actor-Critic Network** (Flax implementation):
- **Shared feature extractor**: Two hidden layers with 128 units each, tanh activations
- **Policy head**: Linear layer → `action_dim` outputs (mean) + learnable log_std parameter
- **Value head**: Linear layer → 1 output (state value)

**Action distribution**: Tanh-squashed Gaussian
- Policy outputs mean and log_std for a Gaussian distribution
- Actions are sampled from Gaussian, then passed through tanh to ensure they're in `[-1, 1]`
- Actions are scaled to environment ranges: acceleration uses asymmetric scaling (accel vs brake), steering uses symmetric scaling

#### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_envs` | 8 | Number of parallel environments |
| `horizon` | 256 | Rollout length (steps per update) |
| `updates` | 1000 | Number of PPO updates |
| `learning_rate` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda parameter |
| `clip_eps` | 0.2 | PPO clip epsilon |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `value_coef` | 0.5 | Value loss coefficient |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `minibatch_size` | 1024 | Minibatch size for updates |
| `update_epochs` | 4 | Number of update epochs per rollout |

#### Behavior Cloning Warmstart

Before PPO training, we perform **behavior cloning (BC)** to initialize the policy with a reasonable expert policy:

**Expert policy**: Simple path-following controller
- Computes steering using cross-product between forward direction and path direction
- Accelerates when far from goal, decelerates when close
- Provides expert demonstrations that can reach the goal

**BC procedure**:
- Collect teacher rollouts in parallel environments
- Train policy to predict teacher actions using mean-squared error loss
- Default: 2000 iterations with learning rate 3e-4

This warmstart dramatically improves convergence, especially in maze-like environments where random initialization would rarely produce successful trajectories.

#### Training Loop

1. **Rollout phase**: Collect trajectories from current policy in parallel environments
2. **GAE computation**: Compute advantages and returns using GAE
3. **Update phase**:
   - Flatten trajectories into a single batch
   - For each epoch:
     - Shuffle data
     - Process in minibatches
     - Compute PPO loss and update network

#### Loss Function

The PPO loss combines three terms:

```
L = L_policy + value_coef * L_value - entropy_coef * H
```

**Policy loss** (clipped surrogate objective):
```
L_policy = -mean(min(ratio * adv_norm, clip(ratio, 1-ε, 1+ε) * adv_norm))
```
where `ratio = exp(log_prob_new - log_prob_old)` and advantages are normalized.

**Value loss**:
```
L_value = mean((returns - value)²)
```

**Entropy bonus**:
```
H = mean(entropy(policy))
```

#### Evaluation

During training, we periodically evaluate the deterministic policy (using mean action, no exploration):
- Runs single episode from initial state
- Tracks minimum goal distance achieved
- Reports success if goal distance < 1.0

Best checkpoints are saved when evaluation goal distance improves.

---

### 2. Computational Optimizations

#### JAX JIT Compilation

**Key optimization**: All critical functions are JIT-compiled using `@jax.jit`:
- Environment `step()` function
- Policy network forward pass
- PPO update step

JAX's JIT compilation transforms Python/JAX code into optimized XLA computations, providing:
- 10-100x speedup compared to eager execution
- Efficient GPU/TPU utilization when available
- Automatic batching and parallelization opportunities

#### Vectorized Environment Execution

**Parallel rollouts**: Using `jax.vmap` to run multiple environments in parallel:
- Single `jax.vmap(env.step)` call processes all environments simultaneously
- No Python loops for environment steps
- All environments execute in parallel, constrained only by hardware parallelism

This provides near-linear speedup with number of parallel environments (up to hardware limits).

#### Efficient Data Structures

**Trajectory storage**: Trajectories are stored as JAX arrays, enabling:
- Efficient vectorized operations
- No Python loops for advantage computation
- Batch processing without data conversion overhead

**Flattening strategy**: Trajectories are flattened (`[horizon, num_envs, ...]` → `[horizon*num_envs, ...]`) only once before minibatching, avoiding repeated reshaping.

#### Minibatch Processing

Instead of processing entire rollouts at once, we:
1. Flatten all trajectories
2. Randomly shuffle
3. Process in minibatches of 1024 samples
4. Repeat for 4 epochs

This provides better sample efficiency and computational efficiency than full-batch updates.

#### Path Recomputation Optimization

The environment pathfinding (expensive operation) is only recomputed every second (every `fps` steps), not every step. This is safe because:
- Path changes slowly as agent moves
- Obstacles move predictably
- Small path errors don't significantly affect reward shaping

This optimization reduces computational cost by ~60x for pathfinding.

#### AutoResetWrapper

The `AutoResetWrapper` automatically resets environments when they terminate, maintaining a full batch of active environments without manual reset logic. This ensures:
- Constant batch size throughout training
- No wasted computation on terminated environments
- Seamless integration with vectorized operations

---

### 3. Difficulties Encountered and Resolution

#### Difficulty 1: Sparse reward problem

**Problem**: Initial attempts with sparse rewards (only on goal/collision) resulted in no learning. The agent never discovered successful trajectories due to exploration challenges.

**Solution**: Implemented dense reward shaping in Stage 1. The path-based rewards provide guidance at every step, making learning feasible. Combined with behavior cloning warmstart, the agent can learn from the start.

#### Difficulty 2: Action space scaling

**Problem**: The environment action space is asymmetric (acceleration: `[-4.0, 3.0]`, steering: `[-1.31, 1.31]` radians). Standard tanh squashing to `[-1, 1]` doesn't directly map to these ranges.

**Solution**: Implemented custom action scaling:
- Acceleration: Positive values scale to `[0, max_accel]`, negative to `[-max_brake, 0]`
- Steering: Symmetric scaling to `[-max_steer, max_steer]`

The policy outputs actions in `[-1, 1]` space, which are then scaled to environment ranges. This keeps the policy output normalized while respecting environment constraints.

#### Difficulty 3: Tanh-Gaussian log probability correction

**Problem**: When using tanh squashing, the log probability must account for the change of variables. The standard Gaussian log probability is incorrect for tanh-squashed actions.

**Solution**: Implemented correct log probability computation:
```python
def tanh_gaussian_log_prob(mean, log_std, tanh_actions):
    atanh = arctanh(clip(tanh_actions, -0.999, 0.999))
    log_prob = gaussian_log_prob(atanh, mean, log_std)
    log_prob -= sum(log(1 - tanh_actions² + 1e-6))  # Jacobian correction
    return log_prob
```

The Jacobian correction term `-log(1 - tanh²(a))` accounts for the tanh transformation, ensuring correct probability computation for PPO's importance sampling.

#### Difficulty 4: Advantage normalization and instability

**Problem**: Advantages can have very different scales across updates, causing training instability. Unnormalized advantages led to erratic policy updates.

**Solution**: Normalize advantages before computing policy loss:
```python
adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
```

This ensures advantages have zero mean and unit variance, providing stable, scale-invariant updates.

#### Difficulty 5: Behavior cloning convergence

**Problem**: Initial BC implementation struggled to learn the expert policy, even with simple expert. The policy would not converge to reasonable behavior.

**Solution**:
- Used tanh-squashed mean prediction (not raw mean) for BC loss
- Increased BC iterations to 2000
- Used same learning rate as PPO (3e-4)

The key insight is that BC should predict the expert's tanh-squashed actions, not the pre-tanh values, because that's what the policy actually outputs.

#### Difficulty 6: Memory efficiency with long horizons

**Problem**: Storing full trajectories for long horizons (256 steps) across many environments (8) with large observations (72 dims) requires significant memory.

**Solution**:
- Process trajectories immediately after rollout
- Flatten and shuffle for minibatching
- Don't store unnecessary intermediate values
- Use JAX's efficient array storage (no Python object overhead)

Additionally, JAX's memory management is more efficient than PyTorch/TensorFlow due to functional programming model and XLA optimizations.

#### Difficulty 7: Debugging JIT-compiled code

**Problem**: JIT compilation makes debugging difficult because code execution is deferred and errors occur in compiled XLA code, not original Python.

**Solution**:
- Use `jax.disable_jit()` for debugging
- Print values outside JIT-compiled functions
- Use `jax.block_until_ready()` to force eager execution for inspection
- Validate with small test cases before full training

---

### 4. Results and Reproduction

#### Results

The PPO implementation successfully trains agents that can navigate to the goal:
- Agents learn to follow the planned path efficiently
- Behavior is smooth and goal-directed
- Success rate improves over training (measured by minimum goal distance)

Training checkpoints are saved in `checkpoints/`:
- `robotaxi_ppo_step000100.msgpack` - After 100 updates
- `robotaxi_ppo_step000200.msgpack` - After 200 updates
- `robotaxi_ppo_step000300.msgpack` - After 300 updates
- Best checkpoints saved when evaluation improves

A demonstration video is available at `videos/robotaxi.mp4` showing a trained policy navigating the environment.

#### Reproduction Instructions

**Train a new policy:**

Full training run:
```bash
PYTHONPATH=src python src/train_robotaxi_ppo.py \
  --num-envs 8 --horizon 256 --pretrain-iters 3000 --updates 100 \
  --learning-rate 3e-4 --seed 0 \
  --save-dir checkpoints --save-every 100 --run-name robotaxi_ppo
```

Quick test run (fewer updates for sanity check):
```bash
PYTHONPATH=src python src/train_robotaxi_ppo.py \
  --num-envs 4 --horizon 128 --updates 10 --learning-rate 3e-4 --seed 0
```

**Play a trained policy and record video:**

```bash
PYTHONPATH=src python src/play_robotaxi_policy.py \
  --checkpoint checkpoints/robotaxi_ppo_step000100.msgpack \
  --map-id 1 --frames 800 \
  --record videos/robotaxi.mp4 --record-fps 60
```

For PNG frames instead of video:
```bash
PYTHONPATH=src python src/play_robotaxi_policy.py \
  --checkpoint checkpoints/robotaxi_ppo_step000100.msgpack \
  --map-id 1 --frames 800 \
  --record videos/frames/ --record-fps 60
```

**Interactive rendering:**

```bash
python src/main.py --map-id 1 --fps 60 --num-ray-sensors 32
```

**Key training outputs:**
- Console logs show: update number, mean reward, minimum goal distance, done rate, and loss components
- Checkpoints saved every `--save-every` steps (default: 100)
- Best checkpoints saved when evaluation goal distance < threshold
- Evaluation runs every 50 updates


