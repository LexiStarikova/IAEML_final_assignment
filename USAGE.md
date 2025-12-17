## Run Training:

```bash
PYTHONPATH=src python src/train_robotaxi_ppo.py \
  --num-envs 8 --horizon 256 --pretrain-iters 3000 --updates 100 \
  --learning-rate 3e-4 --seed 0 \
  --save-dir checkpoints --save-every 100 --run-name robotaxi_ppo
```

or

```bash
PYTHONPATH=src python src/train_robotaxi_ppo.py \
  --run-name robotaxi_ppo --save-dir checkpoints \
  --best-threshold 0.999
```


## Run Renderer:

```bash
python src/main.py --map-id 1 --fps 60 --num-ray-sensors 32
```

## Play a saved policy + record video

```bash
PYTHONPATH=src python src/play_robotaxi_policy.py \
  --checkpoint checkpoints/robotaxi_ppo_step000100.msgpack \
  --map-id 1 --frames 800 \
  --record videos/robotaxi.mp4 --record-fps 60
```

If `.mp4` writing fails (missing deps), pass `--record videos/frames/` to save PNG frames.
