## Run Training:

`python src/train_robotaxi_ppo.py --num-envs 8 --horizon 256 --pretrain-iters 3000 --updates 1500 --learning-rate 3e-4 --seed 0`


## Run Renderer:

`python src/main.py --map-id 1 --fps 60 --num-ray-sensors 32`
