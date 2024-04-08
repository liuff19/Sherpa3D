#!/bin/bash

set -e

cd ..

obj="airplane"
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=25678 train.py --config configs/${obj}_geo.json -o ${obj}_geo
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=25678 train.py --config configs/${obj}_app.json -bm outputs/geometry/${obj}_geo/dmtet_mesh/mesh.obj -o ${obj}_app
