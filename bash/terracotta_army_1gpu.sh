#!/bin/bash

set -e

cd ..

obj="army"
python3 train.py --config configs/${obj}_geo.json -o ${obj}_geo
python3 train.py --config configs/${obj}_app.json -bm outputs/geometry/${obj}_geo/dmtet_mesh/mesh.obj -o ${obj}_app