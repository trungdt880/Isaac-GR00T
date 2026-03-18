#!/bin/bash
sudo docker run -it --rm --name=groot --network=host --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /Data2/cache:/root/.cache -v /Data2/trungdt:/Data2/trungdt gr00t-dev