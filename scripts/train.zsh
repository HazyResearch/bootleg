#!/usr/bin/zsh

export BOOTLEG_STRIP="true"
export BOOTLEG_LOWER="true"
export BOOTLEG_LANG_CODE="en"
export CONFIG_PATH="configs/standard/train.yaml"
export NUM_GPUS=1

# Train Bootleg
if [ $NUM_GPUS -lt 2 ]; then
  python3 -m bootleg.run --config_script $CONFIG_PATH
else
  python3 -m torch.distributed.run --nproc_per_node $NUM_GPUS --config_script $CONFIG_PATH
fi

echo "To load Bootleg model, run..."
echo "from bootleg.end2end.bootleg_annotator import BootlegAnnotator"
echo "ann = BootlegAnnotator(config=$CONFIG_PATH)"
