#   ______                                           __                 
#  /      \                                         /  |                
# /$$$$$$  | __   __   __   ______   _______        $$ |       __    __ 
# $$ |  $$ |/  | /  | /  | /      \ /       \       $$ |      /  |  /  |
# $$ |  $$ |$$ | $$ | $$ |/$$$$$$  |$$$$$$$  |      $$ |      $$ |  $$ |
# $$ |  $$ |$$ | $$ | $$ |$$    $$ |$$ |  $$ |      $$ |      $$ |  $$ |
# $$ \__$$ |$$ \_$$ \_$$ |$$$$$$$$/ $$ |  $$ |      $$ |_____ $$ \__$$ |
# $$    $$/ $$   $$   $$/ $$       |$$ |  $$ |      $$       |$$    $$/ 
#  $$$$$$/   $$$$$/$$$$/   $$$$$$$/ $$/   $$/       $$$$$$$$/  $$$$$$/ 
#
# File: train.py
# Author: Owen Lu
# Date:
# Email: jiangxiluning@gmail.com
# Description:
import sys
from typing import *
import logging
from pathlib import Path
import shutil

from src.tools.train_net import train

from loguru import logger
import click
import anyconfig
from easydict import EasyDict as edict
import yaml
import tensorflow as tf

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'

def parse_output(cfgs):
    dir_name = '{model_size}_{multiheads}_{encoders}_{decoders}_{encoder_ff}_{decoder_ff}_{encoder_dropout}_{decoder_dropout}_{optim}_{desc}' \
        .format(model_size=cfgs.model.model_size,
                multiheads=cfgs.model.multiheads,
                encoders=cfgs.model.encoder.stacks,
                decoders=cfgs.model.decoder.stacks,
                encoder_ff=cfgs.model.encoder.feed_forward_size,
                decoder_ff= cfgs.model.decoder.feed_forward_size,
                encoder_dropout=cfgs.model.encoder.dropout,
                decoder_dropout=cfgs.model.decoder.dropout,
                optim=cfgs.train.optim.name,
                desc=cfgs.system.desc)
    output_root = OUTPUT_ROOT / dir_name

    output_root.mkdir(parents=True, exist_ok=True)



    cfgs.system.outputs.root = output_root.as_posix()
    cfgs.system.outputs.tb_log_dir = (output_root / cfgs.system.outputs.tb_log_dir).as_posix()
    cfgs.system.outputs.training_log = (output_root / cfgs.system.outputs.training_log).as_posix()
    cfgs.system.outputs.checkpoints = (output_root / cfgs.system.outputs.checkpoints).as_posix()

def prepare_outputs(configs):
    output_root = Path(configs.system.outputs.root)

    conf_path = output_root / 'train.yaml'
    with conf_path.open(mode='w') as f:
        import json
        c = json.loads(json.dumps(configs))
        yaml.dump(c, f)

    code_output = output_root / 'code'
    if code_output.exists():
        shutil.rmtree(code_output.as_posix())

    shutil.copytree(PROJECT_ROOT, output_root / 'code', \
                    ignore=shutil.ignore_patterns('outputs', '.git', '.idea', 'configs', '.pytest_cache'))

def setup_logger(cfgs):
    logging_file = (Path(cfgs.system.outputs.root) / Path(cfgs.system.outputs.training_log)).as_posix()
    logger.remove()
    logger.add(sys.stdout, level=logging.INFO)
    logger.add(logging_file, level=logging.DEBUG)

@click.command()
@click.option('--resume', '-r', help='checkpoints', is_flag=True)
@click.option('--finetune', '-f', help='finetune with provided checkpoint')
@click.option('--config', '-c', type=click.File())
@click.option('--debug', '-d', default=False, is_flag=True)
def main(config, debug:bool, resume:bool, finetune:str):
    config = anyconfig.load(config)
    config = edict(config)
    parse_output(config)

    config['system'].root = PROJECT_ROOT.as_posix()
    config['system'].debug = debug
    config.train.checkpoints.resume = resume
    config.train.checkpoints.finetune = finetune
    tf.config.experimental_run_functions_eagerly(debug)

    setup_logger(config)
    prepare_outputs(config)
    train(config)


if __name__ == '__main__':
    main()