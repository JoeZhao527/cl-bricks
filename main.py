import os
import yaml
import pprint
import argparse
from easydict import EasyDict
from ensemble.pipeline import run, log


def parse_args():
    parser = argparse.ArgumentParser(description="Model configuration and execution arguments")

    parser.add_argument(
        'config',
        type=str,
        help="Path to the configuration file"
    )

    parser.add_argument(
        '--skip_train', 
        action='store_true', 
        help="Flag to skip training the model"
    )
    
    parser.add_argument(
        '--skip_test', 
        action='store_true', 
        help="Flag to skip making predictions for test samples"
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        help="Directory where the trained model is saved"
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))

    if args.skip_train:
        cfg.train = False
    
    if args.skip_test:
        cfg.test = False
    
    if args.model_dir:
        assert os.path.exists(args.model_dir), f"{args.model_dir} not exists"
        cfg.trained_model_dir = args.model_dir

    log(f"Pipeline starts with configurations:")
    pprint.pprint(cfg)

    run(cfg)