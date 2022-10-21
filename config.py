import yaml
import argparse

parser = argparse.ArgumentParser(description='keras automatic speech recognition')
parser.add_argument('--config_path',
                    type=str,
                    default=r'yamls/evaluate_multitask.yaml',
                    help='config file path')
parser.add_argument('--train',
                        action='store_true',
                        help='train or not train')
args = parser.parse_args()

with open(args.config_path, "r", encoding='utf-8') as f:
    cfg = yaml.load(f, yaml.SafeLoader)

class Config:
    def __init__(self, is_train=True):
        for key, value in cfg.items():
            if key == 'train':
                if is_train:
                    for key_train, value_train in value.items():
                        setattr(self, key_train, value_train)
                continue
            if key == 'evaluate':
                if not is_train:
                    for key_eval, value_eval in value.items():
                        setattr(self, key_eval, value_eval)
                continue
            setattr(self, key, value)

config = Config(is_train=args.train)