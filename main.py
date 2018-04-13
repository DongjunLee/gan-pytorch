#!/usr/bin/env python

from hbconfig import Config
import numpy as np

import argparse
import atexit

from data_loader import make_data_loader
from model import Model



def main(mode):
    model = Model(mode)
    model_func = model.build_fn()

    if mode == "train":
        train_loader = make_data_loader("train", Config.train.batch_size)
        model_func(train_loader)
    elif mode == "generate":
        latent_z = np.random.normal(0, 1)
        model_func(latent_z)
    else:
        raise ValueError(f"unknown mode: {self.mode}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode (train/test/train_and_evaluate)')
    args = parser.parse_args()

    # Print Config setting
    Config(args.config)
    print("Config: ", Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    # After terminated Notification to Slack
    # atexit.register(utils.send_message_to_slack, config_name=args.config)

    main(args.mode)
