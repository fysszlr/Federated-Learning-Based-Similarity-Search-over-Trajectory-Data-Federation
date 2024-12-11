import sys
import logging
import argparse

from model.fed_trajcl import TrajCLTrainer, lcss_test
from config import Config
from utils import tool_funcs


def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.

    parser = argparse.ArgumentParser(description="TrajCL/train.py")
    parser.add_argument('--dumpfile_uniqueid', type=str, help='see config.py')
    parser.add_argument('--seed', type=int, help='')
    parser.add_argument('--dataset', type=str, help='')
    parser.add_argument('--moon_loss_weight', type=float, default=0.1, help='')
    parser.add_argument('--ldp', type=int, help='1? use ldp: not use ldp')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


# nohup python train.py --dataset porto &> result &
if __name__ == '__main__':
    Config.update(parse_args())
    logging.basicConfig(level=logging.DEBUG if Config.debug else logging.INFO,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[
                            logging.FileHandler(Config.root_dir + '/exp/log/' + tool_funcs.log_file_name(), mode='w'),
                            logging.StreamHandler()]
                        )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    trajcl = TrajCLTrainer(Config.trajcl_aug1, Config.trajcl_aug2)
    # trajcl.load_checkpoint()
    trajcl.train()
    trajcl.test()
    # lcss_test()
    # trajcl.knn_test('discrete_frechet')
    # trajcl.personal_test()
