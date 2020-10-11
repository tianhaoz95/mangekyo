import argparse
from cond_mnist import train_cond_mnist
from mnist import train_mnist

parser = argparse.ArgumentParser(description='GAN getting started.')
parser.add_argument('--id', dest='project_id',
                    default='mnist',
                    help='The project ID.')
parser.add_argument('--epoch', dest='epoch', type=int, default=200,
                    help='How many epoch to train.')
parser.add_argument('--train_per_epoch', dest='train_per_epoch', type=int, default=300,
                    help='How many iteration to train per epoch.')
parser.add_argument('--interval', dest='interval', type=int, default=20,
                    help='Interval between eval.')
args = parser.parse_args()

if __name__ == '__main__':
    if 'cond_mnist' in args.project_id or 'cond_fashion_mnist' in args.project_id:
        train_cond_mnist(
            project_id=args.project_id,
            epoch=args.epoch,
            train_per_epoch=args.train_per_epoch,
            interval=args.interval
        )
    elif 'cond' not in args.project_id and ('mnist' in args.project_id or 'fashion_mnist' in args.project_id):
        train_mnist(
            project_id=args.project_id,
            epoch=args.epoch,
            train_per_epoch=args.train_per_epoch,
            interval=args.interval
        )
    else:
        print('Project ID not found')
