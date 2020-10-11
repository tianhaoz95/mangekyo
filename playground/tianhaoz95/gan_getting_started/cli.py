import argparse
from cond_mnist import train_cond_mnist
from mnist import train_mnist

parser = argparse.ArgumentParser(description='GAN getting started.')
parser.add_argument('--id', dest='project_id',
                    default='mnist',
                    help='The project ID.')
parser.add_argument('--epoch', dest='epoch', type=int, default=200,
                    help='How many epoch to train.')
args = parser.parse_args()

if __name__ == '__main__':
    if 'cond_mnist' in args.project_id or 'cond_fashion_mnist' in args.project_id:
        train_cond_mnist(args.project_id, args.epoch)
    elif 'cond' not in args.project_id and ('mnist' in args.project_id or 'fashion_mnist' in args.project_id):
        train_mnist(args.project_id, args.epoch)
    else:
        print('Project ID not found')
