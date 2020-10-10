import argparse
from cond_mnist import train_cond_mnist
from mnist import train_mnist

parser = argparse.ArgumentParser(description='GAN getting started.')
parser.add_argument('--id', dest='project_id',
                    default='mnist',
                    help='The project ID.')
args = parser.parse_args()

if __name__ == '__main__':
    if 'cond_mnist' in args.project_id or 'cond_fashion_mnist' in args.project_id:
        train_cond_mnist(args.project_id)
    elif 'cond' not in args.project_id and ('mnist' in args.project_id or 'fashion_mnist' in args.project_id):
        train_mnist(args.project_id)
    else:
        print('Project ID not found')
