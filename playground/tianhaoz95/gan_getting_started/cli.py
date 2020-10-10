import argparse
from cond_mnist import train_cond_mnist
from pokemon import train_pokemon
from mnist import train_mnist

parser = argparse.ArgumentParser(description='GAN getting started.')
parser.add_argument('--id', dest='project_id',
                    default='mnist',
                    help='The project ID.')
args = parser.parse_args()

if __name__ == '__main__':
    if args.project_id == 'pokemon':
        train_pokemon()
    elif args.project_id == 'mnist' or args.project_id == 'fashion_mnist':
        train_mnist(args.project_id)
    elif args.project_id == 'cond_mnist' or args.project_id == 'cond_fashion_mnist':
        train_cond_mnist(args.project_id)
    else:
        print('Project ID not found')
