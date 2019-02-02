
import argparse
parser = argparse.ArgumentParser()


parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)

args = parser.parse_args()

print(args.list)
