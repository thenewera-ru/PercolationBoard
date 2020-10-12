import argparse
from pathlib import Path
import os


def root() -> str:
    return str(Path(os.getcwd()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=30,
                        help='Size of the grid')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='Output the grid')
    parser.add_argument('--save_graph', type=str, default=False,
                        help='Save the graph or not')
    return parser.parse_args()
