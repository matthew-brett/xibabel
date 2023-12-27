#!/usr/bin/env python3
""" Fetch named test set(s)
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter

from . import fetcher


def get_parser():
    parser = ArgumentParser(description=__doc__,  # Usage from docstring
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('test_set_name', nargs='+',
                        help='Name(s) of test sets to fetch')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    for test_set in args.test_set_name:
        fetcher.get_set(test_set)


if __name__ == '__main__':
    main()
