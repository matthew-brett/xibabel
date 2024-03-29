#!/usr/bin/env python3
""" Run Tornado static file server at given path
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import asyncio

from tornado import web


def make_app(path):
    return web.Application([
        (r"/(.*)", web.StaticFileHandler, {"path": path})
    ])


async def run_server(path, port=8899):
    (path / 'server_marker').write_text('static server')
    app = make_app(path)
    app.listen(int(port))
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()


def get_parser():
    parser = ArgumentParser(description=__doc__,  # Usage from docstring
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('files_path',
                        help='Path from which to serve files')
    parser.add_argument('-p', '--port', default='8899',
                        help='Port for server')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='If set, show messages')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.verbose:
        print(f'Serving {args.files_path} on port {args.port}')
    asyncio.run(run_server(Path(args.files_path), args.port))


if __name__ == '__main__':
    main()
