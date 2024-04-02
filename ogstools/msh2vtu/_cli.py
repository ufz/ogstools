# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import argparse

from ogstools import __version__
from ogstools.msh2vtu import msh2vtu


def argparser() -> argparse.ArgumentParser:
    # parsing command line arguments
    def get_help(arg: str) -> str:
        assert msh2vtu.__doc__ is not None
        return msh2vtu.__doc__.split(arg + ":")[1].split(":param")[0].strip()

    parser = argparse.ArgumentParser(
        description=msh2vtu.__doc__.split(":param")[0].strip()
        if msh2vtu.__doc__ is not None
        else ""
    )
    add_arg = parser.add_argument
    add_arg("filename", help=get_help("filename"))
    add_arg("-o", "--output_path", default="", help=get_help("output_path"))
    add_arg("-p", "--prefix", default="", help=get_help("prefix"))
    add_arg("-d", "--dim", type=int, nargs="*", default=0, help=get_help("dim"))
    add_arg("-z", "--delz", action="store_true", help=get_help("delz"))
    add_arg("-s", "--swapxy", action="store_true", help=get_help("swapxy"))
    add_arg("-r", "--reindex", action="store_true", help=get_help("reindex"))
    add_arg("-k", "--keep_ids", action="store_true", help=get_help("keep_ids"))
    add_arg("-a", "--ascii", action="store_true", help=get_help("ascii"))
    add_arg("-l", "--log_level", default="DEBUG", help=get_help("log_level"))
    version = f"msh2vtu (part of ogstools {__version__}, Dominik Kern)"
    add_arg("-v", "--version", action="version", version=version)

    return parser


def cli() -> int:
    """command line use"""
    args = argparser().parse_args()

    return msh2vtu(
        filename=args.filename,
        output_path=args.output_path,
        output_prefix=args.prefix,
        dim=args.dim,
        delz=args.delz,
        swapxy=args.swapxy,
        reindex=args.reindex,
        keep_ids=args.keep_ids,
        ascii=args.ascii,
        log_level=args.log_level,
    )
