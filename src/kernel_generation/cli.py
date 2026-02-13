import argparse
from kernel_generation.commands.evolve import evolve, setup_parser as setup_parser_evolve
from kernel_generation.commands.extract import extract, setup_parser as setup_parser_extract
from kernel_generation.commands.compare import compare, setup_parser as setup_parser_compare
from kernel_generation.commands.eval_server import eval_server, setup_parser as setup_parser_eval_server
from kernel_generation.commands.memory import memory, setup_parser as setup_parser_memory


def main():
    parser = argparse.ArgumentParser(description="Kernel Generation CLI")
    subparsers = parser.add_subparsers(dest="command")
    setup_parser_evolve(subparsers)
    setup_parser_extract(subparsers)
    setup_parser_compare(subparsers)
    setup_parser_eval_server(subparsers)
    setup_parser_memory(subparsers)

    args = parser.parse_args()
    if args.command == "evolve":
        evolve(args)
    elif args.command == "extract":
        extract(args)
    elif args.command == "compare":
        compare(args)
    elif args.command == "eval_server":
        eval_server(args)
    elif args.command == "memory":
        memory(args)
    else:
        parser.print_help()
