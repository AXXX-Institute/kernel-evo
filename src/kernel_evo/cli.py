import argparse
from kernel_evo.commands.evolve import evolve, setup_parser as setup_parser_evolve
from kernel_evo.commands.extract import extract, setup_parser as setup_parser_extract
from kernel_evo.commands.compare import compare, setup_parser as setup_parser_compare
from kernel_evo.commands.eval_server import eval_server, setup_parser as setup_parser_eval_server
from kernel_evo.commands.memory import memory, setup_parser as setup_parser_memory

KERNEL_EVO_COMMANDS = {
    "evolve": evolve,
    "extract": extract,
    "compare": compare,
    "eval-server": eval_server,
    "memory": memory,
}

def main():
    parser = argparse.ArgumentParser(description="Kernel Generation CLI")
    subparsers = parser.add_subparsers(dest="command")
    setup_parser_evolve(subparsers)
    setup_parser_extract(subparsers)
    setup_parser_compare(subparsers)
    setup_parser_eval_server(subparsers)
    setup_parser_memory(subparsers)

    args = parser.parse_args()
    command_fn = KERNEL_EVO_COMMANDS.get(args.command)
    if command_fn:
        command_fn(args)
    else:
        parser.print_help()
