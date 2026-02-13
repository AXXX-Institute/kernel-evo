import argparse
import uvicorn

from kernel_generation.core.eval.server_rpc import app


def setup_parser(subparsers: argparse.ArgumentParser):
    parser = subparsers.add_parser("eval_server", help="Run kernel evaluation server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")


def eval_server(args: argparse.Namespace):
    print(f"Starting validation server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
