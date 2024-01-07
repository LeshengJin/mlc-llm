"""A command line tool for directly chatting with open source LLMs"""
import os

from mlc_chat import ChatConfig, ChatModule
from mlc_chat.callback import StreamToStdout

from ..support.argparse import ArgumentParser


def print_available_commands():
    """Function printing available internal commands."""
    print("You can use the following special commands:")
    print("\t/help\t\tprint the special commands")
    print("\t/exit\t\tquite the cli")
    print("\t/stats\t\tprint out the latest stats (token/sec)")
    print("\t/reset\t\trestart a fresh chat")
    print(
        "\t/reload [model]\treload model `model` from disk, or reload\
             the current model if `model` not not specified"
    )


def chat(modelname: str, devicename: str, chatconfig: ChatConfig, modellib: str):
    """Command line chat with a specific open source LLM."""
    cm = ChatModule(
        model=modelname, device=devicename, chat_config=chatconfig, model_lib_path=modellib
    )

    print("Using MLC config: ", os.path.abspath(cm.config_file_path))
    print("Using model weights:", os.path.abspath(cm.model_path))
    print("Using model library:", os.path.abspath(cm.model_lib_path))

    print_available_commands()
    ps = cm._get_role_0()  # pylint: disable=W0212

    prompt = input(f"{ps} ")

    while prompt != "/exit":
        if prompt == "/help":
            print_available_commands()
        elif prompt == "/stats":
            print(f"Statistics: {cm.stats()}\n")
        elif prompt == "/reset":
            cm.reset_chat()
        elif prompt == "/reload":
            print("not yet implemented.")
        else:
            cm.generate(
                prompt,
                progress_callback=StreamToStdout(callback_interval=2),
            )
        prompt = input(f"{ps} ")

    # Print prefill and decode performance statistics
    print(f"Statistics: {cm.stats()}\n")


def main(argv):
    """Main entry point."""
    parser = ArgumentParser(description="Chat with open source LLMs")
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model or path to the quantized weights",
        required=True,
    )
    parser.add_argument(
        "--model-lib",
        type=str,
        default=None,
        help="Path to the model library",
        required=False,
    )
    parser.add_argument(
        "--chat-config",
        type=str,
        default=None,
        help="The model's configuration file",
        required=False,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="The GPU to use.",
        required=False,
    )

    args = parser.parse_args(argv)

    chat(
        modelname=args.model,
        devicename=args.device,
        chatconfig=args.chat_config,
        modellib=args.model_lib,
    )
