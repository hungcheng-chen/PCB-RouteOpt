import argparse
import os
import time


class Opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="2-opt Algorithm Configuration"
        )
        self.parser.add_argument(
            "--file_path",
            type=str,
            default="data/coords_df.csv",
            help="The path of the file containing the coordinates of the points",
        )
        self.parser.add_argument(
            "--optim_steps",
            type=int,
            default=5,
            help="The number of optimization steps for the 2-opt algorithm",
        )

    def parse(self, args=""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        # Get the root directory of the project
        opt.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Construct the absolute path for the file
        opt.file_path = os.path.join(opt.root_dir, opt.file_path)
        if not os.path.exists(opt.file_path):
            raise FileNotFoundError(f"File not found: {opt.file_path}")

        # Generate a timestamp for creating unique save directories
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        opt.save_dir = os.path.join(opt.root_dir, "runs", f"{time_str}")

        # Create the save directory if it doesn't exist
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir, exist_ok=True)
            print(f"Created directory {opt.save_dir}")

        # Define the path for saving the result
        opt.save_path = os.path.join(opt.save_dir, "result.json")

        return opt
