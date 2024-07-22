import json

import pandas as pd
from opts import Opts
from solver import RouteOpt
from utils import calculate_distance_matrix, cost_distance, plot_route


def main(opt):
    coords_df = pd.read_csv(opt.file_path)
    # Calculate the distance matrix for the coordinates
    dist_matrix = calculate_distance_matrix(coords_df)
    # Get the original path (in the order of input)
    orig_path = coords_df.index.tolist()
    # Calculate the total distance of the original path
    orig_dist = cost_distance(dist_matrix, orig_path)
    print("Original Distance: {:.2f}".format(orig_dist))

    # Start the optimization process
    print("Start to optimize the path...")
    route_opt = RouteOpt(dist_matrix=dist_matrix)

    # Solve the initial path using a greedy algorithm
    greedy_path = route_opt.solve()
    greedy_dist = cost_distance(dist_matrix, greedy_path)
    print("Greedy Distance: {:.2f}".format(greedy_dist))

    # Further optimize the path using 2-opt algorithm
    print("Start to optimize the path with 2-opt...")
    opt_path = route_opt.optimize_with_2opt(optim_steps=opt.optim_steps)
    opt_dist = cost_distance(dist_matrix, opt_path)
    print("Optimized Distance: {:.2f}".format(opt_dist))

    # Extract the Node ID sequence in the optimized path order
    opt_permutation = coords_df.reindex(opt_path)["Node_ID"].tolist()
    # Plot the optimized path
    plot_route(coords_df, opt_permutation)

    # Save the optimized route and distance to a JSON file
    result = {"route": opt_permutation, "distance": opt_dist}
    json.dump(result, open(opt.save_path, "w"))
    print(f"Results saved to {opt.save_path}")


if __name__ == "__main__":
    opt = Opts().parse()
    main(opt)
