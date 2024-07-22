import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance


def calculate_distance_matrix(df, metric="euclidean"):
    """
    Calculate the distance matrix for the given dataframe.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the coordinates.
    metric (str): Distance metric to use for calculating distances.

    Returns:
    numpy.ndarray: Distance matrix.
    """
    points = df[["x_int", "y_int"]].values
    return distance.cdist(points, points, metric=metric)


def cost_distance(distance_matrix, route):
    """
    Calculate the total distance of the given route based on the distance matrix.

    Parameters:
    distance_matrix (numpy.ndarray): Precomputed distance matrix.
    route (list[int]): List of indices representing the route.

    Returns:
    float: Total distance of the route.
    """
    return distance_matrix[route[:-1], route[1:]].sum().item()


def plot_route(coords_df, route):
    """
    Plot the given route on a 2D plane.

    Parameters:
    coords_df (pandas.DataFrame): DataFrame containing coordinates with 'Node_ID', 'x_int', and 'y_int' columns.
    route (list[int]): List of indices representing the route.
    """
    # Reorder the DataFrame according to the route
    coords_df = coords_df.set_index("Node_ID").reindex(route).reset_index()

    # Create a plot
    _, ax = plt.subplots()
    ax.plot(
        coords_df["x_int"], coords_df["y_int"], color="gray", linewidth=1
    )  # Plot the route
    ax.scatter(coords_df["x_int"], coords_df["y_int"], s=1)  # Plot the points
    ax.set_title("Plot Route")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Display the plot
    plt.show()
