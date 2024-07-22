import numpy as np
from tqdm import tqdm


class RouteOpt:
    def __init__(self, dist_matrix: np.ndarray, end_point: tuple[int, int] = None):
        """
        Initialize the RouteOpt class with a distance matrix and an optional endpoint.

        Args:
            dist_matrix (np.ndarray): A 2D numpy array representing the distance matrix.
            end_point (tuple[int, int], optional): A tuple containing the start and end points. If None, the route is not a loop.
        """
        self.dist_matrix = dist_matrix
        self.N = len(dist_matrix)
        self.end_point = end_point
        self.start, self.end = end_point or (None, None)
        # Determine if the route is a loop
        self.is_loop = (self.start is not None) and (self.start == self.end)
        self.check_validity()

    def check_validity(self):
        """
        Check the validity of the start and end points, and the distance matrix.

        Raises:
            ValueError: If the start or end points are out of valid range or if the distance matrix is not properly formed.
        """
        # Check if the start point is within the valid range
        if self.start is not None and not (0 <= self.start < self.N):
            raise ValueError("Start point is out of valid range")
        # Check if the end point is within the valid range
        if self.end is not None and not (0 <= self.end < self.N):
            raise ValueError("End point is out of valid range")

        # Handle special cases based on the size of the distance matrix
        if self.N == 0:  # If the distance matrix is empty
            return []
        if self.N == 1:  # If the distance matrix has only one element
            return [0, 0] if self.is_loop else [0]
        if (
            self.N == 2 and self.is_loop
        ):  # If the distance matrix has two elements and is a loop
            return [self.start, 1 - self.start, self.start]

        self._assert_triangular()

    def _assert_triangular(self):
        """
        Ensure that the distance matrix is at least lower triangular.

        Raises:
            ValueError: If the distance matrix is not at least lower triangular.
        """
        for i, row in enumerate(self.dist_matrix):
            if len(row) < i:
                raise ValueError(
                    f"Distance matrix must be at least lower triangular. Row {i} must have at least {i} elements"
                )

    def pairs_by_dist(self) -> np.ndarray:
        """
        Generate pairs of nodes sorted by distance.

        Returns:
            np.ndarray: An array of node pairs sorted by the distances between them in ascending order.
        """
        pairs = np.triu_indices(
            self.N, k=1
        )  # Get the indices of the upper triangle of the matrix
        pairs = np.transpose(pairs)  # Transpose to get the required format
        pairs = np.sort(pairs, axis=1)[:, ::-1]  # Sort each pair
        # Sort the pairs by the distances between them
        pairs = pairs[
            np.argsort(self.dist_matrix[pairs[:, 0], pairs[:, 1]], kind="mergesort")
        ]
        return pairs

    def edge_connects_endpoint_segments(
        self, i: int, j: int, segments: list[list[int]]
    ) -> bool:
        """
        Check if the edge connects the endpoint segments.

        Args:
            i (int): The first node of the edge.
            j (int): The second node of the edge.
            segments (list[list[int]]): A list of segments where each segment is a list of nodes.

        Returns:
            bool: True if the edge connects the endpoint segments, False otherwise.
        """
        si, sj = segments[i], segments[j]
        ss, se = segments[self.start], segments[self.end]
        # Check if the edge connects the start and end segments
        return (si is ss) and (sj is se) or (sj is ss) and (si is se)

    def restore_path(self) -> list[int]:
        """
        Restore the path from the connections.

        Returns:
            list[int]: A list of node indices representing the restored path.
        """
        need_revert = False
        if self.start is None:
            if self.end is None:
                # Find the first node with only one connection
                self.start = next(
                    idx for idx, conn in enumerate(self.connections) if len(conn) == 1
                )
            else:
                # In this case, start from the end point and then reverse the path
                self.start = self.end
                need_revert = True

        # Generate the path
        path = [self.start]
        prev_point = None
        cur_point = self.start
        # Iterate over all connections to generate the complete path
        for _ in range(len(self.connections) - (0 if self.is_loop else 1)):
            next_point = next(
                pnt for pnt in self.connections[cur_point] if pnt != prev_point
            )
            path.append(next_point)
            prev_point, cur_point = cur_point, next_point
        if need_revert:
            return path[::-1]  # Reverse the path if needed
        else:
            return path

    def solve(self) -> list[int]:
        """
        Execute the solver to find the optimal path.

        Returns:
            list[int]: A list of node indices representing the optimal path.
        """
        # Initially, each node has 2 "sticky ends" (available connection points)
        node_valency = np.array([2] * self.N)
        has_both_endpoints = (self.start is not None) and (self.end is not None)

        if not self.is_loop:
            # If not a loop, start and end points have only 1 "sticky end"
            if self.start is not None:
                node_valency[self.start] = 1
            if self.end is not None:
                node_valency[self.end] = 1

        # Store 1 or 2 connections for each node
        connections = [set() for _ in range(self.N)]
        segments = [[i] for i in range(self.N)]

        sorted_pairs = self.pairs_by_dist()

        edges_left = self.N - 1
        # Iterate over the node pairs sorted by distance
        for i, j in tqdm(sorted_pairs, desc="Greedy ", total=len(sorted_pairs)):
            i, j = int(i), int(j)
            if node_valency[i] and node_valency[j] and (segments[i] is not segments[j]):
                # Skip if the edge connects start and end points but is not the last edge
                if (
                    has_both_endpoints
                    and edges_left != 1
                    and self.edge_connects_endpoint_segments(i, j, segments)
                ):
                    continue

                node_valency[i] -= 1
                node_valency[j] -= 1
                connections[i].add(j)
                connections[j].add(i)

                # Merge the two segments
                new_segment = segments[i] + segments[j]
                for node_idx in new_segment:
                    segments[node_idx] = new_segment

                edges_left -= 1
                if edges_left == 0:
                    break

        # If looking for a loop, close it
        if self.is_loop:
            """
            Modify connections to close the loop
            """
            i, j = (i for i, conn in enumerate(connections) if len(conn) == 1)
            connections[i].add(j)
            connections[j].add(i)

        self.connections = connections
        self.path = self.restore_path()

        return self.path

    def optimize_with_2opt(self, optim_steps):
        """
        Perform 2-opt optimization, prioritizing segments that can significantly
        improve path length to accelerate convergence.

        Parameters:
        optim_steps (int): The number of optimization steps to perform.
        """
        for passn in range(optim_steps):
            # Create a progress bar to monitor the optimization process
            progress_bar = tqdm(
                range(self.N - 4), desc="2-opt steps = {}".format(passn)
            )

            d_total = 0.0  # Initialize total distance change
            swap_count = 0  # Initialize swap count

            for a in progress_bar:
                b, c, d = a + 1, a + 3, a + 4

                # Define path segments
                path_a = self.path[a]  # Current point
                path_b = self.path[b]  # Point next to current
                path_c = self.path[c : self.N - 1]  # Segment starting two steps ahead
                path_d = self.path[d : self.N]  # Segment starting three steps ahead

                # Calculate distances for current connections (a-b and c-d)
                ds_ab = self.dist_matrix[path_a][path_b]
                ds_cd = self.dist_matrix[path_c, path_d]

                # Calculate distances for potential new connections (a-c and b-d)
                ds_ac = self.dist_matrix[path_a][path_c]
                ds_bd = self.dist_matrix[path_b][path_d]

                # Compute the distance difference from swapping paths
                delta_d = ds_ab + ds_cd - (ds_ac + ds_bd)

                if np.any(delta_d > 0):  # Check if any swap results in improvement
                    index = np.argmax(delta_d)  # Find the most beneficial swap
                    d_total += delta_d[index]  # Accumulate total distance improvement
                    swap_count += 1  # Increment swap count

                    # Update progress bar with current optimization metrics
                    progress_bar.set_postfix(
                        {"Opt Dist": d_total, "Swap Count": swap_count},
                        refresh=True,
                    )

                    # Identify the optimal connections to swap
                    c_opt = path_c[index]
                    d_opt = path_d[index]

                    # Update connections to perform the 2-opt swap
                    self.connections[path_a].remove(path_b)
                    self.connections[path_a].add(c_opt)
                    self.connections[path_b].remove(path_a)
                    self.connections[path_b].add(d_opt)

                    self.connections[c_opt].remove(d_opt)
                    self.connections[c_opt].add(path_a)
                    self.connections[d_opt].remove(c_opt)
                    self.connections[d_opt].add(path_b)

                    # Restore the path after the swap
                    self.path[:] = self.restore_path()

            progress_bar.close()
        return self.path
