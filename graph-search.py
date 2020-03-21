import numpy as np
from collections import deque
from matplotlib import pyplot as plt
import math
from queue import PriorityQueue
import time


class Graph(object):
  def __init__(self):
    self.n_vertices = 500
    size_world = 10.0

    # Compute vertices.
    # np.random.seed(1)
    self.vertices = np.random.rand(self.n_vertices, 2) * size_world

    # Compute adjacency matrix.
    self.adjacency = np.zeros((self.n_vertices, self.n_vertices))
    for row in range(self.n_vertices):
      for col in range(row + 1, self.n_vertices):
        if np.linalg.norm(
                self.vertices[row] -
                self.vertices[col]) < np.random.exponential(
                size_world /
                math.sqrt(self.n_vertices)):
          self.adjacency[row, col] = 1
          self.adjacency[col, row] = 1

    # Define start and goal nodes.
    self.start_node = np.random.randint(self.n_vertices)
    self.end_node = np.random.randint(self.n_vertices)
    # print(f'start: {self.start_node}')
    # print(f'end:   {self.end_node}')

  def plot(self, solution):
    fig, ax = plt.subplots()
    # Plot graph
    # Vertices
    ax.scatter(self.vertices[:, 0], self.vertices[:, 1], c='black')
    for idx, vertex in enumerate(self.vertices):
      ax.annotate(str(idx), tuple(vertex + np.array([0.1, 0.1])), c='blue')
    # Edges
    idxs_x, idxs_y = np.nonzero(self.adjacency)
    for idx_x, idx_y in zip(idxs_x, idxs_y):
      ax.plot(self.vertices[[idx_x, idx_y], 0],
              self.vertices[[idx_x, idx_y], 1], 'black', zorder=-1)

    # Plot the solution if one exists.
    if solution:
      # Plot solution.
      sol = np.array([self.vertices[vertex] for vertex in solution])
      ax.plot(sol[:, 0], sol[:, 1], 'ro', linestyle='-', linewidth=3, zorder=0)

    # Plot the Start and End nodes in green and blue.
    ax.scatter(self.vertices[self.start_node, 0], self.vertices[self.start_node, 1], c='green',
               zorder=1)
    ax.scatter(self.vertices[self.end_node, 0], self.vertices[self.end_node, 1], c='blue', zorder=2)
    plt.show()

  def get_node_children(self, node: int):
    return np.flatnonzero(self.adjacency[node])

  def solve_bfs(self):
    # Solve with breadth-first-search.
    frontier = deque([self.start_node])
    parent = {self.start_node: None}

    # Expand the frontier BFS-style.
    while frontier:
      # Explore the next node in the list.
      current_node = frontier.popleft()
      # print(f'expanding {current_node}')
      # If we haven't reached our goal yet, keep expanding.
      if current_node != self.end_node:
        # Add each of the children nodes to the frontier, unless they are already there.
        for child in self.get_node_children(current_node):
          # If the child node is already in the explored list (parent exists, so it's already been
          # explored / already has a parent), it must be that shortest path to that
          # node doesn't come from the current node because the algorithm has already explored paths
          # equal to or shorter than the current exploration depth.
          if child not in parent:
            parent[child] = current_node
            frontier.append(child)
      else:
        break

    # Backtrace the solution.
    if self.end_node in parent:
      # A solution was found since end_node has a parent.
      solution = []
      node = self.end_node
      while node is not None:
        solution.append(node)
        node = parent[node]
    else:
      # end_node has no parent, so no solution exists.
      solution = None

    # print(f"solution: {solution}")
    return solution

  def euclidean_distance(self, node_a: int, node_b: int):
    return np.linalg.norm(self.vertices[node_a] - self.vertices[node_b])

  def edge_cost(self, node_a: int, node_b: int):
    """ return infinite cost for nodes without an edge between them, otherwise return euclidean
    distance.
    """
    if self.adjacency[node_a, node_b]:
      return self.euclidean_distance(node_a, node_b)
    else:
      raise ValueError('Did not expect this to be called.')
      return math.inf

  def solve_dijkstra(self):
    # Start the queue to be explored (the frontier) with the start node.
    best_cost_to = {self.start_node: 0}
    frontier = PriorityQueue()
    frontier.put((best_cost_to[self.start_node], self.start_node))
    optimal_parent = {self.start_node: None}

    # If the frontier is empty, the queue has been exhausted without finding the terminal node.
    while not frontier.empty():
      # Get the current shortest-path node.
      current_cost, current_node = frontier.get()
      if current_cost > best_cost_to[current_node]:
        # This node is already known to be closer than this queue item understands. We've already
        # expanded this node (because queue items with a lower cost will have higher priority).
        continue
      # print(f"Exploring {current_node} at {current_cost:.2f} from start node.")

      # If it's the terminal node, we're done!
      if current_node == self.end_node:
        # print(f"Found {current_node} at {current_cost:.2f} from start node.")
        break

      for child_node in self.get_node_children(current_node):
        # Otherwise, examine all children of the node, updating their best_cost_to according to the
        # (known smallest) distance to the current node (the parent) plus the distance from parent
        # to the child.
        child_cost = current_cost + self.edge_cost(current_node, child_node)

        if child_node not in best_cost_to or best_cost_to[child_node] > child_cost:
          # This is just for debugging.
          # if child_node not in best_cost_to:
            # print(f"Added {child_node} to the frontier with best-so-far distance of {child_cost:.2f}")
          # else:
            # print(f"Updated {child_node} with best-so-far distance of {child_cost:.2f}")

          # If we're now closer to a child node, add it to the distance-prioritized frontier.
          # Note: It may be that the node already exists in the queue. If it does, no prob! When
          # it's popped off the queue, we'll ignore it (above).
          best_cost_to[child_node] = child_cost
          optimal_parent[child_node] = current_node
          frontier.put((child_cost, child_node))
        # else:
          # print(f"Not updating {child_node} because best-so-far {best_cost_to[child_node]:.2f} < current {current_cost:.2f}")

    if self.end_node in optimal_parent:
      # An optimal solution was found. Backtrace from terminal node to find optimal solution.
      solution = []
      node = self.end_node
      while node is not None:
        solution.append(node)
        node = optimal_parent[node]
    else:
      # The terminal node was never found. No solution exists.
      return None

    return solution

  def solve_a_star(self):
    best_cost_to = {self.start_node: 0.0}
    frontier = PriorityQueue()
    # Just put it with 0 estimated cost since it'll be popped immediately.
    frontier.put((0, self.start_node))
    optimal_parent = {self.start_node: None}

    while not frontier.empty():
      estimated_total_cost, current_node = frontier.get()
      # print(f"Expanding {current_node} with estimated total cost to goal of "
      #      f"{estimated_total_cost:.2f}.")

      # If it's the terminal node, end early.
      if current_node == self.end_node:
        # print(f"Found goal {current_node} with total cost of {best_cost_to[current_node]}."
        #       "Terminating early.")
        break

      for child_node in self.get_node_children(current_node):
        current_cost_to = best_cost_to[current_node]
        child_cost_to = current_cost_to + self.edge_cost(current_node, child_node)

        if child_node not in best_cost_to or best_cost_to[child_node] > child_cost_to:
          # Debugging only
          # if child_node not in best_cost_to:
          #   # print(f"Have not seen {child_node} before.")
          # elif best_cost_to[child_node] > child_cost_to:
          #   # print(f"New cost to {child_node} {child_cost_to} is better than previous "
          #         f"{best_cost_to[child_node]}.")

          best_cost_to[child_node] = child_cost_to
          optimal_parent[child_node] = current_node
          child_estimated_total_cost = child_cost_to + \
              self.euclidean_distance(child_node, self.end_node)
          frontier.put((child_estimated_total_cost, child_node))
        # else:
        #   # print(f"Cost to {child_node} from {current_node} is no better than previous best path.")

    if self.end_node in optimal_parent:
      # An optimal solution was found. Backtrace from terminal node to find optimal solution.
      solution = []
      node = self.end_node
      while node is not None:
        solution.append(node)
        node = optimal_parent[node]
    else:
      # The terminal node was never found. No solution exists.
      return None

    return solution


def main():
  while True:
    g = Graph()
    time1 = time.time()
    solution = g.solve_bfs()
    time2 = time.time()
    solution = g.solve_dijkstra()
    time3 = time.time()
    solution = g.solve_a_star()
    time4 = time.time()
    print("BFS:      ", solution)
    print('{:.3f} ms'.format((time2-time1) * 1000.0))
    print("Dijkstra: ", solution)
    print('{:.3f} ms'.format((time3-time2) * 1000.0))
    print("A star:   ", solution)
    print('{:.3f} ms'.format((time4-time3) * 1000.0))
    time.sleep(1.0)
  g.plot(solution)


if __name__ == '__main__':
  main()
