import numpy as np
import random
import math
from matplotlib import pyplot as plt

random.seed(1)


class Node:
  def __init__(self, position, parent, cost):
    self.position = position
    self.parent = parent
    # Cost to get to this node from root.
    self.cost = cost

  def __str__(self):
    return f"parent: {self.parent}; cost: {self.cost}"


class Tree:
  def __init__(self, root, distance_metric):
    self.max_nodes = 10000
    self.nodes = [root]
    self.goal_node_idxs = []
    self.distance_metric = distance_metric

  def add(self, node):
    if len(self.nodes) >= self.max_nodes:
      return None

    self.nodes.append(node)
    idx_added_node = len(self.nodes) - 1

    return idx_added_node

  def near_idxs(self, position, radius):
    near_nodes_idx = []
    near_nodes_ds = []
    for i, node in enumerate(self.nodes):
      this_ds = self.distance_metric(node.position, position)
      if this_ds < radius:
        near_nodes_idx.append(i)
        near_nodes_ds.append(this_ds)

    # Return the near set.
    return near_nodes_idx

  def nearest(self, position):
    nearest_idx = None
    nearest_dist = 10e9
    for i, node in enumerate(self.nodes):
      this_dist = self.distance_metric(node.position, position)
      if this_dist < nearest_dist:
        nearest_idx = i
        nearest_dist = this_dist

    # Return the near set.
    return nearest_idx


class World:
  def __init__(self):
    self.xrange = [-10, 10]
    self.yrange = [-10, 10]
    self.initial_position = (-8, 0)
    # Refers to Z dimension.
    self.goal_pose_z = 0
    # Distance at which to create new nodes from network.
    self.dq = 1.0
    # Precision of collision checking along paths between nodes.
    self.precision = 0.25

    num_obstacles = 3
    self.obstacles = self.make_obstacles(self.xrange, self.yrange, num_obstacles)

  def plot_obstacles(self, ax):
    for ob in self.obstacles:
      xmin, ymin, xmax, ymax = ob
      ax.plot(np.array([xmin, xmin, xmax, xmax, xmin]),
              np.array([ymin, ymax, ymax, ymin, ymin]), 'red')

  @staticmethod
  def make_obstacles(xrange, yrange, num_obstacles):
    obstacles = []
    for i in range(num_obstacles):
      # Box obstacles
      xmin = random.uniform(*xrange)
      xmax = random.uniform(*xrange)
      if xmax < xmin:
        xmin, xmax = xmax, xmin

      ymin = random.uniform(*yrange)
      ymax = random.uniform(*yrange)
      if ymax < ymin:
        ymin, ymax = ymax, ymin

      obstacle = xmin, ymin, xmax, ymax
      obstacles.append(obstacle)
    return obstacles

  @staticmethod
  def X_distance(position1, position2):
    x1, y1 = position1
    x2, y2 = position2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

  def is_collision(self, position):
    def is_collision_obstacle(position, obstacle):
      x, y = position
      xmin, ymin, xmax, ymax = obstacle
      if xmin <= x <= xmax and ymin <= y <= ymax:
        return True
      return False

    for obstacle in self.obstacles:
      if is_collision_obstacle(position, obstacle):
        return True
    return False

  def has_collision(self, X0, X1):
    if X0 == X1:
      return self.is_collision(X0)

    # Number of intermediate points, including origin.
    path_count = math.ceil(self.X_distance(X0, X1) / self.precision)
    x0, y0 = X0
    x1, y1 = X1
    dx = (x1 - x0) / path_count
    dy = (y1 - y0) / path_count

    for i in range(path_count):
      xi = x0 + i * dx
      yi = y0 + i * dy
      Xi = (xi, yi)
      if self.is_collision(Xi):
        return True
    if self.is_collision(X1):
      return True
    return False

  def at_goal(self, position):
    z_resolution = 0.1
    if abs(self.z(position) - self.goal_pose_z) < z_resolution:
      return True
    else:
      return False

  # In our state space, the robot has DOFs X, Y, and "pose" / output state z = f(X) = f(x,y)
  def z(self, position):
    g_x, g_y = 7.5, 0
    x, y = position
    z = (x - g_x)**2 + (y - g_y)**2
    return z

  def random_X(self):
    return (random.uniform(*self.xrange), random.uniform(*self.yrange))

  def steer(self, root_position, goal_position):
    d = self.X_distance(root_position, goal_position)
    dt = self.dq / d  # parametric distance from root to goal

    x_root, y_root = root_position
    x_goal, y_goal = goal_position

    dx = (x_goal - x_root) * dt
    dy = (y_goal - y_root) * dt

    x_out = x_root + dx
    y_out = y_root + dy
    return x_out, y_out

  def rrt_star(self):
    root = Node(self.initial_position, None, 0)
    distance_metric = self.X_distance
    tree = Tree(root, distance_metric)

    # TODO Handle case where root is already at goal.
    # TODO Handle case where root is in collision.
    # TODO Handle case where goal is in collision.

    fig, ax = plt.subplots()

    max_expansion = 1000
    for i in range(max_expansion):
      # TODO Add occasional greedy choice.
      X_random = self.random_X()
      nearest_node_idx = tree.nearest(X_random)
      X_nearest = tree.nodes[nearest_node_idx].position
      X_new = self.steer(X_nearest, X_random)

      # Add to node list if it's not in collision.
      if not self.has_collision(X_nearest, X_new):
        # Note: This does not yet contain X_new
        neighbor_idxs = tree.near_idxs(position=X_new, radius=2.0)
        # Connect X_new to best "near" node. Cost to traverse is euclidean distance in X.
        n_nearest = tree.nodes[nearest_node_idx]
        best_parent_idx = nearest_node_idx
        # Minimum cost to get to X_new through neighbors.
        cost_through_best_parent = n_nearest.cost + self.X_distance(n_nearest.position, X_new)
        for neighbor_idx in neighbor_idxs:
          n_neighbor = tree.nodes[neighbor_idx]
          new_cost_through_neighbor = n_neighbor.cost + self.X_distance(n_neighbor.position, X_new)
          if new_cost_through_neighbor < cost_through_best_parent and not self.has_collision(
                  n_neighbor.position, X_new):
            best_parent_idx = neighbor_idx
            cost_through_best_parent = new_cost_through_neighbor

        # Add X_new to tree through best "near" node.
        n_new = Node(X_new, best_parent_idx, cost_through_best_parent)
        # print("added node")
        # print(n_new)
        n_new_idx = tree.add(n_new)

        # Connect all neighbors of X_new to X_new if that path cost is less.
        for neighbor_idx in neighbor_idxs:
          # TODO don't search over best_cost_idx (the best parent for n_new)
          n_neighbor = tree.nodes[neighbor_idx]
          neighbor_cost_through_new = n_new.cost + self.X_distance(n_neighbor.position,
                                                                   n_new.position)
          if neighbor_cost_through_new < n_neighbor.cost and not self.has_collision(
                  n_new.position, n_neighbor.position):
            # Best path for neighbor is now through x_new
            n_neighbor.parent = n_new_idx
            n_neighbor.cost = neighbor_cost_through_new
            # Update the neighbor node in the tree.
            tree.nodes[neighbor_idx] = n_neighbor

        # If the new node is at the goal, so noted.
        if self.at_goal(n_new.position):
          tree.goal_node_idxs.append(n_new_idx)

        # Plot
        ax.scatter(X_new[0], X_new[1], c='black')
        # ax.annotate(str(n_new_idx), (X_new[0] + 0.1, X_new[1] + 0.1), c='blue')
        X_parent = tree.nodes[n_new.parent].position
        ax.plot(np.array([X_parent[0], X_new[0]]), np.array([X_parent[1], X_new[1]]), 'black')

    if tree.goal_node_idxs:
      print("reached goal at nodes:", tree.goal_node_idxs)
      goal_node_idx = tree.goal_node_idxs[0]
      goal_path = []
      parent = goal_node_idx
      while parent:
        goal_path.append(parent)
        parent = tree.nodes[parent].parent
      # Print starting at root.
      goal_path = list(reversed(goal_path))
      print(f"path through {goal_node_idx}: {goal_path}")
      print("cost:", tree.nodes[goal_node_idx].cost)

    self.plot_obstacles(ax)


w = World()
w.rrt_star()
plt.show()
