import numpy as np
import random


class Node:
  def __init__(self, position, parent, cost):
    self.position = position
    self.parent = parent
    # Cost to get to this node from root.
    self.cost = cost

  def distance_squared(self, other_position):
    return (self.position[0] - other_position[0])**2 + (self.position[1] - other_position[1])**2


class Tree:
  def __init__(self, root):
    self.max_nodes = 10000
    self.nodes = [root]
    self.goal_node_idxs = []

  def add(self, node):
    if len(self.nodes >= self.max_nodes):
      return None

    self.nodes.append(node)
    idx_added_node = len(self.nodes) - 1

    if self.at_goal(node.position):
     self.goal_node_idxs.append(idx_added_node)

    return idx_added_node

  def nearest_idxs(self, position, radius=None):
    near_nodes_idx = []
    near_nodes_ds = []
    for i, node in enumerate(self.nodes):
      this_ds = node.distance_squared(position)
      if this_ds < min_ds:
        near_nodes_idx.append(i)
        near_nodes_ds.append(this_ds)

    # Return the near set.
    return near_nodes_idx


class World:
  def __init__(self):
    self.xrange = [-10, 10]
    self.yrange = [-10, 10]
    self.initial_position = (0,0)
    # Refers to Z dimension.
    self.goal_pose_z = 5
    # Distance at which to create new nodes from network.
    self.dq = 0.5
    # Precision of collision checking along paths between nodes.
    self.precision = 0.25

    num_obstacles = 2
    self.obstacles = self.make_obstacles(self.xrange, self.yrange, num_obstacles)

  @staticmethod
  def distance(position1, position2):
    x1, y1 = position1
    x2, y2 = position2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

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

  def is_collision(self, position):
    is_collision = False

    def is_collision_obstacle(position, obstacle):
      x, y = position
      xmin, ymin, xmax, ymax = obstacle
      if xmin <= x <= xmax and ymin <= y <= ymax:
        return True
      return False

    for obstacle in self.obstacles:
      is_collision = is_collision_obstacle(position, obstacle)
    return is_collision

  def has_collision(self, X0, X1):
    # Number of intermediate points, including origin.
    path_count = math.floor(self.distance(X0, X1) / self.precision)
    x0, y0 = X0
    x1, y1 = X1
    dx = x0 / path_count
    dy = y0 / path_count

    for i in range(path_count):
      xi = x0 + i * dx
      yi = y0 + i * dy
      Xi = (xi, yi)
      if is_collision(Xi):
        return True
    return False

  def at_goal(self, position):
    if abs(z(position) - self.goal_pose_z) < self.dq:
      return True
    else:
      return False

  # In our state space, the robot has DOFs X, Y, and "pose" / output state z = f(X) = f(x,y)
  def z(self, position):
    x, y = X
    z = 2 * x - y
    return z

  def random_X(self):
    return (random.uniform(*self.xrange), random.uniform(*self.yrange))

  def steer(self, root_position, goal_position):
    d = self.distance(root_position, goal_position)
    dt = self.dq / d  # parametric distance from root to goal

    x_root, y_root = root_position
    x_goal, y_goal = goal_position

    dx = (x_goal - x_root) * dt
    dy = (y_goal - y_root) * dt

    x_out = x_root + dx
    y_out = y_root + dy
    return x_root, y_root

  def rrt_star(self):
    root = Node(self.initial_position, None, 0)
    tree = Tree(root)

    max_expansion = 10
    for i in range(max_expansion):
      # TODO Add occasional greedy choice.
      X_random = self.random_X()
      nearest_node_idx = tree.nearest(X_random)
      X_nearest = tree.nodes[nearest_node_idx].position
      X_new = self.steer(X_nearest, X_random)

      # Add to node list if it's not in collision.
      if not self.has_collision(X_nearest, X_new):
        # Note: This does not yet contain X_new
        neighbor_idxs = tree.near_idxs(position=X_new, radius=1.0)
        # Connect X_new to best "near" node.
        best_parent = nearest_node_idx
        # Cost to traverse is euclidean distance in X.
        n_nearest = tree.nodes[nearest_node_idx]
        best_parent_idx = nearest_node_idx
        # Minimum cost to get to X_new through neighbors.
        cost_through_best_parent = n_nearest.cost + self.distance(n_nearest.position, X_new)
        for neighbor_idx in neighbor_idxs:
          n_neighbor = tree.nodes[neighbor_idx]
          new_cost_through_neighbor = n_neighbor.cost + self.distance(n_neighbor.position, X_new)
          if new_cost_through_neighbor < cost_through_best_parent and not self.has_collision(
                  n_neighbor.position, X_new):
            best_parent_idx = neighbor_idx
            cost_through_best_parent = new_cost_through_neighbor

        # Add X_new to tree through best "near" node.
        n_new = Node(X_new, best_cost_idx, best_cost)
        n_new_idx = tree.add(n_new)

        # Connect all neighbors of X_new to X_new if that path cost is less.
        for neighbor_idx in neighbor_idxs:
          # TODO don't search over best_cost_idx (the best parent for n_new)
          n_neighbor = tree.nodes[neighbor_idx]
          neighbor_cost_through_new = n_new.cost + self.distance(n_neighbor.position,
                                                                 n_new.position)
          if neighbor_cost_through_new < n_neighbor.cost and not self.has_collision(
                  n_new.position, n_neighbor.position):
            # Best path for neighbor is now through x_new
            n_neighbor.parent = n_new_idx
            n_neighbor.cost = neighbor_cost_through_new
            # Update the neighbor node in the tree.
            tree[neighbor_idx] = n_neighbor

    if tree.goal_node_idxs:
      print("reached goal at nodes:", tree.goal_node_idxs)
      goal_node_idx = tree.goal_node_idxs[0]
      goal_path = []
      parent = goal_node_idx
      while parent:
        goal_path.append(parent)
        parent = tree.nodes[parent].parent
      print(f"path through {goal_node_idx}: {goal_path}")
      print("cost:", tree.nodes[goal_node_idx].cost)

w = World()
w.rrt_star()
