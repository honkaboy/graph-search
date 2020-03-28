class Node:
  def __init__(self, position, best_parent, best_cost):
    self.position = position
    self.best_parent = best_parent
    self.best_cost = best_cost


class World:
  def __init__(self):
    self.xrange = [-10, 10]
    self.yrange = [-10, 10]
    self.initial_state = [0, 0]
    # Refers to Z dimension.
    self.goal_pose_z = 5

    num_obstacles = 2
    self.obstacles = obstacles(xrange, yrange, num_obstacles)

  @staticfunction
  def obstacles(xrange, yrange, num_obstacles):
    obstacles = []
    for i in range(num_obstacles):
      # Box obstacles
      xmin = random.uniform(xrange)
      xmax= random.uniform(xrange)
      if xmax < xmin:
        xmin, xmax = xmax, xmin

      ymin = random.uniform(xrange)
      ymax= random.uniform(xrange)
      if ymax < ymin:
        ymin, ymax = ymax, ymin

      obstacle = xmin, ymin, xmax, ymax
      obstacles.append(obstacle)
    return obstacles

  def is_collision(self, position):
    is_collision = False
    def is_collision_obstacle(position, obstacle):
      x,y = position
      xmin,ymin,xmax,ymax = obstacle
      if xmin <= x <= xmax and ymin <= y <= ymax:
        return True
      return False

    for obstacle in self.obstacles:
      is_collision = is_collision_obstacle(position, obstacle)
    return is_collision


  # In our state space, the robot has DOFs X, Y, and "pose" / output state Z = f(X,Y)
  def z(self, x, y):
    z = 2 * x - y
    return z


  def rrt_star(self):

