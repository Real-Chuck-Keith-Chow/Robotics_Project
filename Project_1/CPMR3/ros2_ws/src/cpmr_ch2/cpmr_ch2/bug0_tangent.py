# ~/CPMR3/ros2_ws/src/cpmr_ch2/cpmr_ch2/bug0_tangent.py
import math, json, os
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from ament_index_python.packages import get_package_share_directory

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# -------------------- small geometry helpers --------------------
EPS = 1e-6

def euler_from_quaternion(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    # roll
    s = 2.0 * (w * x + y * z); c = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(s, c)
    # pitch
    s = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, s)))
    # yaw
    s = 2.0 * (w * z + x * y); c = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(s, c)
    return roll, pitch, yaw

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def segment_circle_hits(p, g, c, R) -> bool:
    """Does segment PG intersect circle center c with radius R?"""
    px, py = p[0]-c[0], p[1]-c[1]
    gx, gy = g[0]-c[0], g[1]-c[1]
    dx, dy = gx-px, gy-py
    A = dx*dx + dy*dy
    B = 2*(px*dx + py*dy)
    C = px*px + py*py - R*R
    disc = B*B - 4*A*C
    if disc < 0:
        return False
    sqrt_disc = math.sqrt(max(0.0, disc))
    t1 = (-B - sqrt_disc) / (2*A)
    t2 = (-B + sqrt_disc) / (2*A)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)

def line_of_sight(p, g, circles) -> bool:
    # Slightly relax the inflated radius so we can leave the tangent cleanly.
    SLACK = 0.02  # 2 cm
    for (cx, cy, R) in circles:
        if segment_circle_hits(p, g, (cx, cy), max(0.0, R - SLACK)):
            return False
    return True

def tangent_points_from_external_point(p, c, R) -> List[Tuple[float,float]]:
    vx, vy = p[0]-c[0], p[1]-c[1]
    d2 = vx*vx + vy*vy
    d = math.sqrt(d2)
    if d <= R + EPS:
        return []
    ux, uy = vx/d, vy/d
    theta = math.acos(R/d)
    def rot(x, y, ang):
        ca, sa = math.cos(ang), math.sin(ang)
        return (x*ca - y*sa, x*sa + y*ca)
    r1 = rot(ux, uy,  theta)
    r2 = rot(ux, uy, -theta)
    t1 = (c[0] + R * r1[0], c[1] + R * r1[1])
    t2 = (c[0] + R * r2[0], c[1] + R * r2[1])
    return [t1, t2]

def choose_leave_point(p, g, circles, hit_idx, boundary_dir='ccw') -> Optional[Tuple[float,float]]:
    cx, cy, R = circles[hit_idx]
    tangents = tangent_points_from_external_point(p, (cx,cy), R)
    if not tangents:
        return None

    ang_p = math.atan2(p[1]-cy, p[0]-cx)
    ang_t = [math.atan2(t[1]-cy, t[0]-cx) for t in tangents]
    def delta_ccw(a,b):
        return (a-b) % (2*math.pi)
    ccw_idx = 0 if delta_ccw(ang_t[0], ang_p) < delta_ccw(ang_t[1], ang_p) else 1
    cw_idx  = 1 - ccw_idx
    order = [ccw_idx, cw_idx] if boundary_dir == 'ccw' else [cw_idx, ccw_idx]

    candidates = []
    for idx in order:
        t = tangents[idx]
        if line_of_sight(t, g, circles):
            ang_tgt = math.atan2(t[1]-cy, t[0]-cx)
            dccw = delta_ccw(ang_tgt, ang_p)
            dcw  = (2*math.pi - dccw)
            arc = R * (dccw if boundary_dir == 'ccw' else dcw)
            total = arc + dist(t, g)
            candidates.append((total, t))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

# -------------------- node --------------------
GO_STRAIGHT, BOUNDARY_FOLLOW, LEAVE_TO_GOAL = 0, 1, 2

class Bug0Tangent(Node):
    def __init__(self):
        super().__init__('bug0_tangent')
        self.declare_parameter('goal_x', 5.0)
        self.declare_parameter('goal_y', 5.0)
        self.declare_parameter('boundary_dir', 'ccw')
        self.declare_parameter('robot_radius', 0.18)
        self.declare_parameter('map', 'default.json')

        self._goal = (float(self.get_parameter('goal_x').value),
                      float(self.get_parameter('goal_y').value))
        self._boundary_dir = str(self.get_parameter('boundary_dir').value).lower()
        self._robot_r = float(self.get_parameter('robot_radius').value)
        map_name = self.get_parameter('map').value

        # load obstacles (JSON)
        pkg_share = get_package_share_directory('cpmr_ch2')
        map_path = map_name if os.path.isabs(map_name) else os.path.join(pkg_share, map_name)
        try:
            with open(map_path, 'r') as f:
                m = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Unable to open map file: {map_path} ({e})")
        self._circles = []
        for _, v in m.items():
            self._circles.append((float(v['x']), float(v['y']), float(v['r']) + self._robot_r))
        self.get_logger().info(f"Loaded {len(self._circles)} obstacles (inflated)")

        # state
        self._pose = (0.0, 0.0, 0.0)  # x,y,theta
        self._state = GO_STRAIGHT
        self._target_wp = None

        # io
        self._sub = self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.add_on_set_parameters_callback(self._on_param_set)

    # ----- ROS callbacks -----
    def _on_param_set(self, params):
        for p in params:
            if p.name == 'goal_x' and p.type_ == Parameter.Type.DOUBLE:
                self._goal = (float(p.value), self._goal[1])
            elif p.name == 'goal_y' and p.type_ == Parameter.Type.DOUBLE:
                self._goal = (self._goal[0], float(p.value))
            elif p.name == 'boundary_dir' and p.type_ == Parameter.Type.STRING:
                self._boundary_dir = str(p.value).lower()
        self.get_logger().info(f"New params: goal={self._goal}, boundary_dir={self._boundary_dir}")
        return SetParametersResult(successful=True)

    def _on_odom(self, msg: Odometry, vel_gain=2.0, max_vel=0.35,
                 reach_eps=0.12, wp_eps=0.20):
        # unpack pose
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(o)
        self._pose = (p.x, p.y, yaw)
        pos = (p.x, p.y)
        goal = self._goal

        if self._state == GO_STRAIGHT:
            if line_of_sight(pos, goal, self._circles):
                self._drive_to(goal, yaw, pos, vel_gain, max_vel)
                if dist(pos, goal) < reach_eps:
                    self._pub.publish(Twist())
                    self.get_logger().info("[DONE] reached goal")
            else:
                first_idx = self._first_intersected_circle(pos, goal)
                if first_idx is None:
                    self._drive_to(goal, yaw, pos, vel_gain, max_vel)
                    return
                leave = choose_leave_point(pos, goal, self._circles, first_idx, self._boundary_dir)
                if leave is None:
                    cx, cy, R = self._circles[first_idx]
                    v = math.atan2(pos[1]-cy, pos[0]-cx)
                    self._target_wp = (cx + R*math.cos(v), cy + R*math.sin(v))
                    self.get_logger().warn(f"[BOUNDARY] fallback wp {self._target_wp}")
                else:
                    self._target_wp = leave
                    self.get_logger().info(f"[BOUNDARY] circle#{first_idx} -> leave@{leave}")
                self._state = BOUNDARY_FOLLOW

        elif self._state == BOUNDARY_FOLLOW:
            wp = self._target_wp
            self._drive_to(wp, yaw, pos, vel_gain, max_vel)
            if dist(pos, wp) < wp_eps:
                # Nudge a bit past the tangent toward the goal
                self._target_wp = self._nudge_toward(pos, goal, d=0.12)
                self._state = LEAVE_TO_GOAL
                self.get_logger().info("[LEAVE] nudging past tangent, then straight-to-goal")

        elif self._state == LEAVE_TO_GOAL:
            if line_of_sight(pos, goal, self._circles):
                self._drive_to(goal, yaw, pos, vel_gain, max_vel)
                if dist(pos, goal) < reach_eps:
                    self._pub.publish(Twist())
                    self.get_logger().info("[DONE] reached goal")
            else:
                # keep moving to nudged waypoint until LOS is clear
                self._drive_to(self._target_wp, yaw, pos, vel_gain, max_vel)
                if dist(pos, self._target_wp) < 0.08:
                    self._target_wp = self._nudge_toward(pos, goal, d=0.12)

    # ----- helpers -----
    def _first_intersected_circle(self, p, g) -> Optional[int]:
        best_t, best_idx = None, None
        for i, (cx,cy,R) in enumerate(self._circles):
            px, py = p[0]-cx, p[1]-cy
            gx, gy = g[0]-cx, g[1]-cy
            dx, dy = gx-px, gy-py
            A = dx*dx + dy*dy
            B = 2*(px*dx + py*dy)
            C = px*px + py*py - R*R
            disc = B*B - 4*A*C
            if disc < 0:
                continue
            sqrt_disc = math.sqrt(max(0.0, disc))
            t1 = (-B - sqrt_disc) / (2*A)
            t2 = (-B + sqrt_disc) / (2*A)
            for t in (t1, t2):
                if 0.0 <= t <= 1.0:
                    if best_t is None or t < best_t:
                        best_t, best_idx = t, i
        return best_idx

    def _drive_to(self, tgt, yaw, pos, k_v=0.6, k_w=2.0, max_v=0.35, max_w=1.5):
        # Unicycle controller
        x_err = tgt[0] - pos[0]
        y_err = tgt[1] - pos[1]
        rho   = math.hypot(x_err, y_err)
        phi   = math.atan2(y_err, x_err)
        ang   = (phi - yaw + math.pi) % (2*math.pi) - math.pi

        v = max(min(k_v * rho, max_v), -max_v)
        w = max(min(k_w * ang, max_w), -max_w)

        twist = Twist()
        twist.linear.x  = v
        twist.angular.z = w
        self._pub.publish(twist)

    def _nudge_toward(self, src, dst, d=0.12):
        ux, uy = dst[0] - src[0], dst[1] - src[1]
        L = math.hypot(ux, uy) or 1.0
        return (src[0] + d * ux / L, src[1] + d * uy / L)

def main(args=None):
    rclpy.init(args=args)
    node = Bug0Tangent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
