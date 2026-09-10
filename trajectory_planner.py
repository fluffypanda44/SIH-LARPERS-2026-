"""
Urban Search & Rescue (USAR) Dynamic Window Trajectory Rollout Planner (DWA).
Evaluates real-time candidate kinematic arcs over the 2D Nav2 metric costmap.
Proves active obstacle avoidance and visual trajectory rollouts (Tesla/Waymo style).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrajectoryCandidate:
    v: float                          # Linear velocity (m/s)
    w: float                          # Angular velocity (rad/s)
    points_px: List[Tuple[int, int]]  # Screen-space (x, y) coordinates
    is_collision: bool                # True if trajectory clips lethal obstacle/rubble
    score: float                      # Total utility score
    clearance: float                  # Minimum clearance to obstacle
    heading_error_deg: float          # Final heading deviation from goal


class DWAPlanner:
    """
    Dynamic Window Approach local trajectory planner.
    Simulates forward candidate motion arcs and selects the optimal collision-free path.
    """

    def __init__(
        self,
        max_speed: float = 0.65,
        max_yaw_rate: float = 1.2,
        sim_time_s: float = 1.4,
        time_step_s: float = 0.10,
        num_yaw_samples: int = 19,
    ):
        self.max_speed = max_speed
        self.max_yaw = max_yaw_rate
        self.sim_time = sim_time_s
        self.dt = time_step_s
        self.num_yaw = num_yaw_samples
        self.yaw_candidates = np.linspace(-self.max_yaw, self.max_yaw, self.num_yaw)

    def plan(
        self,
        costmap: np.ndarray,
        goal_heading_deg: float,
        target_dist_m: float = 5.0,
        is_emergency_halt: bool = False,
    ) -> Tuple[float, float, List[TrajectoryCandidate], Optional[TrajectoryCandidate]]:
        """
        Rolls out kinematic trajectories over the Nav2 metric costmap.
        Returns:
            best_v: Optimal linear velocity (m/s)
            best_w: Optimal angular velocity (rad/s)
            candidates: All evaluated trajectory arcs
            best_candidate: The chosen winning trajectory
        """
        h, w = costmap.shape[:2]
        origin_x = w // 2
        origin_y = int(h * 0.96)

        if is_emergency_halt:
            return 0.0, 0.0, [], None

        candidates: List[TrajectoryCandidate] = []
        best_candidate: Optional[TrajectoryCandidate] = None
        best_score = -1e9

        # Convert goal heading to radians
        goal_rad = math.radians(goal_heading_deg)

        # Scale factor: meters to costmap pixels (estimated ~250 pixels per meter forward view)
        px_per_m = h * 0.38

        for yaw_rate in self.yaw_candidates:
            # Slower linear speed on sharp turns for stability
            turn_ratio = abs(yaw_rate) / self.max_yaw
            linear_v = self.max_speed * (1.0 - 0.45 * turn_ratio)

            # Integrate forward unicycle kinematics
            x_m = 0.0
            y_m = 0.0
            theta = 0.0  # 0 rad points straight ahead (upwards)

            pts: List[Tuple[int, int]] = []
            is_collision = False
            min_clearance = 255.0
            mean_cost_accum = 0.0
            step_count = 0

            t = 0.0
            while t <= self.sim_time:
                # Forward kinematics: y is forward, x is lateral
                theta += yaw_rate * self.dt
                # In standard robotics: positive yaw = counter-clockwise (Left)
                dx = -linear_v * math.sin(theta) * self.dt
                dy = linear_v * math.cos(theta) * self.dt

                x_m += dx
                y_m += dy

                # Project metric displacement into costmap pixel coordinates
                px_x = int(origin_x + x_m * px_per_m)
                px_y = int(origin_y - y_m * px_per_m)

                # Boundary check
                if px_x < 15 or px_x >= w - 15 or px_y < 15 or px_y >= h - 10:
                    break

                pts.append((px_x, px_y))

                # Sample costmap cell and neighbors
                cell_cost = int(costmap[px_y, px_x])
                mean_cost_accum += cell_cost
                step_count += 1

                # Lethal obstacle threshold (walls, furniture, drop-offs)
                if cell_cost >= 220:
                    is_collision = True
                    break

                min_clearance = min(min_clearance, 255.0 - cell_cost)
                t += self.dt

            if len(pts) < 4:
                continue

            # Heading error relative to target goal
            final_heading_deg = math.degrees(-theta)
            heading_err = abs(final_heading_deg - goal_heading_deg)

            # DWA Multi-Objective Score function:
            # Score = alpha * (180 - heading_error) + beta * clearance + gamma * velocity
            if is_collision:
                score = -1e6 + len(pts)  # penalize collision, allow path to draw up to impact point
            else:
                heading_score = (180.0 - min(180.0, heading_err)) / 180.0
                clearance_score = min_clearance / 255.0
                velocity_score = linear_v / self.max_speed

                score = (
                    0.50 * heading_score +
                    0.35 * clearance_score +
                    0.15 * velocity_score
                )

            cand = TrajectoryCandidate(
                v=round(linear_v, 3),
                w=round(yaw_rate, 3),
                points_px=pts,
                is_collision=is_collision,
                score=score,
                clearance=min_clearance,
                heading_error_deg=heading_err,
            )
            candidates.append(cand)

            if not is_collision and score > best_score:
                best_score = score
                best_candidate = cand

        # Fallback if all paths collide: emergency stop and turn in place towards goal
        if best_candidate is None:
            best_v = 0.0
            best_w = 0.4 if goal_heading_deg > 0 else -0.4
        else:
            best_v = best_candidate.v
            best_w = best_candidate.w

        return best_v, best_w, candidates, best_candidate

    def render_trajectories(
        self,
        overlay_img: np.ndarray,
        candidates: List[TrajectoryCandidate],
        best_candidate: Optional[TrajectoryCandidate],
    ):
        """
        Renders colorful Waymo/Tesla-style candidate trajectory splines on the live camera view:
        - Green = Winning optimal trajectory
        - Cyan/Yellow = Valid alternative candidates
        - Red = Colliding/unsafe trajectories
        """
        if not candidates:
            return

        canvas = overlay_img.copy()

        # Draw rejected or alternative trajectories first
        for cand in candidates:
            if len(cand.points_px) < 2:
                continue

            pts = np.array(cand.points_px, dtype=np.int32).reshape((-1, 1, 2))

            if cand.is_collision:
                # Collision: thin red line
                cv2.polylines(canvas, [pts], isClosed=False, color=(40, 40, 220), thickness=1, lineType=cv2.LINE_AA)
                # Impact marker (red X)
                impact_pt = cand.points_px[-1]
                cv2.circle(canvas, impact_pt, 3, (0, 0, 255), -1)
            elif cand is not best_candidate:
                # Safe alternative: subtle cyan/gray line
                cv2.polylines(canvas, [pts], isClosed=False, color=(160, 160, 60), thickness=1, lineType=cv2.LINE_AA)

        # Draw winning optimal trajectory on top
        if best_candidate and len(best_candidate.points_px) >= 2:
            pts = np.array(best_candidate.points_px, dtype=np.int32).reshape((-1, 1, 2))
            # Glowing green trajectory ribbon
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 120), thickness=3, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [pts], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            # Draw trajectory arrowhead at tip
            tip = best_candidate.points_px[-1]
            prev = best_candidate.points_px[-2]
            angle = math.atan2(tip[1] - prev[1], tip[0] - prev[0])
            arrow_len = 12.0
            p_left = (int(tip[0] - arrow_len * math.cos(angle - 0.5)), int(tip[1] - arrow_len * math.sin(angle - 0.5)))
            p_right = (int(tip[0] - arrow_len * math.cos(angle + 0.5)), int(tip[1] - arrow_len * math.sin(angle + 0.5)))
            arrow_pts = np.array([tip, p_left, p_right], dtype=np.int32)
            cv2.fillPoly(canvas, [arrow_pts], (0, 255, 120))

        # Alpha-blend splines onto camera image
        cv2.addWeighted(canvas, 0.75, overlay_img, 0.25, 0, overlay_img)
