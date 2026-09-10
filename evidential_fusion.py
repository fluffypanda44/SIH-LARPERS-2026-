"""
Urban Search & Rescue (USAR) Evidential Multi-Modal Fusion Engine.
Replaces naive if/else rule lists with Dempster-Shafer evidential reasoning.
Fuses visual keypoint posture, metric distance, and acoustic DoA signals to compute
formal belief probabilities and entrapment severity indices.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Types from existing modules
from acoustic_doa import AcousticTarget
from survivor_detector import SurvivorTarget


@dataclass
class FusedMissionBelief:
    belief_victim_present: float         # Fused belief probability [0.0 - 1.0]
    uncertainty_mass: float              # Epistemic uncertainty mass m(Theta)
    entrapment_severity_index: float     # Calibrated triage severity index [0 - 100]
    hypothesis: str                      # "LINE_OF_SIGHT_LOCK", "SUB_SURFACE_BURIED", "RECON_PATROL"
    target_heading_deg: float            # Fused search & approach heading
    target_distance_m: float             # Metric target distance
    recommended_speed: float             # Kinematic speed recommendation (m/s)
    action_text: str                     # Mission dispatch statement
    is_emergency_halt: bool              # True if vehicle has arrived or critical obstacle


class EvidentialFusionEngine:
    """
    Dempster-Shafer Multi-Sensor Evidential Fusion Engine for USAR operations.
    Combines conflicting, uncertain sensory evidence from vision and acoustics.
    """

    def __init__(self):
        self.last_fused_heading = 0.0

    def fuse(
        self,
        survivors: List[SurvivorTarget],
        acoustic_target: AcousticTarget,
        costmap: np.ndarray,
    ) -> FusedMissionBelief:
        """
        Fuses visual survivor detections, acoustic DoA radar, and costmap traversability.
        """
        h, w = costmap.shape[:2]

        # -------------------------------------------------------------
        # 1. VISUAL EVIDENCE MASS FUNCTION: m_v
        # -------------------------------------------------------------
        if len(survivors) > 0:
            target = survivors[0]
            c_vis = target.confidence
            vis_joints = target.visible_joints

            # Posture and entrapment modifiers
            if target.posture == "PRONE_FLAT":
                posture_mod = 1.25
            elif target.posture == "PINNED_UPPER":
                posture_mod = 1.35
            else:
                posture_mod = 1.0

            # Mass for Survivor hypothesis from vision
            m_v_surv = min(0.95, c_vis * posture_mod * (vis_joints / 17.0 * 0.5 + 0.5))
            m_v_nonsurv = max(0.02, (1.0 - c_vis) * 0.40)
            m_v_theta = max(0.01, 1.0 - m_v_surv - m_v_nonsurv)

            # Entrapment Severity Index: (0 - 100)
            # Low visible joints + horizontal aspect ratio = high structural pinning probability
            missing_joints = 17 - vis_joints
            bx1, by1, bx2, by2 = target.bbox
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            aspect_ratio = bw / bh

            severity_raw = (missing_joints / 17.0) * 60.0 + min(40.0, aspect_ratio * 25.0)
            severity_index = max(10.0, min(100.0, severity_raw))

            has_visual = True
            v_heading = target.bearing_deg
            v_dist = target.est_distance_m
        else:
            m_v_surv = 0.05
            m_v_nonsurv = 0.65
            m_v_theta = 0.30
            severity_index = 0.0
            has_visual = False
            v_heading = 0.0
            v_dist = 10.0

        # -------------------------------------------------------------
        # 2. ACOUSTIC EVIDENCE MASS FUNCTION: m_a
        # -------------------------------------------------------------
        if acoustic_target.is_active:
            snr_factor = min(1.0, max(0.1, (acoustic_target.energy_db + 60.0) / 40.0))
            c_aud = acoustic_target.confidence

            m_a_surv = min(0.85, c_aud * snr_factor)
            m_a_nonsurv = max(0.05, (1.0 - c_aud) * 0.30)
            m_a_theta = max(0.05, 1.0 - m_a_surv - m_a_nonsurv)
            has_acoustic = True
            a_heading = acoustic_target.azimuth_deg
        else:
            m_a_surv = 0.05
            m_a_nonsurv = 0.70
            m_a_theta = 0.25
            has_acoustic = False
            a_heading = 0.0

        # -------------------------------------------------------------
        # 3. DEMPSTER'S RULE OF COMBINATION
        # -------------------------------------------------------------
        # Conflict factor K
        k = m_v_surv * m_a_nonsurv + m_v_nonsurv * m_a_surv
        normalization = 1.0 - k + 1e-12

        # Combined fused belief for Survivor present
        combined_surv = (
            m_v_surv * m_a_surv +
            m_v_surv * m_a_theta +
            m_v_theta * m_a_surv
        ) / normalization

        combined_theta = (m_v_theta * m_a_theta) / normalization
        fused_belief = float(np.clip(combined_surv, 0.0, 1.0))
        epistemic_uncertainty = float(np.clip(combined_theta, 0.0, 1.0))

        # -------------------------------------------------------------
        # 4. HYPOTHESIS ARBITRATION & NAV DISPATCH
        # -------------------------------------------------------------
        # Check forward path obstacle density
        forward_sector = costmap[int(h * 0.72):, int(w * 0.35):int(w * 0.65)]
        mean_forward_cost = float(np.mean(forward_sector)) if forward_sector.size > 0 else 0.0

        is_emergency_halt = False

        if has_visual and v_dist <= 1.2:
            hypothesis = "LINE_OF_SIGHT_EXTRACTION"
            heading = v_heading
            dist = v_dist
            speed = 0.0
            is_emergency_halt = True
            action = f"HALT: SURVIVOR AT {dist:.1f}m | TRIAGE SEVERITY: {severity_index:.0f}/100"

        elif has_visual:
            hypothesis = "LINE_OF_SIGHT_APPROACH"
            # Visual and acoustic spatial consensus check
            if has_acoustic and abs(v_heading - a_heading) < 25.0:
                # High-confidence multi-modal alignment
                heading = 0.75 * v_heading + 0.25 * a_heading
                speed = 0.55
                action = f"MULTI-MODAL LOCK: ADVANCE ON TARGET (SEV {severity_index:.0f}/100) | ALIGNED WITH BEACON"
            else:
                heading = v_heading
                speed = 0.50 if severity_index < 60 else 0.35
                turn_dir = "RIGHT" if heading > 0 else "LEFT"
                action = f"VISUAL INTERCEPT: STEER {abs(heading):.1f}° {turn_dir} -> ADVANCE {v_dist:.1f}m"
            dist = v_dist

        elif has_acoustic and acoustic_target.confidence > 0.25:
            # High acoustic evidence with zero line-of-sight visual
            hypothesis = "SUB_SURFACE_BURIED_VICTIM"
            heading = a_heading
            dist = 4.5
            speed = 0.35
            turn_dir = "RIGHT" if heading > 0 else "LEFT"
            action = f"HYPOTHESIS [BURIED SURVIVOR]: HOMING {abs(heading):.1f}° {turn_dir} ON DISTRESS BEACON"

        else:
            # Autonomous patrol on stable ground
            hypothesis = "TERRAIN_RECON_PATROL"
            if mean_forward_cost > 150:
                heading = 28.0
                speed = 0.0
                action = "OBSTACLE DETECTED: ROLLOUT SCANNING ALTERNATE CORRIDOR"
            else:
                heading = 0.0
                speed = 0.60
                action = "SECTOR PATROL: FORWARD PATH CLEAR ON STABLE GROUND"
            dist = 10.0

        self.last_fused_heading = heading

        return FusedMissionBelief(
            belief_victim_present=fused_belief,
            uncertainty_mass=epistemic_uncertainty,
            entrapment_severity_index=severity_index,
            hypothesis=hypothesis,
            target_heading_deg=heading,
            target_distance_m=dist,
            recommended_speed=speed,
            action_text=action,
            is_emergency_halt=is_emergency_halt,
        )
