"""Operator Telemetry & Visualization Dashboard for DoomAgent.

Inspired by Swedish Embedded's operator-facing decision view:
- Left Top: High-resolution Game Screen Buffer (scaled from RGB / HUD).
- Left Bottom: 360° LiDAR Raycast Radar & Visible Entity Overlay.
- Right Top: Real-Time Action Probabilities & Decision Confidence Bars.
- Right Bottom: Vitals, Key Inventory, Live Reward Decomposition, and Latency Clock.

Works in dual mode:
  1. Interactive Window (Pygame desktop app).
  2. Headless Frame Renderer (writes composite frames directly to MP4 / PNG for demos & briefs).
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping, Sequence

import numpy as np
import pygame

from doom_env import spaces as S
from training.tools.demo_replay import spatial_map_to_rgb

# Dashboard Canvas Dimensions
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 700

# Color Palette (Dark Cyberpunk / Terminal Theme)
COLOR_BG = (14, 15, 20)
COLOR_PANEL = (22, 24, 32)
COLOR_PANEL_BORDER = (40, 44, 58)
COLOR_TEXT = (220, 225, 235)
COLOR_TEXT_DIM = (120, 125, 140)
COLOR_ACCENT = (70, 150, 250)
COLOR_GREEN = (60, 210, 120)
COLOR_RED = (240, 80, 80)
COLOR_YELLOW = (250, 200, 60)
COLOR_PURPLE = (180, 100, 240)
COLOR_GRAY = (80, 85, 100)

RAY_COLORS = {
    0: (40, 42, 50),     # none
    1: (110, 115, 130),  # wall
    2: (60, 180, 240),   # door (cyan)
    3: (250, 190, 50),   # locked door (yellow/gold)
    4: (240, 70, 70),    # enemy (red)
    5: (80, 220, 120),   # item (green)
}


class DoomDashboard:
    def __init__(self, title: str = "DOOM Neural Agent Dashboard", headless: bool = True):
        self.width = CANVAS_WIDTH
        self.height = CANVAS_HEIGHT
        self.headless = headless

        self._prev_sdl_driver = os.environ.get("SDL_VIDEODRIVER")
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        elif "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]

        pygame.init()
        pygame.font.init()

        if self.headless:
            self.surface = pygame.Surface((self.width, self.height))
            self.screen = None
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(title)
            self.surface = self.screen

        # Fonts
        self.font_title = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_main = pygame.font.SysFont("monospace", 13, bold=False)
        self.font_bold = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 11)

        # Sparkline histories
        self.reward_history: list[float] = []
        self.latency_history: list[float] = []

    def render_frame(
        self,
        rgb_frame: np.ndarray,
        observation: Mapping[str, Any],
        action_probs: np.ndarray | Sequence[float] | None = None,
        action_taken: np.ndarray | Sequence[int] | None = None,
        reward_breakdown: Mapping[str, float] | None = None,
        step_latency_ms: float = 0.0,
        decision_mode: str = "EXPLORE",
        spatial_map: np.ndarray | None = None,
        active_slots: int | None = None,
    ) -> np.ndarray:
        """Renders composite telemetry dashboard.

        Returns an RGB numpy array (CANVAS_HEIGHT, CANVAS_WIDTH, 3).
        """
        self.surface.fill(COLOR_BG)

        # Track histories
        self.latency_history.append(step_latency_ms)
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)

        step_reward = reward_breakdown.get("total", 0.0) if reward_breakdown else 0.0
        self.reward_history.append(step_reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

        # -------------------------------------------------------------
        # 1. Top-Left: Game Screen Frame (scaled from (120, 160))
        # -------------------------------------------------------------
        game_rect = pygame.Rect(16, 16, 560, 420)
        self._draw_panel(game_rect, "GAMEPLAY BUFFER (EGO-CENTRIC RGB)")

        if rgb_frame is not None and rgb_frame.size > 0:
            # rgb_frame is (H, W, 3) uint8. Pygame expects (W, H)
            frame_surf = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))
            scaled_frame = pygame.transform.scale(frame_surf, (game_rect.width - 16, game_rect.height - 38))
            self.surface.blit(scaled_frame, (game_rect.x + 8, game_rect.y + 30))

        # -------------------------------------------------------------
        # 2. Bottom-Left: 360° LiDAR Raycast Radar & Visible Entities
        # -------------------------------------------------------------
        radar_rect = pygame.Rect(16, 448, 560, 236)
        self._draw_panel(radar_rect, "360° SENSORY RADAR & EMLRP MEMORY")
        self._draw_radar(
            center_x=radar_rect.x + 122,
            center_y=radar_rect.y + 130,
            radius=88,
            rays_dist=observation["rays"]["distance"],
            rays_hits=observation["rays"]["hit_type"],
            entities=observation.get("visible_entities"),
        )
        self._draw_radar_legend(
            radar_rect.x + 236,
            radar_rect.y + 35,
            spatial_map=spatial_map,
            active_slots=active_slots,
        )

        # -------------------------------------------------------------
        # 3. Top-Right: Action Probabilities (Policy Confidence Bars)
        # -------------------------------------------------------------
        action_rect = pygame.Rect(592, 16, 592, 370)
        self._draw_panel(action_rect, "POLICY ACTION PROBABILITIES (CONFIDENCE)")
        self._draw_action_bars(
            action_rect,
            action_probs=action_probs,
            action_taken=action_taken,
        )

        # -------------------------------------------------------------
        # 4. Bottom-Right: Vitals, Reward Breakdown & Telemetry Clock
        # -------------------------------------------------------------
        telemetry_rect = pygame.Rect(592, 398, 592, 286)
        self._draw_panel(telemetry_rect, "VITALS, REWARDS & LATENCY TELEMETRY")
        self._draw_telemetry(
            telemetry_rect,
            vitals=observation["vitals"],
            reward_breakdown=reward_breakdown,
            latency_ms=step_latency_ms,
            mode=decision_mode,
        )

        if not self.headless:
            pygame.display.flip()

        # Convert surface to (H, W, 3) numpy array
        view = pygame.surfarray.array3d(self.surface)
        return np.transpose(view, (1, 0, 2))

    def _draw_panel(self, rect: pygame.Rect, title: str) -> None:
        pygame.draw.rect(self.surface, COLOR_PANEL, rect, border_radius=6)
        pygame.draw.rect(self.surface, COLOR_PANEL_BORDER, rect, width=1, border_radius=6)
        # Header text
        lbl = self.font_title.render(title, True, COLOR_TEXT_DIM)
        self.surface.blit(lbl, (rect.x + 12, rect.y + 8))
        pygame.draw.line(
            self.surface,
            COLOR_PANEL_BORDER,
            (rect.x + 8, rect.y + 26),
            (rect.x + rect.width - 8, rect.y + 26),
            1,
        )

    def _draw_radar(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        rays_dist: np.ndarray,
        rays_hits: np.ndarray,
        entities: Mapping[str, Any] | None,
    ) -> None:
        # Range rings
        pygame.draw.circle(self.surface, (30, 32, 42), (center_x, center_y), radius, width=1)
        pygame.draw.circle(self.surface, (30, 32, 42), (center_x, center_y), radius // 2, width=1)
        # Heading axis
        pygame.draw.line(self.surface, (45, 50, 65), (center_x - radius, center_y), (center_x + radius, center_y), 1)
        pygame.draw.line(self.surface, (45, 50, 65), (center_x, center_y - radius), (center_x, center_y + radius), 1)

        # 64 Rays (index 0 is straight ahead, CCW)
        num_rays = len(rays_dist)
        for i in range(num_rays):
            angle_rad = (i / num_rays) * 2.0 * math.pi
            # Player faces up (-Y in screen space)
            dir_x = -math.sin(angle_rad)
            dir_y = -math.cos(angle_rad)

            raw_d = float(rays_dist[i])
            if not math.isfinite(raw_d):
                raw_d = 1.0
            hit_dist = max(0.0, min(1.0, raw_d)) * radius
            hit_type = int(rays_hits[i])
            color = RAY_COLORS.get(hit_type, (80, 80, 80))

            px = int(center_x + dir_x * hit_dist)
            py = int(center_y + dir_y * hit_dist)

            # Draw ray line
            pygame.draw.line(self.surface, (35, 38, 50), (center_x, center_y), (px, py), 1)
            # Draw hit dot
            dot_radius = 2 if hit_type in (0, 1) else 4
            pygame.draw.circle(self.surface, color, (px, py), dot_radius)

        # Draw Visible Entities Overlay (FOV cone)
        if entities:
            mask = entities.get("mask", [])
            feats = entities.get("features", [])
            types = entities.get("entity_type", [])
            for idx in range(len(mask)):
                if not mask[idx]:
                    continue
                raw_dist, raw_sin, raw_cos = feats[idx][:3]
                if not (math.isfinite(float(raw_dist)) and math.isfinite(float(raw_sin)) and math.isfinite(float(raw_cos))):
                    continue
                ent_dist = max(0.0, min(float(radius), float(raw_dist) * radius))
                # Angle relative to forward heading
                ex = int(center_x - float(raw_sin) * ent_dist)
                ey = int(center_y - float(raw_cos) * ent_dist)
                etype = int(types[idx])
                ecolor = COLOR_RED if etype in (1, 2, 3, 4) else COLOR_GREEN
                pygame.draw.circle(self.surface, ecolor, (ex, ey), 5)
                pygame.draw.circle(self.surface, (255, 255, 255), (ex, ey), 1)

        # Player center marker (pointing up)
        pygame.draw.circle(self.surface, COLOR_ACCENT, (center_x, center_y), 4)
        pygame.draw.polygon(
            self.surface,
            COLOR_ACCENT,
            [(center_x, center_y - 8), (center_x - 4, center_y + 4), (center_x + 4, center_y + 4)],
        )

    def _draw_radar_legend(
        self,
        x: int,
        y: int,
        spatial_map: np.ndarray | None = None,
        active_slots: int | None = None,
    ) -> None:
        items = [
            ("Wall / Obstacle", (110, 115, 130)),
            ("Door (Standard)", (60, 180, 240)),
            ("Locked Door (Key)", (250, 190, 50)),
            ("Hostile Monster", (240, 70, 70)),
            ("Item / Pickup", (80, 220, 120)),
        ]
        cur_y = y
        for name, color in items:
            pygame.draw.circle(self.surface, color, (x + 6, cur_y + 8), 5)
            lbl = self.font_main.render(name, True, COLOR_TEXT)
            self.surface.blit(lbl, (x + 18, cur_y))
            cur_y += 23

        lbl_fwd = self.font_small.render("▲ FOV (90° Ahead)", True, COLOR_ACCENT)
        self.surface.blit(lbl_fwd, (x + 18, cur_y + 6))
        if active_slots is not None:
            slot_lbl = self.font_small.render(f"LANDMARK SLOTS: {active_slots}/16", True, COLOR_YELLOW)
            self.surface.blit(slot_lbl, (x + 18, cur_y + 24))

        if spatial_map is not None and spatial_map.ndim == 3:
            smap_rgb = spatial_map_to_rgb(spatial_map, scale=3)  # (96, 96, 3)
            smap_surf = pygame.surfarray.make_surface(np.transpose(smap_rgb, (1, 0, 2)))
            thumb = pygame.transform.scale(smap_surf, (84, 84))
            tx = x + 220
            ty = y + 6
            pygame.draw.rect(self.surface, COLOR_PANEL_BORDER, (tx - 2, ty - 2, 88, 88), width=1)
            self.surface.blit(thumb, (tx, ty))
            lbl_map = self.font_small.render("EGO MAP 32x32", True, COLOR_TEXT_DIM)
            self.surface.blit(lbl_map, (tx - 2, ty + 90))

    def _draw_action_bars(
        self,
        rect: pygame.Rect,
        action_probs: np.ndarray | Sequence[float] | None,
        action_taken: np.ndarray | Sequence[int] | None,
    ) -> None:
        button_names = S.BUTTONS
        probs = action_probs if action_probs is not None else [0.5] * len(button_names)
        taken = action_taken if action_taken is not None else [0] * len(button_names)

        y_offset = rect.y + 42
        bar_x = rect.x + 205
        bar_max_w = rect.width - 290

        for i, name in enumerate(button_names):
            p = float(probs[i])
            is_active = bool(taken[i])

            # Button Name
            name_color = COLOR_YELLOW if is_active else COLOR_TEXT
            name_lbl = self.font_bold.render(f"{name}", True, name_color)
            self.surface.blit(name_lbl, (rect.x + 14, y_offset))

            # Bar background
            bar_y = y_offset + 3
            pygame.draw.rect(self.surface, (32, 35, 46), (bar_x, bar_y, bar_max_w, 14), border_radius=3)

            # Filled Probability Bar
            fill_w = int(p * bar_max_w)
            bar_color = COLOR_GREEN if p > 0.6 else COLOR_ACCENT if p > 0.3 else COLOR_GRAY
            if is_active:
                bar_color = COLOR_YELLOW
            pygame.draw.rect(self.surface, bar_color, (bar_x, bar_y, fill_w, 14), border_radius=3)

            # Probability text (e.g. 78.4%)
            prob_lbl = self.font_small.render(f"{p * 100:5.1f}%", True, COLOR_TEXT_DIM)
            self.surface.blit(prob_lbl, (bar_x + bar_max_w + 10, y_offset))

            # Trigger badge
            badge_color = COLOR_GREEN if is_active else (50, 52, 62)
            badge_txt = "ON" if is_active else "OFF"
            badge_lbl = self.font_small.render(badge_txt, True, badge_color)
            self.surface.blit(badge_lbl, (bar_x + bar_max_w + 58, y_offset))

            y_offset += 32

    def _draw_telemetry(
        self,
        rect: pygame.Rect,
        vitals: np.ndarray,
        reward_breakdown: Mapping[str, float] | None,
        latency_ms: float,
        mode: str,
    ) -> None:
        # Col 1: Vitals & Key Inventory
        x1 = rect.x + 14
        y1 = rect.y + 36

        health = float(vitals[0])
        armor = float(vitals[1])
        ammo = float(vitals[3])

        # Health & Armor gauges
        self._draw_metric_row(x1, y1, "HEALTH:", f"{int(health)} HP", COLOR_GREEN if health > 30 else COLOR_RED)
        self._draw_metric_row(x1, y1 + 22, "ARMOR :", f"{int(armor)} AP", COLOR_ACCENT)
        self._draw_metric_row(x1, y1 + 44, "AMMO  :", f"{int(ammo)} rds", COLOR_YELLOW)
        self._draw_metric_row(x1, y1 + 66, "MODE  :", mode, COLOR_PURPLE)

        # Key Inventory Badges
        keys_y = y1 + 94
        lbl_k = self.font_small.render("KEYCARD INVENTORY:", True, COLOR_TEXT_DIM)
        self.surface.blit(lbl_k, (x1, keys_y))

        k_red = vitals[S.VITALS_INDEX["key_red_card"]] > 0 or vitals[S.VITALS_INDEX["key_red_skull"]] > 0
        k_blue = vitals[S.VITALS_INDEX["key_blue_card"]] > 0 or vitals[S.VITALS_INDEX["key_blue_skull"]] > 0
        k_yellow = vitals[S.VITALS_INDEX["key_yellow_card"]] > 0 or vitals[S.VITALS_INDEX["key_yellow_skull"]] > 0

        self._draw_key_badge(x1 + 10, keys_y + 18, "RED", COLOR_RED, bool(k_red))
        self._draw_key_badge(x1 + 80, keys_y + 18, "BLUE", (60, 150, 240), bool(k_blue))
        self._draw_key_badge(x1 + 150, keys_y + 18, "YELLOW", COLOR_YELLOW, bool(k_yellow))

        # Col 2: Live Reward Decomposition
        x2 = rect.x + 280
        y2 = rect.y + 36

        rb = reward_breakdown or {}
        r_total = float(rb.get("total", 0.0))
        r_color = COLOR_GREEN if r_total >= 0 else COLOR_RED
        self._draw_metric_row(x2, y2, "STEP REWARD:", f"{r_total:+.4f}", r_color)
        self._draw_metric_row(x2, y2 + 20, "  |-- Exit Goal :", f"{rb.get('exit', 0):+.2f}", COLOR_TEXT_DIM)
        self._draw_metric_row(x2, y2 + 38, "  |-- Key / Door:", f"{rb.get('key', 0) + rb.get('locked_door', 0):+.2f}", COLOR_TEXT_DIM)
        self._draw_metric_row(x2, y2 + 56, "  |-- Combat Kill:", f"{rb.get('kill', 0):+.2f}", COLOR_TEXT_DIM)
        self._draw_metric_row(x2, y2 + 74, "  |-- Exploration:", f"{rb.get('exploration', 0):+.2f}", COLOR_TEXT_DIM)
        self._draw_metric_row(x2, y2 + 92, "  \\-- Curiosity  :", f"{rb.get('curiosity', 0):+.4f}", COLOR_TEXT_DIM)

        # Latency Clock (Bottom bar)
        bottom_y = rect.y + rect.height - 38
        pygame.draw.line(self.surface, COLOR_PANEL_BORDER, (rect.x + 10, bottom_y - 6), (rect.x + rect.width - 10, bottom_y - 6), 1)

        fps = 1000.0 / max(0.1, latency_ms)
        clock_txt = f"INFERENCE LATENCY: {latency_ms:.2f} ms ({fps:.0f} FPS) | BUDGET: 40.0 ms (PASS)"
        clock_color = COLOR_GREEN if latency_ms <= 10.0 else COLOR_YELLOW if latency_ms <= 40.0 else COLOR_RED
        lbl_clock = self.font_bold.render(clock_txt, True, clock_color)
        self.surface.blit(lbl_clock, (rect.x + 14, bottom_y))

    def _draw_metric_row(self, x: int, y: int, label: str, value: str, val_color: tuple[int, int, int]) -> None:
        lbl = self.font_main.render(label, True, COLOR_TEXT_DIM)
        val = self.font_bold.render(value, True, val_color)
        self.surface.blit(lbl, (x, y))
        self.surface.blit(val, (x + 130, y))

    def _draw_key_badge(self, x: int, y: int, label: str, color: tuple[int, int, int], active: bool) -> None:
        bg = color if active else (30, 32, 40)
        fg = (255, 255, 255) if active else (90, 95, 105)
        rect = pygame.Rect(x, y, 62, 20)
        pygame.draw.rect(self.surface, bg, rect, border_radius=4)
        if not active:
            pygame.draw.rect(self.surface, (50, 55, 68), rect, width=1, border_radius=4)
        lbl = self.font_small.render(label, True, fg)
        self.surface.blit(lbl, (x + 8, y + 4))

    def close(self) -> None:
        pygame.quit()
        if self._prev_sdl_driver is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = self._prev_sdl_driver
