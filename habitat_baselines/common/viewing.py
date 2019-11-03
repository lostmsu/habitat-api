# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
To run the demo
1. Simple demo on test scenes
- Download test scenes (http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip)
  and unzip into ${HABITAT_API_REPO}/data
- Update `configs/tasks/pointnav.yaml` to have higher resolution if you want bigger pictures
- `python examples/pointgoal_demo.py --task-config configs/tasks/pointnav.yaml --overlay`
2. Simple demo on test scenes with depth
- `python examples/pointgoal_demo.py --task-config configs/tasks/pointnav_rgbd.yaml --overlay`
3. Demo on replica scene with blind agent, with saving actions and videos
- Download pretrained blind agent 
  (get blind_agent_state.pth from https://www.dropbox.com/s/e63uf6joerkf7pe/agent_demo.zip?dl=0 and put into examples/agent_demo)
- Download replica dataset (https://github.com/facebookresearch/Replica-Dataset)
  (put under data/replica)
- Generate episodes for a replica scene (this takes a while to run)
  `mkdir data/replica_demo/pointnav`
  `python examples/gen_episodes.py -o data/replica_demo/pointnav/test.json --scenes data/replica/apartment_0/habitat/mesh_semantic.ply`
  `gzip data/replica_demo/pointnav/test.json`
- Create yaml config file for replica and put in `data/replica_demo/replica_test.yaml`
  DATASET:
  TYPE: PointNav-v1
  SPLIT: test
  POINTNAVV1:
    DATA_PATH: data/replica_demo/pointnav/{split}.json.gz 
- Run demo 
  `python examples/pointgoal_demo.py --task-config configs/tasks/pointnav.yaml,data/replica_demo/replica_test.yaml --agent blind --overlay --scenes-dir . --save-video --save-actions test.json`
  NOTE: video is saved to xyz.avi if you select to replay actions (select 1/2/3 for the agent to replay)
  NOTE: actions are save to simple json file

Future improvements to demo:
1. Selection of episodes
2. Precompute episodes for replica dataset and save action trace for shortest path follower / blind agents
3. Support new episodes (random/user specified)
"""

import argparse
import json
import math
from time import sleep
from typing import List, NamedTuple, Tuple

import numpy as np

import cv2
import habitat
from habitat.core.logging import logger
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps

LINE_SPACING = 50

# TODO: Some of the functions below are potentially useful across other examples/demos
#       and should be moved to habitat/utils/visualizations

class Rect(NamedTuple):
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def center(self):
        return (
            self.left + int(self.width / 2),
            self.top + int(self.height / 2),
        )

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def write_textlines(
    output, textlines, size=1, offset=(0, 0), fontcolor=(255, 255, 255)
):
    for i, text in enumerate(textlines):
        x = offset[1]
        y = offset[0] + int((i + 1) * size * LINE_SPACING) - 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            output, text, (x, y), font, size, fontcolor, 2, cv2.LINE_AA
        )


def draw_text(textlines=[], width=300, fontsize=0.8):
    text_height = int(fontsize * LINE_SPACING * len(textlines))
    text_img = np.zeros((text_height, width, 3), np.uint8)
    write_textlines(text_img, textlines, size=fontsize)
    return text_img


def add_text(img, textlines=[], fontsize=0.8, top=False):
    combined = img
    if len(textlines) > 0:
        text_img = draw_text(textlines, img.shape[1], fontsize)
        if top:
            combined = np.vstack((text_img, img))
        else:
            combined = np.vstack((img, text_img))
    return combined


def draw_gradient_circle(img, center, size, color, bgcolor):
    ''' Draws a circle that fades from color (at the center)
        to bgcolor (at the boundaries)
    '''
    for i in range(1, size):
        a = 1 - i / size
        c = np.add(
            np.multiply(color[0:3], a), np.multiply(bgcolor[0:3], 1 - a)
        )
        cv2.circle(img, center, i, c, 2)


def draw_gradient_wedge(
    img, center, size, color, bgcolor, start_angle, delta_angle
):
    ''' Draws a wedge that fades from color (at the center)
        to bgcolor (at the boundaries)
    '''
    for i in range(1, size):
        a = 1 - i / size
        c = np.add(np.multiply(color, a), np.multiply(bgcolor, 1 - a))
        cv2.ellipse(
            img,
            center,
            (i, i),
            start_angle,
            -delta_angle / 2,
            delta_angle / 2,
            c,
            2,
        )


def draw_goal_radar(
    pointgoal,
    img,
    r: Rect,
    start_angle=0,
    fov=0,
    goalcolor=(50, 0, 184, 255),
    wincolor=(0, 0, 0, 0),
    maskcolor=(85, 75, 70, 255),
    bgcolor=(255, 255, 255, 255),
    gradientcolor=(174, 112, 80, 255),
):
    ''' Draws a radar that shows the goal as a dot
    '''
    angle = pointgoal[1]  # angle
    mag = pointgoal[0]    # magnitude (>=0)
    nm = mag / (mag + 1)  # normalized magnitude (0 to 1)
    xy = (-math.sin(angle), -math.cos(angle))
    size = int(round(0.45 * min(r.width, r.height)))
    center = r.center
    target = (
        int(round(center[0] + xy[0] * size * nm)),
        int(round(center[1] + xy[1] * size * nm)),
    )
    if wincolor is not None:
        cv2.rectangle(
            img, (r.left, r.top), (r.right, r.bottom), wincolor, -1
        )  # Fill with window color
    cv2.circle(img, center, size, bgcolor, -1)  # Circle with background color
    if fov > 0:
        masked = 360 - fov
        cv2.ellipse(
            img,
            center,
            (size, size),
            start_angle + 90,
            -masked / 2,
            masked / 2,
            maskcolor,
            -1,
        )
    if gradientcolor is not None:
        if fov > 0:
            draw_gradient_wedge(
                img,
                center,
                size,
                gradientcolor,
                bgcolor,
                start_angle - 90,
                fov,
            )
        else:
            draw_gradient_circle(img, center, size, gradientcolor, bgcolor)
    cv2.circle(img, target, 4, goalcolor, -1)


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(info["top_down_map"]["map"])
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


class Viewer:
    def __init__(
        self,
        initial_observations,
        pointgoal_name="pointgoal",
        overlay_goal_radar=None,
        goal_display_size=128,
        show_map=False,
    ):
        self.overlay_goal_radar = overlay_goal_radar
        self.show_map = show_map
        self._pointgoal_name = pointgoal_name

        # What image sensors are active
        all_image_sensors = ["rgb", "depth"]
        self.active_image_sensors = [
            s for s in all_image_sensors if s in initial_observations
        ]
        total_width = 0
        total_height = 0
        for s in self.active_image_sensors:
            total_width += initial_observations[s].shape[1]
            total_height = max(initial_observations[s].shape[0], total_height)

        self.draw_info = {}
        if self.overlay_goal_radar:
            img = np.zeros((goal_display_size, goal_display_size, 4), np.uint8)
            self.draw_info[self._pointgoal_name] = {
                "image": img,
                "region": Rect(0, 0, goal_display_size, goal_display_size),
            }
        else:
            side_img_height = max(total_height, goal_display_size)
            self.side_img = np.zeros(
                (side_img_height, goal_display_size, 3), np.uint8
            )
            self.draw_info[self._pointgoal_name] = {
                "image": self.side_img,
                "region": Rect(0, 0, goal_display_size, goal_display_size),
            }
            total_width += goal_display_size
        self.window_size = (total_height, total_width)

    def draw_observations(self, observations, info=None):
        active_image_observations = [
            observations[s] for s in self.active_image_sensors
        ]
        for i, img in enumerate(active_image_observations):
            if img.shape[2] == 1:
                img *= 255.0 / img.max()  # naive rescaling for visualization
                active_image_observations[i] = cv2.cvtColor(
                    img, cv2.COLOR_GRAY2BGR
                ).astype(np.uint8)
            elif img.shape[2] == 3:
                active_image_observations[i] = transform_rgb_bgr(img)

        # draw pointgoal
        goal_draw_surface = self.draw_info[self._pointgoal_name]
        # TODO: get fov from agent
        draw_goal_radar(
            observations[self._pointgoal_name],
            goal_draw_surface["image"],
            goal_draw_surface["region"],
            start_angle=0,
            fov=90,
        )
        if self.overlay_goal_radar:
            goal_region = goal_draw_surface["region"]
            bottom = self.window_size[0]
            top = bottom - goal_region.height
            left = self.window_size[1] // 2 - goal_region.width // 2
            right = left + goal_region.width
            stacked = np.hstack(active_image_observations)
            alpha = 0.5 * (goal_draw_surface["image"][:, :, 3] / 255)
            rgb = goal_draw_surface["image"][:, :, 0:3]
            overlay = np.add(
                np.multiply(
                    stacked[top:bottom, left:right],
                    np.expand_dims(1 - alpha, axis=2),
                ),
                np.multiply(rgb, np.expand_dims(alpha, axis=2)),
            )
            stacked[top:bottom, left:right] = overlay
        else:
            stacked = np.hstack(active_image_observations + [self.side_img])
        if info is not None:
            if (
                self.show_map
                and info.get("top_down_map") is not None
                and "heading" in observations
            ):
                top_down_map = draw_top_down_map(
                    info, observations["heading"], stacked.shape[0]
                )
                stacked = np.hstack((top_down_map, stacked))
        return stacked
