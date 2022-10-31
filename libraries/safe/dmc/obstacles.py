"""Walled room with central constraint"""

from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
import os

import numpy as np


_GROUNDPLANE_QUAD_SIZE = 0.25
_TOP_CAMERA_DISTANCE = 7.0
_TOP_CAMERA_Y_PADDING_FACTOR = 2.5


class Obstacle(composer.Arena):

    def _build(self, size=(11, 11), name='obstacle'):
        super()._build(name=name)

        self._aesthetic = 'outdoor_natural'
        self.size = size

        sky_info = locomotion_arenas_assets.get_sky_texture_info(self._aesthetic)

        self._ground_texture = self._mjcf_root.asset.add(
            'texture',
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            type='2d',
            builtin='checker',
            name='groundplane',
            width=200,
            height=200,
            mark='edge',
            markrgb=[0.8, 0.8, 0.8]
        )

        self._ground_material = self._mjcf_root.asset.add(
            'material',
            name='groundplane',
            texrepeat=[2, 2],
            texuniform=True,
            reflectance=0.2,
            texture=self._ground_texture
        )

        self._ground_geom = self._mjcf_root.worldbody.add(
            'geom',
            type='plane',
            name='groundplane',
            material=self._ground_material,
            # rgba=[1, 1, 1, 1],
            size = list(size) + [_GROUNDPLANE_QUAD_SIZE],
            pos = [0, 0, 0]
        )

        #TODO: Work out why the ground doesn't render correctly 
        ground_size = max(size)
        top_camera_fovy = (360 / np.pi) * np.arctan2(
            _TOP_CAMERA_Y_PADDING_FACTOR * ground_size / 2,
            _TOP_CAMERA_DISTANCE)
        self._top_camera = self._mjcf_root.worldbody.add(
            'camera',
            name='top_camera',
            pos=[0, 0, _TOP_CAMERA_DISTANCE],
            zaxis=[0, 0, 1],
            fovy=top_camera_fovy
        )
        
        self._obstacle = self._mjcf_root.worldbody.add(
            'geom',
            type='box',
            name='box_obstacle',
            size=[2.0, 2.0, 0.5],
            rgba=[1, 0, 0, 1],
            pos = [0.5, 0, 0]
        )

        self._wall_height = 1.0
        self._wall_thickness = 0.5
        self._wall_colour = [0, 0, 0, 0.0]
        self._wall_left = self._mjcf_root.worldbody.add(
            'geom',
            type='box',
            name='wall_left',
            size=[self._wall_thickness, size[1], self._wall_height],
            rgba=self._wall_colour,
            pos=[size[0], 0.0, 0.0]
        )
        self._wall_right = self._mjcf_root.worldbody.add(
            'geom',
            type='box',
            name='wall_right',
            size=[self._wall_thickness, size[1], self._wall_height],
            rgba=self._wall_colour,
            pos=[-size[0], 0.0, 0.0]
        )
        self._wall_front = self._mjcf_root.worldbody.add(
            'geom',
            type='box',
            name='wall_front',
            size=[size[0], self._wall_thickness, self._wall_height],
            rgba=self._wall_colour,
            pos=[0.0, size[1], 0.0]
        )
        self._wall_back = self._mjcf_root.worldbody.add(
            'geom',
            type='box',
            name='wall_back',
            size=[size[0], self._wall_thickness, self._wall_height],
            rgba=self._wall_colour,
            pos=[0.0, -size[1], 0.0]
        )

        self._objective = self._mjcf_root.worldbody.add(
            'geom',
            type='sphere',
            name='goal_position',
            size=[0.7],
            rgba=[0, 0, 1, 1],
            pos=[-5.0, 0.0, 0.0]
        )

    def _build_observables(self):
        return ObstacleObservables(self)

    @property
    def ground_geoms(self):
        return (self._ground_geom,)

    @property
    def top_camera(self):
        return self._top_camera

    def regenerate(self, random_state):
        pass


class ObstacleObservables(composer.Observables):

    @composer.observable
    def top_camera(self):
        return observable.MJCFCamera(self._entity.top_camera)
