import cv2
from collections import OrderedDict
import numpy as np
from envs import game
import matplotlib.pyplot as plt

class VisMap:
    def __init__(self, image, cmap_name='viridis'):
        self.img = image
        self.cmapname = cmap_name
        self.cmap = None
        self.norm = None

    def calculate_heatmap(self, img):
        self.cmap = plt.cm.get_cmap(self.cmapname)
        self.norm = plt.Normalize(vmin=img.min(), vmax=img.max())

    def heat_image(self):
        if self.cmap is None or self.norm  is None:
            self.calculate_heatmap(self.img)
        return self.cmap(self.norm(self.img))

    def reverse_heat(self, image):
        return  self.cmap.reverse(self.norm.inverse(image))



class Visualisation:
    def __init__(self, game: game.Game2D):
        self.game = game

    @property
    def heatmaps(self):
        showmaps = [('source', self._heat_source_map),
        ('target', self._heat_target_map),
        ('result', self._heat_result_map)]
        return OrderedDict(showmaps)

    @property
    def window_positions(self):
        win_poses = [
            ('source', (25, 300)),
            ('target', (450, 300)),
            ('result', (900,300)),
            ('source_cursor', (25, 100)),
            ('target_cursor', (450, 100)),
            ('result_cursor', (900, 100))
        ]
        return OrderedDict(win_poses)

    @heatmaps.setter
    def heatmaps(self, val):
        assert len(val) == 3, "setting too long"
        self._heat_source_map = val[0]
        self._heat_target_map = val[1]
        self._heat_result_map = val[2]

    def _update_maps(self):
        self._source_img = self.game.env.source_map.copy()
        self._target_img = self.game.env.target_map.copy()
        _result_img = self.game.env.result_map.copy()
        self._result_img = np.argmax(_result_img, axis=2)
        self._vis_source_map = VisMap(self._source_img)
        self._vis_target_map = VisMap(self._target_img)
        self._vis_result_map = VisMap(self._result_img)
        self._heat_source_map = self._vis_source_map.heat_image()
        self._heat_target_map = self._vis_target_map.heat_image()
        self._heat_result_map = self._vis_result_map.heat_image()

    def add_cursor_to_maps(self):
        for _, hmap in self.heatmaps.items():
            self.game.cursor.set_range(hmap, np.array([1,0,0, 1.]), region_type='source_input')

    def show_cursor(self, name, data):
        cv2.imshow(name, data)

    def extract_cursor_data(self):
        for name, data in self.heatmaps.items():
            cursor_data = self.game.cursor.get_range(data)
            name = '{}_cursor'.format(name)
            cv2.imshow(name,cursor_data)
            cv2.moveWindow(name, *self.window_positions[name])


    def update(self, wait=0):
        self._update_maps()
        self.extract_cursor_data()
        self.add_cursor_to_maps()
        for name, map in self.heatmaps.items():
            map = cv2.resize(map, (400,400))
            cv2.imshow(name, map)

            cv2.moveWindow(name, *self.window_positions[name])
        cv2.waitKey(wait)
