import os
import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import Point, Polygon

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
 
CMD_LIST = ['Turn Right', 'Turn Left', 'Go Straight']
COLOR_VECTORS = ['cornflowerblue', 'royalblue', 'slategrey']
SCORE_THRESH = 0.3
MAP_SCORE_THRESH = 0.3
OCC_GRID_STEP = 0.4
OCC_CLASS_NAMES = {
    0: 'barrier',
    1: 'car',
    2: 'large_vehicle',
    3: 'two_wheeler',
    4: 'pedestrian',
    5: 'traffic_cone',
    6: 'driveable_surface',
    7: 'sidewalk',
    8: 'terrain',
    9: 'manmade',
    10: 'vegetation',
}
OCC_COLORS = [
    '#ff0000',  # barrier
    '#0066ff',  # car
    '#ff8c00',  # large_vehicle
    '#8a2be2',  # two_wheeler
    '#ff1493',  # pedestrian
    '#ffd700',  # traffic_cone
    '#00aa00',  # driveable_surface
    '#00cfff',  # sidewalk
    '#a0522d',  # terrain
    '#808080',  # manmade
    '#9acd32',  # vegetation
]
OCC_BOX_CLASSES = {0, 1, 2, 3, 4, 5}
color_mapping = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
]) / 255


class BEVRender:
    def __init__(
        self, 
        plot_choices,
        out_dir,
        xlim = 40,
        ylim = 40,
    ):
        self.plot_choices = plot_choices
        self.xlim = xlim
        self.ylim = ylim
        self.gt_dir = os.path.join(out_dir, "bev_gt")
        self.pred_dir = os.path.join(out_dir, "bev_pred")
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

    def reset_canvas(self):
        plt.close()
        self.fig, self.axes = plt.subplots(1, 1, figsize=(20, 20))
        self.axes.set_xlim(- self.xlim, self.xlim)
        self.axes.set_ylim(- self.ylim, self.ylim)
        self.axes.axis('off')

    def render(
        self,
        data, 
        result,
        index,
    ):
        self.reset_canvas()
        self.draw_detection_gt(data)
        self.draw_motion_gt(data)
        self.draw_map_gt(data)
        self.draw_occ_gt(data)
        self.draw_planning_gt(data)
        self._render_sdc_car()
        self._render_command(data)
        self._render_legend()
        save_path_gt = os.path.join(self.gt_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path_gt)

        self.reset_canvas()
        self.draw_detection_pred(result)
        self.draw_track_pred(result)
        self.draw_motion_pred(result)
        self.draw_map_pred(result)
        self.draw_occ_pred(result)
        self.draw_planning_pred(data, result)
        self._render_sdc_car()
        self._render_command(data)
        self._render_legend()
        save_path_pred = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path_pred)

        return save_path_gt, save_path_pred

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(filename)

    def draw_detection_gt(self, data):
        if not self.plot_choices['det']:
            return

        for i in range(data['gt_labels_3d'].shape[0]):
            label = data['gt_labels_3d'][i]
            if label == -1: 
                continue
            color = color_mapping[i % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(data['gt_bboxes_3d'])[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_detection_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[result['instance_ids'][i] % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(bboxes)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_track_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['track'] and "anchor_queue" in result):
            return
        
        temp_bboxes = result["anchor_queue"]
        period = result["period"]
        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
            center = bboxes[i, :3]
            centers = [center]
            for j in range(period[i]):
                # draw corners
                corners = box3d_to_corners(temp_bboxes[:, -1-j])[i, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')
                centers.append(center)

            centers = np.stack(centers)
            xs = centers[:, 0]
            ys = centers[:, 1]
            self.axes.plot(xs, ys, color=color, linewidth=2, linestyle='-')

    def draw_motion_gt(self, data):
        if not self.plot_choices['motion']:
            return

        for i in range(data['gt_labels_3d'].shape[0]):
            label = data['gt_labels_3d'][i]
            if label == -1: 
                continue
            color = color_mapping[i % len(color_mapping)]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            center = data['gt_bboxes_3d'][i, :2]
            masks = data['gt_agent_fut_masks'][i].astype(bool)
            if masks[0] == 0:
                continue
            trajs = data['gt_agent_fut_trajs'][i][masks]
            trajs = trajs.cumsum(axis=0) + center
            trajs = np.concatenate([center.reshape(1, 2), trajs], axis=0)
            
            self._render_traj(trajs, traj_score=1.0,
                            colormap='winter', dot_size=dot_size)

    def draw_motion_pred(self, result, top_k=3):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['motion'] and "trajs_3d" in result):
            return
        
        bboxes = result['boxes_3d']
        labels = result['labels_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            label = labels[i]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            traj_score = result['trajs_score'][i].numpy()
            traj = result['trajs_3d'][i].numpy()
            num_modes = len(traj_score)
            center = bboxes[i, :2][None, None].repeat(num_modes, 1, 1).numpy()
            traj = np.concatenate([center, traj], axis=1)

            sorted_ind = np.argsort(traj_score)[::-1]
            sorted_traj = traj[sorted_ind, :, :2]
            sorted_score = traj_score[sorted_ind]
            norm_score = np.exp(sorted_score[0])

            for j in range(top_k - 1, -1, -1):
                viz_traj = sorted_traj[j]
                traj_score = np.exp(sorted_score[j])/norm_score
                self._render_traj(viz_traj, traj_score=traj_score,
                                colormap='winter', dot_size=dot_size)
    
    def draw_map_gt(self, data):
        if not self.plot_choices['map']:
            return
        vectors = data['map_infos']
        for label, vector_list in vectors.items():
            color = COLOR_VECTORS[label]
            for vector in vector_list:
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                self.axes.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_map_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['map'] and "vectors" in result):
            return

        for i in range(result['scores'].shape[0]):
            score = result['scores'][i]
            if  score < MAP_SCORE_THRESH:
                continue
            color = COLOR_VECTORS[result['labels'][i]]
            pts = result['vectors'][i]
            x = pts[:, 0]
            y = pts[:, 1]
            plt.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_occ_gt(self, data):
        if not self.plot_choices.get('occ', False):
            return
        polygon_geoms = data.get('polygon_occ_geoms')
        if polygon_geoms is None:
            return
        annos = self._normalize_occ_annos(polygon_geoms)
        self._draw_occ_raster(annos, alpha=0.72)
        self._draw_occ_outlines(annos, linestyle='--')

    def draw_occ_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices.get('occ', False)):
            return
        occ_result = result.get('occ', result)
        if 'polygons' not in occ_result:
            return
        annos = self._normalize_occ_pred_annos(occ_result)
        self._draw_occ_raster(annos, alpha=0.85)
        self._draw_occ_outlines(annos, linestyle='-')

    def draw_planning_gt(self, data):
        if not self.plot_choices['planning']:
            return

        # draw planning gt
        masks = data['gt_ego_fut_masks'].astype(bool)
        if masks[0] != 0:
            plan_traj = data['gt_ego_fut_trajs'][masks]
            cmd = data['gt_ego_fut_cmd']
            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)
            plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
            self._render_traj(plan_traj, traj_score=1.0,
                colormap='autumn', dot_size=50)

    def draw_planning_pred(self, data, result, top_k=3):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['planning'] and "planning" in result):
            return

        if self.plot_choices['track'] and "ego_anchor_queue" in result:
            ego_temp_bboxes = result["ego_anchor_queue"]
            ego_period = result["ego_period"]
            for j in range(ego_period[0]):
                # draw corners
                corners = box3d_to_corners(ego_temp_bboxes[:, -1-j])[0, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')
        # import ipdb; ipdb.set_trace()
        plan_trajs = result['planning'].cpu().numpy()
        num_cmd = len(CMD_LIST)
        num_mode = plan_trajs.shape[1]
        plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        plan_score = result['planning_score'].cpu().numpy()

        cmd = data['gt_ego_fut_cmd'].argmax()
        plan_trajs = plan_trajs[cmd]
        plan_score = plan_score[cmd]

        sorted_ind = np.argsort(plan_score)[::-1]
        sorted_traj = plan_trajs[sorted_ind, :, :2]
        sorted_score = plan_score[sorted_ind]
        norm_score = np.exp(sorted_score[0])

        for j in range(top_k - 1, -1, -1):
            viz_traj = sorted_traj[j]
            traj_score = np.exp(sorted_score[j]) / norm_score
            self._render_traj(viz_traj, traj_score=traj_score,
                            colormap='autumn', dot_size=50)

    def _render_traj(
        self, 
        future_traj, 
        traj_score=1, 
        colormap='winter', 
        points_per_step=20, 
        dot_size=25
    ):
        total_steps = (len(future_traj) - 1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps))[:, :3]
        dot_colors = dot_colors * traj_score + \
            (1 - traj_score) * np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps - 1):
            unit_vec = future_traj[i // points_per_step +
                                   1] - future_traj[i // points_per_step]
            total_xy[i] = (i / points_per_step - i // points_per_step) * \
                unit_vec + future_traj[i // points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(
            total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)

    def _render_sdc_car(self):
        sdc_car_png = cv2.imread('resources/sdc_car.png')
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        im = self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))
        im.set_zorder(2)

    def _render_legend(self):
        if self.plot_choices.get('occ', False):
            handles = [
                Patch(facecolor=OCC_COLORS[idx], edgecolor='black', label=OCC_CLASS_NAMES[idx])
                for idx in sorted(OCC_CLASS_NAMES.keys())
            ]
            legend = self.axes.legend(
                handles=handles,
                loc='upper right',
                bbox_to_anchor=(0.985, 0.985),
                fontsize=10,
                framealpha=0.95,
                facecolor='white',
                edgecolor='black',
                ncol=1,
            )
            legend.set_zorder(20)
            return
        legend = cv2.imread('resources/legend.png')
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(15, 40, -40, -30))

    def _render_command(self, data):
        cmd = data['gt_ego_fut_cmd'].argmax()
        self.axes.text(-38, -38, CMD_LIST[cmd], fontsize=60)

    def _normalize_occ_annos(self, polygon_geoms):
        annos = {}
        for label, polygon_list in polygon_geoms.items():
            label_int = int(label)
            for polygon in polygon_list:
                pts = self._polygon_to_coords(polygon)
                if pts is None:
                    continue
                annos.setdefault(label_int, []).append(pts)
        return annos

    def _normalize_occ_pred_annos(self, occ_result):
        annos = {}
        for polygon, label, score in zip(
            occ_result['polygons'], occ_result['labels'], occ_result['scores']
        ):
            if float(score) < MAP_SCORE_THRESH:
                continue
            pts = self._polygon_to_coords(polygon)
            if pts is None:
                continue
            annos.setdefault(int(label), []).append(pts)
        return annos

    def _polygon_to_coords(self, polygon):
        if hasattr(polygon, 'exterior'):
            pts = np.asarray(polygon.exterior.coords, dtype=np.float32)
        else:
            pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            return None
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) < 3:
            return None
        return pts

    def _ordered_occ_labels(self, annos):
        return sorted(
            annos.keys(),
            key=lambda label: (label in OCC_BOX_CLASSES, label),
        )

    def _rasterize_occ_polygons(self, annos):
        xs = np.arange(-self.xlim + OCC_GRID_STEP / 2, self.xlim, OCC_GRID_STEP)
        ys = np.arange(-self.ylim + OCC_GRID_STEP / 2, self.ylim, OCC_GRID_STEP)
        mask = np.full((len(xs), len(ys)), -1, dtype=np.int32)

        for label in self._ordered_occ_labels(annos):
            for polygon in annos[label]:
                poly = Polygon(np.asarray(polygon, dtype=np.float32))
                if poly.is_empty or poly.area <= 0:
                    continue
                minx, miny, maxx, maxy = poly.bounds
                xi0 = max(0, int(np.floor((minx + self.xlim) / OCC_GRID_STEP)))
                xi1 = min(len(xs), int(np.ceil((maxx + self.xlim) / OCC_GRID_STEP)))
                yi0 = max(0, int(np.floor((miny + self.ylim) / OCC_GRID_STEP)))
                yi1 = min(len(ys), int(np.ceil((maxy + self.ylim) / OCC_GRID_STEP)))
                for i in range(xi0, xi1):
                    for j in range(yi0, yi1):
                        point = Point(float(xs[i]), float(ys[j]))
                        if poly.contains(point) or poly.touches(point):
                            mask[i, j] = label
        return mask

    def _draw_occ_raster(self, annos, alpha):
        if not annos:
            return
        mask = self._rasterize_occ_polygons(annos)
        masked = np.ma.masked_where(mask < 0, mask).T
        cmap = mcolors.ListedColormap(OCC_COLORS)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, len(OCC_COLORS) + 0.5), cmap.N)
        self.axes.imshow(
            masked,
            origin='lower',
            extent=(-self.xlim, self.xlim, -self.ylim, self.ylim),
            interpolation='nearest',
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            zorder=0,
        )

    def _draw_occ_outlines(self, annos, linestyle):
        for label in self._ordered_occ_labels(annos):
            color = OCC_COLORS[label % len(OCC_COLORS)]
            for polygon in annos[label]:
                pts = np.asarray(polygon, dtype=np.float32)
                closed = np.concatenate([pts, pts[:1]], axis=0)
                self.axes.plot(
                    closed[:, 0],
                    closed[:, 1],
                    color=color,
                    linewidth=2,
                    linestyle=linestyle,
                    zorder=5,
                )
