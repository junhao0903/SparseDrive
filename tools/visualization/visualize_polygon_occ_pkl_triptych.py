import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from shapely.geometry import Point, Polygon


LABEL_NAMES = {
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

VIVID_COLORS = [
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

BOX_CLASSES = {0, 1, 2, 3, 4, 5}


def parse_indices(indices, num_samples):
    if indices is None:
        return list(range(num_samples))
    parsed = []
    for item in indices.split(','):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0 or index >= num_samples:
            raise ValueError(f'Index out of range: {index}, num_samples={num_samples}')
        parsed.append(index)
    return parsed


def rasterize_polygons(annos, x_min, x_max, y_min, y_max, step, box_on_top=True):
    xs = np.arange(x_min + step / 2, x_max, step)
    ys = np.arange(y_min + step / 2, y_max, step)
    mask = np.full((len(xs), len(ys)), -1, dtype=np.int32)

    ordered_keys = sorted(
        annos.keys(),
        key=lambda k: (int(k) in BOX_CLASSES, int(k)) if box_on_top else int(k),
    )

    for label_key in ordered_keys:
        label = int(label_key)
        for polygon in annos[label_key]:
            poly = Polygon(np.asarray(polygon, dtype=np.float32))
            if poly.is_empty or poly.area <= 0:
                continue
            minx, miny, maxx, maxy = poly.bounds
            xi0 = max(0, int(np.floor((minx - x_min) / step)))
            xi1 = min(len(xs), int(np.ceil((maxx - x_min) / step)))
            yi0 = max(0, int(np.floor((miny - y_min) / step)))
            yi1 = min(len(ys), int(np.ceil((maxy - y_min) / step)))
            for i in range(xi0, xi1):
                for j in range(yi0, yi1):
                    p = Point(float(xs[i]), float(ys[j]))
                    if poly.contains(p) or poly.touches(p):
                        mask[i, j] = label
    return mask


def draw_axes(ax, x_min, y_min):
    ax.annotate(
        '',
        xy=(x_min + 8, y_min + 4),
        xytext=(x_min + 2, y_min + 4),
        arrowprops=dict(arrowstyle='->', color='crimson', lw=2.2),
    )
    ax.text(x_min + 8.5, y_min + 4.5, 'x+', color='crimson', fontsize=11)
    ax.annotate(
        '',
        xy=(x_min + 2, y_min + 12),
        xytext=(x_min + 2, y_min + 4),
        arrowprops=dict(arrowstyle='->', color='dodgerblue', lw=2.2),
    )
    ax.text(x_min + 2.5, y_min + 12.5, 'y+', color='dodgerblue', fontsize=11)


def draw_triptych(token, annos, out_path, x_min, x_max, y_min, y_max, step, colors, box_on_top):
    mask = rasterize_polygons(annos, x_min, x_max, y_min, y_max, step, box_on_top=box_on_top)
    masked = np.ma.masked_where(mask < 0, mask).T
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.set_xlabel('x / meter')
        ax.set_ylabel('y / meter')
        draw_axes(ax, x_min, y_min)

    axes[0].imshow(
        masked,
        origin='lower',
        extent=(x_min, x_max, y_min, y_max),
        interpolation='nearest',
        cmap=cmap,
        norm=norm,
        alpha=0.96,
    )
    axes[0].set_title('PKL-derived occ')

    used = set()
    for label_key in sorted(annos.keys(), key=lambda k: int(k)):
        label = int(label_key)
        color = colors[label % len(colors)]
        name = LABEL_NAMES.get(label, str(label))
        for polygon in annos[label_key]:
            pts = np.asarray(polygon, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                continue
            closed = np.concatenate([pts, pts[:1]], axis=0)
            axes[1].plot(
                closed[:, 0],
                closed[:, 1],
                color=color,
                linewidth=1.8,
                marker='o',
                markersize=2.4,
                label=name if name not in used else None,
            )
            used.add(name)
    axes[1].set_title('Polygon OCC')
    if used:
        handles, labels = axes[1].get_legend_handles_labels()
        by = dict(zip(labels, handles))
        axes[1].legend(by.values(), by.keys(), loc='upper right', fontsize=8, framealpha=0.92)

    axes[2].imshow(
        masked,
        origin='lower',
        extent=(x_min, x_max, y_min, y_max),
        interpolation='nearest',
        cmap=cmap,
        norm=norm,
        alpha=0.6,
    )
    for label_key in sorted(annos.keys(), key=lambda k: int(k)):
        label = int(label_key)
        color = colors[label % len(colors)]
        for polygon in annos[label_key]:
            pts = np.asarray(polygon, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                continue
            closed = np.concatenate([pts, pts[:1]], axis=0)
            axes[2].plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.8, marker='o', markersize=2.2)
    axes[2].set_title('Overlay')

    fig.suptitle(token, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize Polygon OCC PKL as occ/polygon/overlay triptych')
    parser.add_argument('--ann-file', default='data/polygon_occ_infos/polygon_occ_train.pkl')
    parser.add_argument('--out-dir', default='vis/pkl_polygon_triptych')
    parser.add_argument('--token', default=None, help='single token to visualize')
    parser.add_argument('--indices', default=None, help='comma-separated sample indices to visualize')
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument(
        '--first-n',
        type=int,
        default=None,
        help='visualize the first N frames; overrides --num-samples when set',
    )
    parser.add_argument('--x-min', type=float, default=-15.0)
    parser.add_argument('--x-max', type=float, default=15.0)
    parser.add_argument('--y-min', type=float, default=-30.0)
    parser.add_argument('--y-max', type=float, default=30.0)
    parser.add_argument('--step', type=float, default=0.4)
    parser.add_argument('--no-box-on-top', action='store_true')
    args = parser.parse_args()

    data = mmcv.load(args.ann_file)
    results = data['results'] if isinstance(data, dict) and 'results' in data else data
    tokens = list(results.keys())
    if not tokens:
        raise ValueError(f'No polygon occ annotations found in {args.ann_file}')

    selected_tokens = []
    if args.token is not None:
        if args.token not in results:
            raise ValueError(f'Token not found in ann-file: {args.token}')
        selected_tokens = [args.token]
    else:
        indices = parse_indices(args.indices, len(tokens))
        if args.indices is None:
            first_n = args.first_n if args.first_n is not None else args.num_samples
            indices = indices[:first_n]
        selected_tokens = [tokens[i] for i in indices]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, token in enumerate(selected_tokens):
        annos = results[token]
        filename = f'{idx:04d}_{token}.png' if len(selected_tokens) > 1 else f'{token}.png'
        out_path = out_dir / filename
        draw_triptych(
            token,
            annos,
            out_path,
            args.x_min,
            args.x_max,
            args.y_min,
            args.y_max,
            args.step,
            VIVID_COLORS,
            box_on_top=not args.no_box_on_top,
        )
        print(out_path)


if __name__ == '__main__':
    main()
