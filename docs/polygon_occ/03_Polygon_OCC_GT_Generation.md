# 03_Polygon_OCC_GT_Generation

## 1. Purpose of This Document

This document defines the **ground-truth generation pipeline** for Polygon OCC V1.

The goal is to convert nuScenes-based semantic occupancy annotations into fixed-length semantic polygon targets compatible with the SparseDrive map branch.

The intended output is conceptually:

```text
gt_polygon_pts
gt_polygon_labels
```

with:

```text
gt_polygon_pts:
    N × 32 × 2

gt_polygon_labels:
    N
```

where:

- `N` is the number of semantic polygon regions in a sample.
- each polygon has exactly 32 ordered vertices.
- each polygon represents one semantic region instance.
- coordinates are expressed in the same local BEV frame expected by the SparseDrive map branch.

This document describes **design intent and processing rules**.

Claude Code must inspect the exact SparseDrive repository and occupancy annotation format before choosing file names, registry names, or storage structures.

---

## 2. High-Level GT Pipeline

The complete Polygon OCC GT generation pipeline is:

```text
Semantic Occupancy GT
        ↓
Sample Alignment
        ↓
Occupancy Grid Decoding
        ↓
Coordinate / ROI Verification
        ↓
Per-Class BEV Projection
        ↓
Binary Semantic Masks
        ↓
Optional Morphology
        ↓
Connected Component Extraction
        ↓
Contour Extraction
        ↓
Contour Filtering
        ↓
Polygon Simplification
        ↓
Metric Coordinate Conversion
        ↓
Uniform Perimeter Sampling
        ↓
32-Point Polygon
        ↓
Orientation Normalization
        ↓
Starting-Vertex Normalization
        ↓
Validity Check
        ↓
Polygon GT
```

Every stage should be independently testable.

Do not implement the whole conversion as one monolithic function.

---

## 3. Preferred Data Source

The preferred source is semantic occupancy annotation aligned with nuScenes.

Candidate sources include:

```text
Occ3D-nuScenes
OpenOccupancy / OpenOcc-style annotations
```

The implementation must verify the actual local dataset.

Before writing conversion logic, inspect:

```text
annotation path
file format
sample token mapping
semantic label IDs
occupancy tensor shape
voxel size
point-cloud range / voxel range
visibility mask
unknown/free labels
coordinate convention
z-axis ordering
```

Do not assume that all nuScenes occupancy datasets use identical formats.

The exact annotation source must be treated as an implementation dependency.

---

## 4. Why Use Occupancy GT Instead of HD Map Only

SparseDrive's original map GT is derived from vector map geometry.

That is appropriate for:

```text
divider
boundary
ped_crossing
```

Polygon OCC targets broader semantic regions.

Examples:

```text
drivable_surface
sidewalk
terrain
vegetation
vehicle
pedestrian
barrier
```

A semantic occupancy source provides a common semantic field from which these region classes can be derived.

The GT conversion becomes:

```text
Dense Semantic Field
        ↓
Sparse Semantic Regions
```

This is the fundamental vectorization process behind Polygon OCC.

---

## 5. Sample Alignment

Sample alignment is a hard requirement.

The Polygon GT must correspond to the exact SparseDrive sample.

Do not align data using filename order alone.

Preferred identifiers include:

```text
sample_token
scene_token
timestamp
lidar sample-data token
```

The exact mapping depends on the occupancy dataset.

The conversion pipeline must verify:

```text
SparseDrive sample
↔
nuScenes sample
↔
occupancy annotation
```

Recommended metadata saved during preprocessing:

```python
{
    "sample_token": "...",
    "scene_token": "...",
    "timestamp": ...,
    "occ_source": "...",
}
```

This makes offline debugging easier.

---

## 6. Occupancy Tensor Inspection

Before processing, print or log:

```text
occupancy shape
dtype
min label
max label
unique labels
voxel size
spatial range
```

For example:

```text
shape: [X, Y, Z]
dtype: uint8
unique labels: [0, 1, 2, ...]
```

The actual dimension order must be verified.

Do not assume:

```text
[X, Y, Z]
```

It may be stored as:

```text
[Z, Y, X]
[Y, X, Z]
```

or flattened.

The conversion must explicitly document the decoded axis order.

---

## 7. Semantic Label Mapping

Create one centralized semantic mapping.

Conceptual example:

```python
OCC_CLASS_TO_POLYGON_CLASS = {
    OCC_DRIVABLE_SURFACE: POLY_DRIVABLE_SURFACE,
    OCC_SIDEWALK: POLY_SIDEWALK,
    OCC_TERRAIN: POLY_TERRAIN,
    OCC_VEGETATION: POLY_VEGETATION,
    OCC_CAR: POLY_VEHICLE,
    OCC_TRUCK: POLY_VEHICLE,
    OCC_BUS: POLY_VEHICLE,
    OCC_PEDESTRIAN: POLY_PEDESTRIAN,
    OCC_BARRIER: POLY_BARRIER,
}
```

This is only a conceptual example.

The actual label IDs must be read from the occupancy annotation definitions.

Do not scatter class remapping logic across multiple functions.

Recommended design:

```python
POLYGON_CLASSES = (...)
OCC_TO_POLYGON = {...}
```

stored in one configuration or one dataset-constant module.

---

## 8. V1 Class Selection

Do not immediately include all occupancy classes.

The initial class set should prioritize semantic regions that are meaningful in 2D BEV.

Recommended class categories to investigate:

```text
drivable_surface
sidewalk
terrain
vegetation
vehicle
pedestrian
barrier
```

The final V1 class set should be selected using dataset statistics.

For every candidate class, collect:

```text
frames containing class
mean occupied BEV area
mean connected components per frame
P95 connected components per frame
mean contour complexity
mean component area
fraction of tiny components
```

Classes with extreme fragmentation may be postponed.

---

## 9. 3D Occupancy to Per-Class BEV Mask

The preferred V1 design uses **per-class independent masks**.

For each Polygon OCC class `c`:

```text
occ_3d
   ↓
select voxels mapped to class c
   ↓
reduce along z
   ↓
mask_c(x, y)
```

Conceptually:

```python
mask_c = np.any(class_voxels, axis=z_axis)
```

This means:

```text
mask_c(x, y) = True
```

if at least one voxel along Z belongs to class `c`.

---

## 10. Why Per-Class Independent Masks

Independent masks preserve overlapping semantic regions.

Example:

```text
road
+
vehicle
```

At the same BEV location:

```text
road_mask(x, y) = 1
vehicle_mask(x, y) = 1
```

This is valid for Polygon OCC.

The final representation can contain:

```text
road polygon
vehicle polygon
```

with geometric overlap.

This is preferable to forcing a single exclusive BEV class label.

---

## 11. Z-Range Filtering

A naive `any along z` projection may include undesirable voxels.

Therefore, the projection function should support configurable Z ranges.

Conceptual interface:

```python
project_class_to_bev(
    occ,
    class_ids,
    z_min=None,
    z_max=None,
)
```

Possible reasons for Z filtering:

```text
ignore high vegetation canopy
ignore overhead structures
restrict to near-ground drivable surface
remove low-confidence vertical noise
```

V1 should not invent class-specific Z rules before inspecting data.

However, the GT code should make Z filtering configurable.

---

## 12. Visibility and Validity Masks

Some occupancy datasets provide:

```text
camera visibility mask
lidar visibility mask
validity mask
unknown mask
```

The converter must inspect these fields.

A semantic voxel should not automatically be used if the dataset marks it invalid.

Possible filtering logic:

```python
valid = semantic_valid_mask
class_voxels = (occ == class_id) & valid
```

The exact policy must match the selected dataset's annotation semantics.

Do not treat `unknown` as a semantic polygon class by default.

---

## 13. BEV Mask Coordinate Convention

A BEV mask index must be convertible to metric coordinates.

Suppose the occupancy range is:

```text
x ∈ [x_min, x_max]
y ∈ [y_min, y_max]
```

with voxel size:

```text
vx
vy
```

A BEV cell `(ix, iy)` may correspond to:

```text
x = x_min + (ix + 0.5) * vx
y = y_min + (iy + 0.5) * vy
```

The exact axis ordering depends on the annotation format.

This mapping must be verified against known geometry.

Required visualization checks:

```text
front is correct
left/right is correct
ego origin is correct
map ROI matches SparseDrive
```

Never assume image row/column directions directly equal BEV x/y.

---

## 14. Recommended Coordinate Processing Order

There are two possible strategies.

### Strategy A: Extract Contours in Pixel Space, Then Convert to Metric

```text
binary mask
    ↓
findContours
    ↓
pixel contour
    ↓
metric coordinate conversion
```

### Strategy B: Convert Occupied Cells to Metric First, Then Build Geometry

```text
occupied cells
    ↓
metric points
    ↓
geometry construction
```

V1 should prefer Strategy A.

Reasons:

- OpenCV contour extraction works naturally in image space.
- connected-component and contour algorithms are efficient.
- polygon topology is simpler in raster coordinates.

After contour extraction, convert contour points to metric BEV coordinates.

---

## 15. Binary Mask Preparation

For each class:

```python
mask = class_mask.astype(np.uint8)
```

Expected values:

```text
0
1
```

For OpenCV, it is often convenient to use:

```text
0
255
```

Conceptually:

```python
mask_u8 = mask.astype(np.uint8) * 255
```

The code should explicitly control dtype.

Avoid relying on implicit conversion from boolean or int64 arrays.

---

## 16. Optional Morphological Processing

Occupancy masks may contain:

```text
small holes
isolated pixels
voxel staircase
tiny gaps
```

Morphology may improve contour quality.

Potential operations:

```text
opening
closing
```

### Closing

Useful for:

```text
small internal holes
small gaps
```

Conceptually:

```python
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

### Opening

Useful for:

```text
isolated noise
thin protrusions
```

Conceptually:

```python
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
```

---

## 17. Morphology Policy for V1

Morphology should be optional and configurable.

Recommended default development sequence:

```text
1. visualize raw masks
2. extract raw contours
3. inspect contour noise
4. add morphology only if necessary
```

Do not automatically apply aggressive smoothing.

Morphology changes GT geometry.

Every kernel size must be interpreted in terms of BEV resolution.

For example:

```text
3 × 3 pixels
```

has a different metric meaning at:

```text
0.2 m / pixel
```

than at:

```text
0.5 m / pixel
```

The code should log the effective metric kernel size.

---

## 18. Connected Component Extraction

For each class-specific binary mask:

```text
mask
  ↓
connected components
```

Each component is an initial semantic region instance.

Possible implementation:

```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    mask,
    connectivity=8,
)
```

Recommended connectivity:

```text
8-connectivity
```

for V1.

The reason is that diagonal semantic cells often belong to the same physical region.

---

## 19. Component Filtering

Not every connected component should become a GT polygon.

Tiny components may represent:

```text
noise
single occupied voxel
annotation artifact
fragmented semantic speckle
```

Filter using configurable thresholds.

Recommended filters:

```text
min_component_pixels
min_component_area_m2
```

Metric area is preferable.

Conceptually:

```python
area_m2 = num_pixels * voxel_size_x * voxel_size_y
```

Then:

```python
if area_m2 < min_area_m2:
    discard
```

Class-specific area thresholds may be required later.

V1 should begin with a global threshold unless statistics strongly justify class-specific thresholds.

---

## 20. Contour Extraction

After component filtering, extract contours.

Recommended OpenCV interface:

```python
contours, hierarchy = cv2.findContours(
    component_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE,
)
```

### Why `RETR_EXTERNAL`

V1 represents each semantic region using one outer polygon.

Internal holes are not modeled explicitly.

Therefore:

```text
external contour only
```

is the simplest design.

### Why `CHAIN_APPROX_NONE`

The raw contour should initially preserve all boundary points.

Simplification is handled explicitly later.

This makes the GT pipeline easier to reason about.

---

## 21. Hole Handling

A semantic region may contain holes.

Example:

```text
###########
###     ###
###     ###
###########
```

A fully general polygon representation may require:

```text
outer ring
+
inner rings
```

Polygon OCC V1 does not support inner rings.

Therefore, V1 uses the external contour only.

Conceptually, the hole becomes filled.

This is an accepted approximation.

Record this design limitation in evaluation.

Do not silently introduce variable numbers of inner contours.

---

## 22. Multiple Contours Within One Component

Normally one connected component should produce one external contour.

However, implementation details or masking operations may produce multiple candidates.

Policy:

1. compute contour areas
2. select the largest valid external contour

Conceptually:

```python
contour = max(contours, key=cv2.contourArea)
```

Log unexpected multiple-contour cases during development.

---

## 23. Raw Contour Validation

Before simplification, validate:

```text
number of points >= 3
contour area > 0
perimeter > minimum threshold
```

Reject degenerate contours.

Conceptual checks:

```python
if len(contour) < 3:
    return None

if cv2.contourArea(contour) <= 0:
    return None
```

Metric-area filtering may already remove many degenerate components.

---

## 24. Contour Simplification

Raster contours can contain many staircase points.

Example:

```text
raw contour: 300 points
```

The model target needs 32 points.

Before resampling, simplify the contour.

Recommended method:

```text
Douglas-Peucker
```

OpenCV:

```python
approx = cv2.approxPolyDP(
    contour,
    epsilon,
    closed=True,
)
```

---

## 25. Simplification Epsilon

A fixed pixel epsilon is easy but not robust across resolutions.

Preferred conceptual design:

```python
epsilon = simplify_ratio * perimeter
```

where:

```text
perimeter = cv2.arcLength(contour, True)
```

Example:

```python
epsilon = 0.005 * perimeter
```

The exact default should be selected empirically.

Recommended configuration:

```python
polygon_simplify_ratio = 0.005
```

Do not treat this numeric value as final without visual inspection.

The GT generation script should expose it as a CLI/config parameter.

---

## 26. Why Simplify Before Metric Conversion

Simplification can be done in pixel or metric space.

If:

```text
vx == vy
```

pixel-space simplification is usually acceptable.

If:

```text
vx != vy
```

pixel-space distance is anisotropic relative to metric geometry.

In that case, convert to metric coordinates before Douglas-Peucker simplification.

Therefore, the implementation must inspect BEV cell resolution.

Recommended rule:

```text
if |vx - vy| is negligible:
    simplify in pixel space
else:
    convert to metric first
```

A geometry library such as Shapely can simplify metric polygons when needed.

---

## 27. Pixel Contour to Metric Coordinates

OpenCV contours use:

```text
(column, row)
```

conceptually:

```text
(u, v)
```

This must not be confused with:

```text
(x, y)
```

BEV coordinates.

Create one explicit conversion function.

Conceptual signature:

```python
def contour_pixel_to_metric(
    contour_xy_pixel,
    occ_range,
    voxel_size,
    axis_order,
    flip_x,
    flip_y,
):
    ...
```

Actual parameters should follow the dataset metadata.

The conversion function should be unit-tested with known cell indices.

---

## 28. ROI Clipping

SparseDrive map prediction may use a specific local ROI.

Polygon OCC GT should match this ROI.

Possible occupancy annotation range may be larger or different.

The converter must:

```text
convert contour to metric
    ↓
clip polygon to SparseDrive ROI
```

Recommended geometry:

```text
ROI = rectangular box
```

Possible implementation:

```python
shapely.geometry.box(...)
```

and:

```python
polygon.intersection(roi_polygon)
```

---

## 29. Why Clip Before 32-Point Sampling

Suppose a polygon crosses the prediction ROI.

If sampled first:

```text
large global contour
    ↓
32 samples
    ↓
clip
```

the remaining ROI segment may have poor point distribution.

Preferred:

```text
polygon
    ↓
ROI clip
    ↓
final local polygon geometry
    ↓
32-point sampling
```

This ensures the 32 vertices represent the geometry the model is asked to predict.

---

## 30. Handling MultiPolygon After ROI Clipping

Clipping may split one polygon into multiple pieces.

Example:

```text
one irregular road region
        ↓
ROI intersection
        ↓
two disconnected polygon pieces
```

Shapely may return:

```text
MultiPolygon
```

V1 policy:

```text
each valid polygon part becomes one semantic region instance
```

Do not automatically merge disconnected pieces.

For each part:

```text
filter by minimum area
resample independently
assign same semantic class
```

This may increase GT polygon count.

Query-count statistics must account for this.

---

## 31. Polygon Geometry Repair

Shapely or equivalent geometry operations may detect invalid polygons.

Common issue:

```text
self-intersection
```

Raw contours should usually be valid, but conversions and simplification may introduce issues.

Possible repair strategy:

```python
polygon = polygon.buffer(0)
```

This is a common geometry repair technique.

However, do not apply it blindly without logging.

Recommended logic:

```text
if polygon invalid:
    attempt repair
    if repair succeeds:
        continue
    else:
        discard and count
```

Track:

```text
invalid_before_repair
repair_success
discarded_invalid
```

---

## 32. Polygon Exterior Extraction

After clipping and repair, extract the exterior ring.

Conceptually:

```python
coords = np.asarray(polygon.exterior.coords)
```

Shapely exterior coordinates may duplicate the first point at the end.

Example:

```text
P0
P1
P2
P3
P0
```

Remove the duplicated closing point before the 32-point resampling function.

Expected input to resampling:

```text
M unique ordered boundary points
```

Closure should be handled mathematically.

---

## 33. Uniform Perimeter Sampling

This is the required sampling method.

Input:

```text
M × 2 ordered polygon boundary points
```

Output:

```text
32 × 2 ordered sampled points
```

The boundary is treated as closed.

---

## 34. Reference Sampling Algorithm

Conceptual implementation:

```python
import numpy as np


def sample_closed_polygon(points: np.ndarray, num_points: int = 32) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")

    if len(points) < 3:
        raise ValueError("polygon requires at least 3 points")

    # Remove duplicated closing point if present.
    if np.allclose(points[0], points[-1]):
        points = points[:-1]

    closed = np.concatenate([points, points[:1]], axis=0)
    edges = closed[1:] - closed[:-1]
    edge_lengths = np.linalg.norm(edges, axis=1)

    perimeter = edge_lengths.sum()
    if perimeter <= 1e-6:
        raise ValueError("degenerate polygon perimeter")

    cumulative = np.concatenate(
        [np.array([0.0], dtype=np.float32), np.cumsum(edge_lengths)]
    )

    target_distances = np.arange(num_points, dtype=np.float32)
    target_distances *= perimeter / num_points

    sampled = []
    edge_idx = 0

    for distance in target_distances:
        while (
            edge_idx < len(edge_lengths) - 1
            and cumulative[edge_idx + 1] <= distance
        ):
            edge_idx += 1

        edge_length = edge_lengths[edge_idx]
        if edge_length <= 1e-8:
            sampled.append(closed[edge_idx].copy())
            continue

        alpha = (
            distance - cumulative[edge_idx]
        ) / edge_length

        point = (
            closed[edge_idx]
            + alpha * edges[edge_idx]
        )
        sampled.append(point)

    return np.asarray(sampled, dtype=np.float32)
```

This code is illustrative.

Claude Code should adapt style and validation to the repository.

---

## 35. Why the Sampling Distance Is `perimeter / num_points`

For a closed polygon with 32 samples:

```text
target distances:
0
1/32 L
2/32 L
...
31/32 L
```

Do not include:

```text
L
```

because:

```text
distance L == distance 0
```

on a closed boundary.

Including both creates a duplicated first/last point.

---

## 36. Orientation Normalization

After sampling, normalize polygon direction.

The project recommendation is:

```text
clockwise
```

One common signed-area convention is:

```python
signed_area = 0.5 * sum(
    x_i * y_{i+1} - x_{i+1} * y_i
)
```

The sign interpretation depends on coordinate convention.

For standard Cartesian:

```text
positive → counter-clockwise
negative → clockwise
```

But the actual SparseDrive coordinate frame must be verified.

Create an explicit helper:

```python
normalize_polygon_orientation(points, clockwise=True)
```

Do not rely on visual intuition.

---

## 37. Reference Signed-Area Function

Conceptual implementation:

```python
def signed_polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * np.sum(
        x * np.roll(y, -1)
        - np.roll(x, -1) * y
    )
```

Then:

```python
def ensure_clockwise(points: np.ndarray) -> np.ndarray:
    if signed_polygon_area(points) > 0:
        return points[::-1].copy()
    return points
```

Again, verify sign convention in the actual coordinate frame.

---

## 38. Deterministic Starting Vertex

After orientation normalization, choose a deterministic first point.

Recommended V1 rule:

```text
minimum x
then minimum y
```

Conceptually:

```python
def normalize_polygon_start(points: np.ndarray) -> np.ndarray:
    order = np.lexsort((points[:, 1], points[:, 0]))
    start_idx = int(order[0])
    return np.roll(points, -start_idx, axis=0)
```

This provides a deterministic cyclic rotation.

---

## 39. Important Limitation of Minimum-X Start Rule

The minimum-x point may be unstable if multiple sampled points have nearly identical x coordinates.

Example:

```text
long vertical edge
```

Small numerical noise may change the selected start point.

This can introduce target discontinuity.

V1 accepts this risk for simplicity.

Potential future improvements:

```text
minimum projected angle around centroid
nearest point to canonical anchor
cyclic-invariant matching
minimum cyclic L1 loss
```

These are not V1 requirements.

The GT generation code should still be written so the start-point policy can be replaced later.

---

## 40. Final Ordering Pipeline

The required order is:

```text
polygon geometry
    ↓
uniform 32-point sampling
    ↓
orientation normalization
    ↓
starting point normalization
```

Do not normalize the starting point before sampling.

The sampled vertices are the actual regression targets.

Their cyclic order must be deterministic.

---

## 41. Polygon Validity Checks

After final 32-point generation, check:

```text
shape == [32, 2]
finite values
no NaN
no Inf
perimeter > threshold
absolute area > threshold
```

Optional geometry check:

```text
Shapely Polygon(points).is_valid
```

Record invalid outputs.

Possible policy:

```text
attempt repair before sampling
reject invalid after sampling only if severely degenerate
```

Do not silently accept NaN or zero-area polygons.

---

## 42. Duplicate Point Check

Uniform sampling may produce duplicate or near-duplicate points when:

```text
polygon perimeter is tiny
simplified polygon has zero-length edges
geometry is degenerate
```

Measure:

```text
min adjacent edge length
```

Conceptually:

```python
closed = np.concatenate([pts, pts[:1]], axis=0)
edge_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
```

Reject or log if many edge lengths are below a small threshold.

---

## 43. Coordinate Normalization for SparseDrive

The Polygon GT should ultimately use the normalization expected by the existing map branch.

Do not invent a second normalization formula.

Recommended integration order:

```text
Occupancy contour
    ↓
metric local BEV coordinates
    ↓
32-point polygon
    ↓
existing SparseDrive map normalization
```

Claude Code must inspect the original map point target preparation.

The same ROI scaling and normalization should be reused if compatible.

---

## 44. Recommended GT Data Structure

Conceptual offline output:

```python
{
    "sample_token": "...",
    "polygon_labels": np.ndarray([N], dtype=np.int64),
    "polygon_pts": np.ndarray([N, 32, 2], dtype=np.float32),
}
```

Optional debug metadata:

```python
{
    "polygon_areas": ...,
    "source_component_sizes": ...,
    "source_class_ids": ...,
}
```

Debug metadata does not need to enter the training DataLoader.

---

## 45. Offline File Storage

Possible formats:

```text
.pkl
.npz
.npy + metadata
```

The preferred final format should follow the existing SparseDrive dataset preprocessing convention.

Avoid introducing a custom database format for V1.

A practical prototype may use:

```text
.npz per sample
```

but the production implementation should consider:

```text
number of files
loading overhead
existing dataset info pipeline
```

Claude Code should inspect how SparseDrive stores processed map annotations.

---

## 46. Offline vs Runtime GT Conversion

### Offline Conversion

Advantages:

```text
fast training
easy visualization
easy statistics
deterministic
easy debugging
```

Disadvantages:

```text
extra storage
preprocessing step
```

### Runtime Conversion

Advantages:

```text
no extra saved Polygon GT
easy parameter iteration
```

Disadvantages:

```text
slower DataLoader
more hidden preprocessing
harder reproducibility
```

V1 recommendation:

```text
offline prototype
then integrate cached/offline GT into training
```

---

## 47. Required Offline Statistics

Before finalizing model configuration, generate dataset statistics.

At minimum:

```text
total samples processed
samples with zero polygons
mean polygons per frame
median polygons per frame
P90 polygons per frame
P95 polygons per frame
P99 polygons per frame
maximum polygons per frame
```

Per class:

```text
polygon count
frames containing class
mean polygon area
median polygon area
P95 polygon area
mean raw contour points
mean simplified contour points
```

Filtering statistics:

```text
discarded tiny components
discarded invalid contours
repair attempts
repair successes
```

These statistics directly affect:

```text
query count
class set
minimum area threshold
simplification ratio
```

---

## 48. Query Count Decision From GT Statistics

Do not choose a new query count based on intuition.

Suppose statistics show:

```text
P95 polygons per frame = 42
max = 83
```

Then an existing query count of 100 may be sufficient.

Suppose:

```text
P95 = 130
```

Then the map query count may need to increase.

The GT generator must therefore produce polygon-count statistics before full model training.

---

## 49. Required Visualization Suite

GT generation is not considered complete without visualization.

Implement at least the following views.

### View A: Raw Semantic BEV Mask

For each class:

```text
binary mask
```

Purpose:

```text
verify class mapping and projection
```

### View B: Connected Components

Each component rendered with a distinct temporary ID.

Purpose:

```text
verify region splitting
```

### View C: Raw Contour

Draw raw contour points.

Purpose:

```text
inspect raster noise
```

### View D: Simplified Polygon

Draw simplified geometry.

Purpose:

```text
verify Douglas-Peucker behavior
```

### View E: 32-Point Polygon

Draw:

```text
points
edges
closure edge
point index 0
```

Purpose:

```text
verify sampling and ordering
```

### View F: Polygon Rasterization

Rasterize the final 32-point polygon.

Compare against the source class mask.

Purpose:

```text
measure representation loss
```

---

## 50. Polygon Approximation Error

Even before training, measure how much information is lost when converting occupancy masks to 32-point polygons.

For each GT region:

```text
source component mask
        vs
rasterized 32-point polygon
```

Measure:

```text
IoU
Dice
area ratio
```

This is a critical preprocessing metric.

The model should not be blamed for representation error already introduced by GT compression.

---

## 51. Recommended GT Approximation Metrics

For each polygon:

```text
mask_iou
mask_dice
source_area
polygon_area
area_relative_error
```

Dataset summary:

```text
mean IoU
median IoU
P10 IoU
per-class mean IoU
```

If a class has poor GT approximation before training, investigate:

```text
32 points insufficient
contour fragmentation
simplification too aggressive
holes
BEV projection problem
```

---

## 52. Why This Metric Is Important

Suppose Polygon OCC prediction reaches a maximum IoU of 0.85.

If the GT conversion itself only reproduces the source occupancy at 0.86 IoU, the model may already be near the representation ceiling.

Therefore:

```text
occupancy → 32-point polygon approximation quality
```

must be measured separately.

This should be part of the Polygon OCC experimental methodology.

---

## 53. Minimum-Area Threshold Selection

Do not select `min_area_m2` arbitrarily.

Generate area histograms per class.

For example:

```text
vehicle:
0.5 m² to 20 m²

road:
10 m² to thousands of m²

pedestrian:
very small
```

A single threshold may remove pedestrians while preserving roads.

Therefore, V1 may require class-specific thresholds.

Recommended decision process:

```text
1. collect class area statistics
2. visualize tiny components
3. define thresholds
4. record thresholds in config
```

Avoid hard-coded inline `if class == ...` logic.

---

## 54. Large Region Handling

Large semantic regions such as:

```text
drivable_surface
terrain
vegetation
```

may span most of the ROI.

One connected region may have highly complex shape.

Possible issues:

```text
32 points insufficient
region wraps around ROI boundary
multiple narrow branches
```

V1 policy:

```text
allow large region as one polygon if connected
```

Do not split it heuristically at first.

Measure representation IoU.

If large regions are poorly approximated, future strategies may include:

```text
region decomposition
multiple polygons per connected component
polygon partitioning
```

These are outside V1.

---

## 55. Vehicle and Pedestrian Regions

Occupancy-derived vehicle and pedestrian masks may be small.

Potential issues:

```text
very few BEV cells
coarse box-like contour
merged nearby instances
```

The converter should not force all polygons to use different sampling counts.

Even a small rectangle is sampled to 32 points.

This creates repeated linear interpolation along edges.

That is valid.

The model always predicts:

```text
32 × 2
```

for every class.

---

## 56. Why Small Rectangles Can Use 32 Points

A vehicle footprint may geometrically require only four corners.

However, fixed 32-point representation provides a uniform model interface.

For a rectangle:

```text
8 points may lie on each side
```

approximately, depending on side lengths.

This is not geometrically wasteful from the model-design perspective.

The regression head remains class-agnostic.

Variable point count per class is deferred.

---

## 57. Class Overlap Handling

Because masks are independent, polygons may overlap.

Example:

```text
road polygon
vehicle polygon
```

This is expected.

Do not perform:

```text
vehicle mask subtraction from road mask
```

in V1 unless the design is explicitly changed.

Overlap should remain visible in GT.

The final prediction is a set of semantic regions, not an exclusive raster label map.

---

## 58. Duplicate / Near-Duplicate Region Handling

Different source classes should not be merged.

Within the same class, connected-component extraction already separates disconnected regions.

After ROI clipping, geometry operations may produce near-duplicate polygons.

Optional debug check:

```text
same class
high polygon IoU
similar area
```

Do not introduce duplicate suppression unless duplicates are observed.

---

## 59. Class-Specific Morphology

V1 should prefer one shared morphology policy.

However, some classes may eventually need different settings.

Examples:

```text
road:
closing may fill small gaps

pedestrian:
closing may merge nearby people
```

Therefore, the configuration should allow:

```python
morphology_cfg = {
    "default": ...,
    "pedestrian": ...,
}
```

but class-specific behavior should only be enabled after visual evidence.

---

## 60. Recommended Conversion Module Interfaces

Conceptual API:

```python
def load_occ_annotation(sample_info):
    ...

def build_class_bev_masks(occ, metadata, class_mapping, cfg):
    ...

def extract_components(mask, cfg):
    ...

def component_to_polygon(component_mask, metadata, cfg):
    ...

def simplify_polygon(points, cfg):
    ...

def sample_closed_polygon(points, num_points):
    ...

def normalize_polygon_orientation(points, clockwise=True):
    ...

def normalize_polygon_start(points):
    ...

def validate_polygon(points, cfg):
    ...

def build_polygon_gt(sample_info, cfg):
    ...
```

The actual repository structure may differ.

The key requirement is separation of responsibilities.

---

## 61. Recommended Main Conversion Pseudocode

```python
def build_polygon_gt(sample_info, cfg):
    occ, occ_meta = load_occ_annotation(sample_info)

    class_masks = build_class_bev_masks(
        occ=occ,
        metadata=occ_meta,
        class_mapping=cfg.class_mapping,
        cfg=cfg,
    )

    polygon_pts = []
    polygon_labels = []

    for class_name, mask in class_masks.items():
        mask = maybe_apply_morphology(
            mask,
            class_name,
            cfg,
        )

        components = extract_components(
            mask,
            class_name,
            cfg,
        )

        for component in components:
            geometries = component_to_clipped_polygons(
                component,
                occ_meta,
                cfg.roi,
                cfg,
            )

            for polygon in geometries:
                polygon = simplify_polygon(
                    polygon,
                    class_name,
                    cfg,
                )

                sampled = sample_closed_polygon(
                    polygon.exterior_coords,
                    num_points=cfg.num_polygon_points,
                )

                sampled = normalize_polygon_orientation(
                    sampled,
                    clockwise=True,
                )

                sampled = normalize_polygon_start(
                    sampled,
                )

                if not validate_polygon(sampled, cfg):
                    continue

                polygon_pts.append(sampled)
                polygon_labels.append(
                    cfg.class_name_to_label[class_name]
                )

    return {
        "gt_polygon_pts": np.asarray(
            polygon_pts,
            dtype=np.float32,
        ),
        "gt_polygon_labels": np.asarray(
            polygon_labels,
            dtype=np.int64,
        ),
    }
```

This is conceptual pseudocode.

Do not copy it blindly without matching SparseDrive conventions.

---

## 62. Empty-GT Samples

Some samples may contain no retained polygons after class filtering and minimum-area filtering.

The GT pipeline must support:

```text
N = 0
```

Expected shapes should be well-defined.

Conceptually:

```python
gt_polygon_pts.shape == (0, 32, 2)
gt_polygon_labels.shape == (0,)
```

Do not return malformed arrays such as:

```text
(0,)
```

for polygon points.

The downstream assigner and DataLoader should be tested on empty-GT samples.

---

## 63. Determinism

Offline GT generation should be deterministic.

Given:

```text
same sample
same config
```

the generated polygons should be identical.

Avoid:

```text
random contour order
random starting vertex
unordered dictionary dependence
```

Sort class processing in a stable order.

If multiple polygon parts exist, define a deterministic ordering for saved GT.

Example:

```text
class label
then centroid x
then centroid y
then area
```

GT set order should not affect Hungarian matching, but deterministic files improve reproducibility and debugging.

---

## 64. Polygon Ordering Within a Sample

A recommended saved-order policy is:

```text
sort by:
1. class label
2. centroid x
3. centroid y
```

This is not a training requirement.

Hungarian matching treats GT as a set.

However, deterministic sample files are easier to diff and inspect.

---

## 65. Cache Versioning

Polygon GT depends on preprocessing parameters.

Examples:

```text
class mapping
z range
morphology
minimum area
simplify ratio
ROI
num_polygon_points
ordering policy
```

Therefore, saved GT must include a preprocessing version.

Conceptually:

```python
"polygon_gt_version": "v1"
```

or config hash.

Do not reuse old cached Polygon GT after changing preprocessing settings without explicit invalidation.

---

## 66. Recommended Preprocessing Metadata

Store one global metadata file containing:

```text
polygon_gt_version
source_occ_dataset
source_class_mapping
polygon_classes
num_polygon_points
roi
voxel size
z projection policy
morphology config
min area thresholds
simplify ratio
orientation policy
start point policy
```

This is valuable for experiment reproducibility.

---

## 67. Unit Tests

The GT generation code should include unit tests for geometry helpers.

Minimum tests:

### Test 1: Rectangle Sampling

Input:

```text
rectangle
```

Verify:

```text
shape == [32, 2]
finite values
closed perimeter sampling
```

### Test 2: Duplicate Closing Point

Input:

```text
P0, P1, P2, P3, P0
```

Verify duplicate closing point is removed.

### Test 3: Clockwise Normalization

Input counter-clockwise polygon.

Verify output orientation.

### Test 4: Start Vertex Normalization

Verify the canonical start point becomes index 0.

### Test 5: Degenerate Polygon

Input:

```text
all points identical
```

Verify rejection.

### Test 6: Empty GT

Verify correct empty shapes.

### Test 7: ROI Clipping

Polygon crossing ROI.

Verify clipped geometry.

### Test 8: MultiPolygon

Geometry split after clipping.

Verify multiple GT instances.

---

## 68. Integration Test

Select one known sample.

Run:

```text
occupancy annotation
        ↓
Polygon GT conversion
```

Then verify:

```text
sample token
class masks
polygon labels
polygon coordinates
32-point shapes
visualization
rasterized approximation
```

Save all intermediate outputs for this sample.

This sample becomes the reference integration test.

---

## 69. DataLoader Validation

Before modifying the model, inspect one DataLoader batch.

Verify:

```text
batch size
number of polygons per sample
polygon point shape
label shape
dtype
device transfer
data container behavior
```

Expected conceptual per-sample target:

```text
[N_i, 32, 2]
```

Do not assume all samples have the same `N_i`.

The batching logic should follow the original map GT pattern.

---

## 70. Required Failure Logging

The preprocessing script should count and report:

```text
missing occupancy annotation
unknown sample mapping
invalid occupancy shape
unknown semantic label
empty class mask
tiny component discarded
invalid contour discarded
polygon repair attempted
polygon repair failed
ROI clipping removed polygon
sampling failed
final validation failed
```

Do not silently skip everything with a broad `try/except`.

Unexpected exceptions should include:

```text
sample token
class name
processing stage
```

---

## 71. Recommended CLI / Configuration Parameters

Conceptual parameters:

```text
--occ-root
--output-root
--num-polygon-points 32
--simplify-ratio
--min-area-m2
--enable-morphology
--morph-kernel-size
--roi-x-min
--roi-x-max
--roi-y-min
--roi-y-max
--visualize
--stats-output
```

Actual implementation should use the project's preferred configuration style.

The important requirement is parameter visibility.

---

## 72. GT Generation Acceptance Criteria

Polygon GT generation is complete only if all conditions are met.

### Alignment

```text
[ ] occupancy samples align with SparseDrive samples
```

### Semantic Mapping

```text
[ ] class IDs verified
[ ] class mapping centralized
```

### BEV Projection

```text
[ ] x/y orientation verified
[ ] z-axis verified
[ ] ROI verified
```

### Polygon Extraction

```text
[ ] connected components verified
[ ] contours visually correct
[ ] tiny region filtering measured
```

### Geometry

```text
[ ] simplification visually correct
[ ] 32-point perimeter sampling correct
[ ] orientation deterministic
[ ] starting point deterministic
```

### Quality

```text
[ ] source mask vs rasterized polygon IoU measured
[ ] per-class approximation quality reported
```

### Integration

```text
[ ] empty-GT sample supported
[ ] DataLoader shape validated
[ ] reference sample visualization saved
```

Do not begin full training before these criteria are satisfied.

---

## 73. Suggested Output Directory Layout

Conceptual layout:

```text
polygon_occ_gt/
├── metadata.json
├── statistics.json
├── samples/
│   ├── <sample_token>.npz
│   ├── <sample_token>.npz
│   └── ...
└── visualization/
    ├── masks/
    ├── contours/
    ├── polygons/
    └── raster_compare/
```

This is a prototype recommendation.

The final storage path should integrate cleanly with SparseDrive's dataset preparation workflow.

---

## 74. Example Saved Sample

Conceptual `.npz` content:

```python
sample_token = "..."
polygon_pts = np.ndarray(
    shape=(N, 32, 2),
    dtype=np.float32,
)
polygon_labels = np.ndarray(
    shape=(N,),
    dtype=np.int64,
)
```

Optional:

```python
polygon_areas = np.ndarray(
    shape=(N,),
    dtype=np.float32,
)
```

The training pipeline only requires geometry and labels.

---

## 75. Example Polygon GT

Conceptually:

```text
Polygon 0
class = drivable_surface
shape = [32, 2]

Polygon 1
class = vegetation
shape = [32, 2]

Polygon 2
class = vehicle
shape = [32, 2]
```

The polygon list is variable length.

The vertex count is fixed.

This is the central GT structure of Polygon OCC V1.

---

## 76. Important Research Note: GT Compression Error

Polygon OCC V1 does not directly supervise the original occupancy grid.

It supervises a compressed representation:

```text
Semantic Occupancy
        ↓
Polygon vectorization
        ↓
32 vertices
```

This introduces an upper-bound approximation error.

Therefore, any paper or experiment should separate:

```text
GT vectorization quality
```

from:

```text
model prediction quality
```

Recommended evaluation chain:

```text
Occ GT
vs
Rasterized Polygon GT

and

Polygon GT
vs
Predicted Polygon
```

This distinction is important for fair interpretation.

---

## 77. Important Research Note: Polygon Count Is Representation Capacity

Dense OCC has fixed spatial capacity:

```text
H × W × Z
```

Polygon OCC has two major capacity parameters:

```text
number of queries
number of vertices per polygon
```

For V1:

```text
query_count
32 vertices
```

The GT statistics should therefore be used to reason about representation capacity.

If one frame contains more semantic regions than available queries, Polygon OCC cannot represent all regions regardless of training quality.

---

## 78. Important Research Note: Per-Class Masks Allow Multi-Layer Semantics

Independent masks provide a useful property.

The scene may contain:

```text
road region
vehicle region above road
vegetation region beside road
```

Polygon OCC can preserve overlapping semantic regions.

This differs from a single-label BEV semantic segmentation target.

The GT pipeline should not accidentally destroy this property by applying class-priority overwrite logic.

---

## 79. Claude Code Instructions

Before implementing the GT pipeline, Claude Code must:

1. inspect the user's SparseDrive checkout
2. identify current map GT generation and loading
3. identify map ROI and coordinate normalization
4. identify the local occupancy dataset format
5. verify semantic label IDs
6. verify occupancy tensor axis order
7. verify sample-token alignment

Then implement an offline GT prototype.

Do not modify the SparseDrive model first.

The required order is:

```text
understand data
    ↓
generate Polygon GT
    ↓
visualize
    ↓
measure approximation error
    ↓
integrate DataLoader
    ↓
modify model
```

---

## 80. Final GT Pipeline Definition

Polygon OCC V1 GT generation is defined as:

```text
nuScenes-based Semantic Occupancy
        ↓
Align by sample identifier
        ↓
Decode occupancy tensor and semantics
        ↓
Generate independent BEV mask per selected class
        ↓
Optional light morphology
        ↓
Extract 8-connected semantic components
        ↓
Filter tiny regions
        ↓
Extract external contour
        ↓
Convert contour to metric local BEV coordinates
        ↓
Clip to SparseDrive map ROI
        ↓
Handle Polygon / MultiPolygon outputs
        ↓
Repair invalid geometry when possible
        ↓
Simplify contour
        ↓
Uniformly sample 32 points along closed perimeter
        ↓
Normalize orientation
        ↓
Normalize cyclic starting vertex
        ↓
Validate final polygon
        ↓
Save / return:
    gt_polygon_pts [N, 32, 2]
    gt_polygon_labels [N]
```

The main objective of this pipeline is not maximum geometric sophistication.

The objective is to produce:

```text
clean
deterministic
fixed-length
debuggable
SparseDrive-compatible
```

semantic polygon targets.

The GT pipeline should remain modular so later Polygon OCC versions can replace:

```text
32-point sampling
vertex ordering
contour simplification
```

without rewriting occupancy decoding or dataset alignment.

For V1, correctness and reproducibility are more important than aggressive preprocessing sophistication.
