# 01_Polygon_OCC_Project_Overview

## 1. Project Name

**Polygon OCC**

> Workspace note: the current active SparseDrive Polygon OCC baseline in this
> repository uses **20 points per polygon**. Historical design discussion in
> this document may still describe an earlier 32-point target.

Target architecture:

**SparseDrive**

Core idea:

> Replace the original LineString-based map representation in SparseDrive with a sparse polygon-based semantic occupancy representation.

This project does **not** append a conventional dense OCC head to SparseDrive.

Instead, it changes the scene representation of the original map branch:

```text
LineString Map Representation
            ↓
Polygon Occupancy Representation
```

The first version focuses on validating whether SparseDrive's sparse query framework can directly predict semantic occupied regions as closed polygons.

---

## 2. Definition of Polygon OCC

Polygon OCC is defined as:

> A sparse vectorized occupancy representation that models semantic occupied regions as a set of closed polygons instead of dense grid or voxel cells.

Traditional semantic occupancy usually represents the scene as:

```text
H × W × Z
```

or, for BEV occupancy:

```text
H × W
```

Each grid or voxel cell contains an occupancy state or semantic category.

Polygon OCC uses a different representation:

```text
N semantic regions
        ↓
N sparse queries
        ↓
Each query predicts one polygon
        ↓
class + confidence + vertices
```

For Version 1:

```text
Polygon Query Output
├── class
├── confidence
└── 32 × (x, y)
```

The fundamental representation is therefore:

```text
Scene
=
{
    Polygon_0,
    Polygon_1,
    ...
    Polygon_N
}
```

Each polygon represents one semantic occupied region.

---

## 3. Polygon OCC Is Not a Dense OCC Head

This distinction is a hard project constraint.

Claude Code must **not** interpret Polygon OCC as:

```text
BEV Feature
    ↓
Dense OCC Head
    ↓
H × W × Z voxel prediction
```

Do not introduce:

- dense voxel decoder
- 3D convolution OCC head
- dense BEV semantic segmentation head
- voxel query decoder
- conventional semantic occupancy pipeline

The V1 target architecture is:

```text
Camera Inputs
      ↓
SparseDrive Backbone
      ↓
Image Features
      ↓
SparseDrive Existing Feature Interaction
      ↓
Sparse Map Queries
      ↓
Polygon Prediction
      ↓
Semantic Polygon Set
```

The goal is to preserve the **sparse query architecture**.

---

## 4. Current SparseDrive Representation

The original SparseDrive map branch predicts vectorized HD map elements.

Typical map classes include:

```text
divider
ped_crossing
boundary
```

Each map instance is represented as a LineString or polyline.

Conceptually:

```text
Map Query
    ↓
class
+
ordered points
```

Example:

```text
P0 → P1 → P2 → ... → P19
```

The geometry is open or line-oriented.

This representation is highly suitable for:

- lane divider
- road divider
- road boundary
- centerline-like geometry

However, it describes geometry primarily as a **curve**.

It does not directly describe the occupied area inside a semantic region.

---

## 5. Motivation for Polygon OCC

Consider a road region.

A LineString may describe:

```text
-------------------------
```

This can represent one road boundary.

However, the actual road occupies an area:

```text
#########################
#########################
#########################
```

A Polygon represents the full region:

```text
P0 -------- P1
|            |
|            |
P3 -------- P2
```

The key representation change is therefore:

```text
1D curve representation
        ↓
2D region representation
```

Polygon OCC is intended to represent semantic areas such as:

```text
drivable_surface
sidewalk
terrain
vegetation
vehicle
pedestrian
barrier
other semantic regions
```

The exact V1 category set must be determined from the selected occupancy GT source and dataset conversion design.

The first implementation must avoid unnecessarily expanding the number of classes before the full GT generation pipeline is validated.

---

## 6. Why Use the Name "Polygon OCC"

The term OCC is retained because the representation is intended to model **semantic occupied regions**.

However, the native model output is not a voxel tensor.

The representation is:

```text
Sparse
+
Vectorized
+
Semantic
+
Region-based
```

Instead of:

```text
Dense
+
Grid-based
+
Cell-centric
```

The conceptual comparison is:

### Traditional OCC

```text
Scene
  ↓
Dense spatial discretization
  ↓
Grid / voxel cells
  ↓
Semantic occupancy per cell
```

### Polygon OCC

```text
Scene
  ↓
Semantic region decomposition
  ↓
Sparse Polygon Queries
  ↓
Closed semantic polygons
```

Therefore, Polygon OCC should be understood as a **polygon-based occupancy representation**, not as a conventional dense occupancy decoder.

---

## 7. Version 1 Objective

Version 1 is a proof-of-concept implementation.

The only major research variable should be:

```text
LineString
    ↓
Polygon
```

Version 1 must implement:

- Polygon GT generation
- fixed 32-point Polygon representation
- Polygon prediction in the original map branch
- polygon-aware visualization
- existing classification supervision
- existing Hungarian matching strategy whenever possible
- vertex regression supervision

Version 1 must **not** implement:

- differentiable rasterization
- Dice Loss
- Focal occupancy loss
- Chamfer Loss
- Hausdorff Loss
- dense OCC head
- 3D voxel prediction
- unified detection and Polygon query architecture

These are future-stage extensions.

---

## 8. Fixed 32-Point Polygon Representation

Each Polygon OCC instance uses exactly 32 vertices.

Representation:

```text
Polygon
=
[
    (x0, y0),
    (x1, y1),
    ...
    (x31, y31)
]
```

Tensor shape:

```text
(32, 2)
```

Regression dimension:

```text
32 × 2 = 64
```

The polygon is closed geometrically by connecting:

```text
P31 → P0
```

The stored tensor should not require duplicating the first vertex as the last vertex.

Incorrect representation:

```text
P0
P1
...
P31
P0
```

This would contain 33 points.

Preferred representation:

```text
P0
P1
...
P31
```

Closure is implicit.

The visualization and rasterization modules must explicitly connect the final vertex back to the first vertex.

---

## 9. Why 32 Vertices

Polygon geometry naturally has variable vertex count.

Examples:

```text
vehicle footprint      → 4 vertices
simple road region     → several vertices
complex vegetation     → hundreds of contour points
```

A neural network regression head requires a fixed output dimension.

Therefore, all GT polygons are converted to:

```text
32 vertices
```

The V1 strategy is:

```text
Raw Polygon / Contour
        ↓
Polygon simplification
        ↓
Uniform perimeter sampling
        ↓
32 ordered vertices
```

The reasons for choosing 32 are:

- higher geometric capacity than the original 20-point map representation
- fixed regression dimension
- simple integration with SparseDrive
- sufficient capacity for the first proof of concept
- low implementation risk
- compatible with point-based regression

The number 32 is a design choice and should remain configurable.

Recommended configuration variable:

```python
num_polygon_points = 32
```

Do not hard-code 32 in multiple source files.

---

## 10. Ground Truth Source

The preferred V1 direction is to derive Polygon GT from semantic occupancy annotations based on nuScenes.

Candidate occupancy datasets include:

```text
Occ3D-nuScenes
OpenOccupancy / OpenOcc-style occupancy annotations
```

The exact dataset must be verified during implementation based on the user's local dataset and repository environment.

The core conversion pipeline is:

```text
Semantic Occupancy GT
        ↓
Project / collapse selected semantics to BEV
        ↓
Per-class semantic mask
        ↓
Connected region extraction
        ↓
Contour extraction
        ↓
Polygon simplification
        ↓
Uniform sampling
        ↓
32-point Polygon GT
```

Final dataset fields should conceptually become:

```text
gt_polygon_pts
gt_polygon_labels
```

Expected shapes:

```text
gt_polygon_pts:
    List[Tensor(32, 2)]

gt_polygon_labels:
    Tensor(num_gt_polygons)
```

The exact data container and naming should follow SparseDrive repository conventions.

Claude Code should inspect the existing map GT data path before changing field names.

---

## 11. Occupancy-to-Polygon Conversion

The high-level GT conversion is:

```text
3D Semantic OCC
        ↓
BEV semantic reduction
        ↓
2D semantic masks
        ↓
Polygon extraction
```

The BEV reduction strategy is a design-sensitive step.

It must not be implemented blindly.

Possible strategies include:

### Strategy A: Any-Occupied Projection

For a semantic class:

```text
BEV(x, y) = class
if any voxel along z belongs to class
```

Advantages:

- simple
- retains vertical objects

Risks:

- semantic overlap between classes
- tall structures may cover ground semantics

### Strategy B: Priority-Based Semantic Projection

Assign class priorities.

Example concept:

```text
vehicle > pedestrian > barrier > vegetation > drivable_surface
```

This is suitable for generating a single semantic BEV mask.

Risks:

- manually designed priority affects GT

### Strategy C: Per-Class Independent Masks

Generate independent masks:

```text
road_mask
vehicle_mask
pedestrian_mask
vegetation_mask
...
```

Then extract polygons independently.

This allows polygon overlap.

This is currently the preferred conceptual design for Polygon OCC.

The reason is that Polygon OCC is region-based and does not require every BEV coordinate to belong to exactly one polygon.

For example:

```text
road polygon
+
vehicle polygon
```

A vehicle polygon may geometrically overlap a road polygon.

This is semantically valid.

Therefore, V1 should strongly consider **per-class independent masks** rather than forcing a single exclusive BEV label map.

---

## 12. Polygon Instances and Semantic Regions

Polygon OCC is instance-like in prediction structure but region-like in semantics.

This requires a clear distinction.

A query predicts one polygon:

```text
Query_i
    ↓
class_i
polygon_i
```

However, a polygon does not necessarily correspond to a physical object instance.

Examples:

```text
vehicle polygon
    → may correspond to one vehicle

pedestrian polygon
    → may correspond to one pedestrian

road polygon
    → corresponds to one connected road semantic region

vegetation polygon
    → corresponds to one connected vegetation region
```

Therefore, Polygon OCC primitives are better understood as:

> Semantic Region Instances

They are not identical to detection instances.

This distinction is important for GT generation and Hungarian matching.

---

## 13. Connected Component Policy

For each semantic class mask:

```text
Binary Mask
    ↓
Connected Components
```

Each connected component may produce one Polygon GT.

Example:

```text
Road region A
Road region B
```

becomes:

```text
Polygon 0: road
Polygon 1: road
```

Likewise:

```text
Vehicle A
Vehicle B
Vehicle C
```

may become three separate polygons if the occupancy mask preserves separation.

Potential issue:

Two nearby objects may be merged into one occupancy component due to voxel resolution.

Therefore, GT generation must record:

- source semantic class
- connected component count
- polygon area
- contour complexity
- number of polygons per sample

The first implementation should include offline statistics.

Recommended statistics:

```text
polygons per frame
polygons per class
polygon area distribution
raw contour point count
discarded small components
32-point resampling error
```

These statistics are necessary before finalizing query count and class configuration.

---

## 14. V1 Loss Strategy

Version 1 reuses the original SparseDrive map regression philosophy.

The intended loss is:

```text
L_total
=
L_cls
+
λ_vertex × L_vertex
```

Where:

```text
L_vertex
=
L1(predicted_vertices, matched_gt_vertices)
```

The exact SparseDrive loss implementation must be inspected and reused whenever possible.

Do not create a new Polygon loss module if the original point regression loss already supports:

```text
N × 64
```

or an equivalent flattened representation.

The implementation goal is:

```text
20 × 2 polyline regression
        ↓
32 × 2 polygon vertex regression
```

rather than introducing a new loss framework.

---

## 15. Vertex Ordering Is a Critical Constraint

A Polygon has cyclic symmetry.

The following polygons are geometrically identical:

```text
[P0, P1, P2, P3]
[P1, P2, P3, P0]
[P2, P3, P0, P1]
[P3, P0, P1, P2]
```

Clockwise and counter-clockwise sequences may also describe the same polygon.

However, plain Vertex L1 sees them as different tensors.

This is one of the most important technical risks in Version 1.

Therefore, GT generation must enforce a deterministic ordering policy.

Minimum required policy:

1. enforce clockwise or counter-clockwise order
2. select a deterministic starting vertex

Recommended V1 ordering:

### Direction

Always clockwise.

### Starting Point

Choose one deterministic rule, for example:

```text
vertex with minimum x
```

If multiple points share similar x:

```text
choose minimum y
```

Equivalent lexicographic rule:

```python
start_idx = argmin((x, y))
```

The exact coordinate convention must match SparseDrive's map coordinate system.

Alternative starting-point policies may be evaluated later.

Claude Code must not ignore cyclic ordering.

Without ordering normalization:

```text
GT Polygon == Prediction Polygon geometrically
```

may still produce a very large Vertex L1 loss.

---

## 16. Hungarian Matching

The initial implementation should preserve SparseDrive's existing Hungarian matching structure.

Conceptually:

```text
matching_cost
=
classification_cost
+
point_regression_cost
```

For Polygon OCC V1:

```text
point_regression_cost
    ↓
polygon_vertex_regression_cost
```

The preferred minimal modification is to reuse the same cost implementation after changing the regression dimension from:

```text
20 × 2
```

to:

```text
32 × 2
```

However, this is valid only after deterministic polygon ordering is guaranteed.

Do not add:

- raster IoU matching cost
- mask Dice matching cost
- Polygon IoU matching cost
- Chamfer matching cost

in Version 1.

These changes would alter both representation and matching simultaneously.

---

## 17. Why Rasterization Is Deferred

A Polygon can naturally be converted into a raster mask:

```text
Polygon
    ↓
Rasterization
    ↓
BEV Mask
```

This makes the following future loss possible:

```text
L_total
=
L_vertex
+
λ_raster × L_raster
```

Possible raster losses:

```text
Dice Loss
Focal Loss
Binary Cross Entropy
Semantic Cross Entropy
```

From a final performance perspective, Vertex Loss plus Raster Loss is expected to provide complementary supervision:

```text
Vertex Loss
    ↓
vertex correspondence and geometry

Raster Loss
    ↓
region area and shape consistency
```

However, V1 intentionally postpones Raster Loss because it introduces new variables:

- raster grid size
- BEV range
- raster resolution
- differentiable rasterizer
- anti-aliasing
- semantic overlap handling
- polygon validity
- self-intersection behavior

The V1 experiment must answer one clean question:

> Can SparseDrive sparse queries learn closed semantic polygons by minimally replacing the original polyline map representation?

---

## 18. V1 Architecture Constraints

The following architecture components should remain unchanged unless a shape/interface dependency makes a small modification necessary:

```text
image backbone
image neck
feature extraction
SparseDrive core query architecture
temporal design
attention blocks
decoder depth
training runner
optimizer
learning rate schedule
detection branch
motion branch
planning branch
```

The main modification target is the existing map task path.

Conceptually:

```text
map GT generation
        ↓
map regression target
        ↓
map query regression dimension
        ↓
map decoder output
        ↓
map visualization/evaluation
```

The repository must be inspected before implementation.

Do not assume file names from this document.

The document describes **design intent**, while Claude Code should derive the actual call graph and exact files from the checked-out SparseDrive source tree.

---

## 19. Minimal Modification Principle

When two implementation options are possible, choose the option with fewer architectural changes.

Preferred:

```text
reuse existing Map Head
change num_sample
change GT geometry
change point interpretation
```

Avoid:

```text
create completely new OCC branch
create new transformer decoder
duplicate Map Head
rewrite Hungarian assigner
rewrite training loop
```

For example, if the original map head already predicts:

```text
num_sample × 2
```

then Version 1 should first test:

```text
num_sample = 32
```

and reinterpret the ordered point sequence as a closed polygon.

This is preferable to creating `polygon_head.py` immediately.

Only create a dedicated Polygon module when the original module semantics or implementation make reuse unsafe or confusing.

---

## 20. Expected V1 Output

For each sample:

```text
Polygon OCC Predictions

├── Polygon 0
│   ├── label
│   ├── score
│   └── vertices [32, 2]
│
├── Polygon 1
│   ├── label
│   ├── score
│   └── vertices [32, 2]
│
└── ...
```

Example conceptual output:

```python
{
    "scores": Tensor[N],
    "labels": Tensor[N],
    "polygons": Tensor[N, 32, 2],
}
```

The actual key names should follow SparseDrive decoder conventions.

Do not rename output keys unnecessarily in the first implementation.

---

## 21. Visualization Requirements

Visualization is mandatory for V1.

For each prediction:

1. obtain the ordered 32 vertices
2. connect adjacent vertices
3. connect vertex 31 back to vertex 0
4. draw semantic class and score
5. optionally fill the polygon with transparency

GT and prediction should be visualized together.

Recommended debug views:

```text
GT Polygon only
Prediction only
GT + Prediction overlay
Rasterized GT mask
Rasterized predicted mask
```

The final two raster views are for debugging and evaluation only in V1.

They do not imply Raster Loss is enabled.

Visualization must help identify:

- reversed vertex order
- wrong starting vertex
- self-intersection
- contour sampling errors
- coordinate normalization errors
- ROI clipping errors

---

## 22. Polygon Validity

A 32-point prediction does not automatically form a valid polygon.

Possible failures:

```text
self-intersection
duplicate points
collapsed area
extreme edge crossing
unordered vertices
```

V1 should not introduce complex topology loss immediately.

However, visualization and offline validation should measure polygon validity.

Recommended debug metrics:

```text
valid polygon ratio
self-intersection ratio
zero-area polygon ratio
mean polygon area
```

If invalid polygons become a major training problem, topology constraints may be considered in a later version.

Do not preemptively complicate V1 before observing this failure.

---

## 23. Dataset Scope

The initial dataset target is nuScenes-based occupancy data.

The expected source combination is:

```text
nuScenes sensor data
+
SparseDrive training metadata
+
semantic occupancy GT
```

A key engineering requirement is sample alignment.

The Polygon GT must align with the exact SparseDrive sample token/frame.

Required checks:

```text
scene
sample token
timestamp
ego pose
coordinate frame
BEV range
semantic class mapping
```

The conversion pipeline must never rely only on filename ordering.

Use stable dataset identifiers where available.

---

## 24. Coordinate System

All Polygon GT must be expressed in the coordinate system expected by the existing SparseDrive map head.

Claude Code must inspect:

- current map GT coordinate frame
- normalization range
- ROI definition
- x/y axis convention
- ego-relative transformation
- decoder denormalization

The Polygon OCC pipeline should reuse the existing map coordinate convention whenever possible.

Preferred design:

```text
Occupancy GT
    ↓
convert to SparseDrive map local BEV coordinate
    ↓
extract/resample polygons
    ↓
reuse existing map normalization
```

Avoid creating a second coordinate convention.

Coordinate inconsistencies are likely to produce training that technically runs but never converges.

---

## 25. Class Mapping

The Polygon OCC class set should not be copied blindly from Occ3D.

The first implementation must inspect the semantic occupancy category definitions and determine which classes are suitable for 2D polygon representation.

Classes naturally suitable for Polygon OCC include:

```text
drivable_surface
sidewalk
terrain
vegetation
vehicle
pedestrian
barrier
```

Potentially problematic classes include:

```text
unknown
free
noise
very sparse vertical structures
classes with fragmented occupancy
```

V1 should use a controlled subset of semantic classes.

The class set should satisfy:

- meaningful 2D BEV region
- enough training samples
- contour extraction is stable
- not dominated by tiny fragments

Class mapping must be stored in one configuration or dataset constant.

Avoid class mapping logic scattered across scripts.

---

## 26. Evaluation Philosophy

The original SparseDrive map metric may not fully represent Polygon OCC quality.

However, V1 should first prioritize:

```text
training stability
qualitative polygon quality
representation feasibility
```

Recommended V1 evaluation:

### Training Metrics

```text
classification loss
vertex regression loss
matching statistics
```

### Polygon Metrics

```text
polygon IoU after rasterization
per-class BEV IoU
valid polygon ratio
```

### Representation Statistics

```text
number of predicted polygons
number of GT polygons
points per polygon = 32
```

Rasterization may be used for **evaluation** even when it is not used for training.

This is important.

The V1 statement is:

```text
No Raster Loss
```

not:

```text
No Rasterization Anywhere
```

Rasterization remains useful for:

- visualization
- polygon IoU
- BEV semantic evaluation
- debugging

---

## 27. Development Milestones

### Milestone 1: Repository Analysis

Claude Code should:

- inspect SparseDrive map GT generation
- locate map dataset pipeline
- locate map head
- locate point encoder/decoder
- locate Hungarian assigner
- locate map loss
- locate map visualization/evaluation

Deliverable:

```text
short source dependency summary
```

Do not write a large source-analysis document.

The source code is the source of truth.

### Milestone 2: Polygon GT Offline Prototype

Implement a standalone conversion script:

```text
occupancy GT
→ semantic masks
→ contours
→ 32-point polygons
```

Before integrating into training, visualize at least several samples.

Deliverables:

```text
Polygon GT file or runtime output
visualization
statistics
```

### Milestone 3: Dataset Integration

Make the SparseDrive data pipeline provide Polygon GT.

Target shape:

```text
N × 32 × 2
```

Validate DataLoader output before training.

### Milestone 4: Model Dimension Modification

Change the map regression point count from the original map setting to 32.

Verify:

```text
prediction shape
target shape
matching shape
loss shape
decoder shape
```

### Milestone 5: Training Smoke Test

Run a small training test.

Success conditions:

```text
forward succeeds
backward succeeds
loss finite
no tensor shape error
matching works
```

### Milestone 6: Overfit Test

Train on a very small subset.

The model should visibly overfit Polygon GT.

This test is mandatory before a full training run.

### Milestone 7: Full Training

Only after the overfit test succeeds should full training begin.

---

## 28. Hard Constraints for Claude Code

Claude Code must follow these constraints.

### Constraint 1

Do not add a conventional dense OCC head.

### Constraint 2

Do not modify the detection branch unless required by a shared-interface bug.

### Constraint 3

Do not redesign SparseDrive.

### Constraint 4

Use fixed 32-point polygons.

### Constraint 5

Keep polygon closure implicit.

### Constraint 6

Normalize polygon vertex order.

### Constraint 7

Reuse Hungarian matching in V1.

### Constraint 8

Reuse vertex/point regression loss in V1 whenever technically possible.

### Constraint 9

Do not add Raster Loss in V1.

### Constraint 10

Rasterization is allowed for visualization and evaluation.

### Constraint 11

Prefer minimal modifications over clean-room rewrites.

### Constraint 12

Before modifying code, inspect the actual repository structure and call graph.

---

## 29. Non-Goals

Version 1 does not attempt to solve:

- full 3D occupancy
- voxel occupancy prediction
- height prediction
- Polygon + height extrusion
- unified detection/map/occupancy queries
- temporal polygon tracking
- polygon motion prediction
- differentiable rasterization
- raster supervision
- topology loss
- variable-length polygon decoding
- Bézier polygon representation
- Fourier descriptors
- implicit shape representation

These are outside V1 scope.

---

## 30. Future Direction

The long-term Polygon OCC roadmap is:

```text
V1
LineString → Polygon
Vertex Loss

        ↓

V2
Polygon Geometry Supervision
Chamfer / Hausdorff

        ↓

V3
Differentiable Rasterization
Vertex Loss + λ Raster Loss

        ↓

V4
Semantic Polygon Expansion
More OCC classes

        ↓

V5
Polygon + Height / Bottom Z
Sparse 3D semantic geometry

        ↓

V6
Unified Scene Representation
Detection + Map + Occupancy
```

The architecture should remain extensible toward these stages.

However, no future-stage feature should be added prematurely to V1.

---

## 31. Final Project Statement

Polygon OCC explores a sparse alternative to dense semantic occupancy.

Instead of predicting semantic labels for every grid or voxel cell, the model predicts a sparse set of closed semantic polygons.

For Version 1, SparseDrive's existing map branch is used as the implementation foundation.

The project makes one primary representation change:

```text
LineString-based Vector Map
            ↓
32-point Semantic Polygon
```

The first research question is:

> Can SparseDrive's sparse map queries directly learn semantic occupied regions represented as fixed-length closed polygons?

All implementation choices in Version 1 should serve this question.

Do not turn the project into a conventional OCC head implementation.

Do not redesign SparseDrive.

Keep the first experiment minimal, interpretable, and reproducible.
