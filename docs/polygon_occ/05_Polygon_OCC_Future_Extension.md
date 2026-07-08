# 05_Polygon_OCC_Future_Extension

## 1. Purpose of This Document

This document defines the future research and engineering roadmap for the **Polygon OCC** project after Version 1.

Version 1 has one primary objective:

> Replace SparseDrive's LineString-based map representation with fixed-length semantic polygons while preserving the sparse query architecture.

Version 1 intentionally avoids:

- Raster Loss
- Chamfer Loss
- Hausdorff Loss
- topology loss
- height prediction
- dense OCC heads
- unified detection/map/occupancy queries

This document describes what may be added **after the Polygon OCC V1 baseline is proven trainable**.

The future roadmap should remain staged.

Do not implement later-stage features inside V1 unless a V1-blocking failure requires a narrowly scoped fix.

---

## 2. Long-Term Roadmap

The recommended Polygon OCC roadmap is:

```text
V1
Sparse Polygon Representation
Vertex L1

        ↓

V2
Polygon Geometry Supervision
Chamfer / cyclic-aware losses

        ↓

V3
Differentiable Rasterization
Vertex Loss + λ Raster Loss

        ↓

V4
Expanded Semantic Polygon OCC
More occupancy categories

        ↓

V5
Polygon + Height / Bottom Z
Sparse 3D Semantic Geometry

        ↓

V6
Temporal Polygon OCC
Motion and temporal consistency

        ↓

V7
Unified Scene Primitive
Detection + Map + Occupancy
```

Each version should answer one clear research question.

---

## 3. V1 Baseline Recap

V1 representation:

```text
Query
├── class
├── confidence
└── 32 × (x, y)
```

Training:

```text
L_total
=
L_cls
+
λ_vertex L_vertex
```

Where:

```text
L_vertex = L1(pred_vertices, gt_vertices)
```

GT:

```text
Semantic Occupancy
        ↓
BEV class masks
        ↓
connected regions
        ↓
contours
        ↓
32-point Polygon GT
```

V1 should establish:

```text
trainability
matching stability
geometry predictability
sparse Polygon feasibility
```

All future changes should compare against this baseline.

---

# Part I — V2: Polygon Geometry Supervision

## 4. Why V2 Is Needed

Vertex L1 assumes point-to-point correspondence.

For a polygon:

```text
P0
P1
...
P31
```

Vertex L1 compares:

```text
pred P_i
↔
gt P_i
```

This is simple but has limitations.

Two polygons may have similar shape while sampled vertices do not align perfectly.

Example:

```text
GT:
P0 P1 P2 P3 ...

Prediction:
P0' P1' P2' P3' ...
```

A small cyclic shift can produce large L1.

Also, point-wise loss does not directly optimize global shape distance.

V2 should study shape-aware geometry supervision.

---

## 5. Candidate V2 Loss: Chamfer Distance

Chamfer Distance compares two point sets.

Conceptually:

```text
CD(P, G)
=
Σ_{p∈P} min_{g∈G} ||p-g||
+
Σ_{g∈G} min_{p∈P} ||g-p||
```

For Polygon OCC:

```text
Pred vertices: [32, 2]
GT vertices:   [32, 2]
```

Chamfer ignores exact vertex index correspondence.

This can reduce cyclic-shift sensitivity.

---

## 6. Chamfer Loss Advantages

Advantages:

- shape-aware
- point-order tolerant
- simple tensor implementation
- stays in vector space
- no rasterization required
- compatible with sparse Polygon representation

It is a natural V2 extension because it preserves:

```text
Vector Prediction
```

rather than introducing a raster domain.

---

## 7. Chamfer Loss Limitations

Chamfer also has weaknesses.

### Weak Topology Awareness

Chamfer treats vertices as a point set.

It does not understand polygon edge order.

A scrambled point sequence may have low Chamfer distance but form an invalid polygon.

Example:

```text
correct boundary points
+
wrong order
=
self-intersecting polygon
```

Chamfer may still be small.

### Density Sensitivity

With fixed 32-point uniform perimeter sampling, this issue is reduced but not eliminated.

### Outlier Behavior

A few badly placed points can affect one side of the loss strongly.

---

## 8. Recommended V2 Chamfer Strategy

Do not replace Vertex L1 immediately.

Use:

```text
L_geometry
=
L_vertex
+
λ_chamfer L_chamfer
```

This preserves ordered correspondence while adding global shape supervision.

Recommended ablation:

```text
A: Vertex only
B: Chamfer only
C: Vertex + Chamfer
```

The likely strongest V2 baseline is:

```text
Vertex + Chamfer
```

but this must be tested.

---

## 9. Candidate V2 Loss: Hausdorff Distance

Hausdorff Distance measures the worst nearest-neighbor discrepancy.

Conceptually:

```text
H(P, G)
=
max(
    max_{p∈P} min_{g∈G} d(p,g),
    max_{g∈G} min_{p∈P} d(g,p)
)
```

It emphasizes the largest geometric error.

This may help avoid isolated extreme polygon vertices.

---

## 10. Hausdorff Advantages and Risks

Advantages:

- sensitive to worst-case geometry
- useful for boundary quality
- may penalize extreme outliers strongly

Risks:

- unstable optimization
- dominated by one bad point
- noisy GT can cause large gradients
- less suitable as the only geometry loss

Recommended use:

```text
optional auxiliary loss
```

Do not make Hausdorff the main V2 geometry loss initially.

---

## 11. Candidate V2 Improvement: Cyclic-Invariant Vertex Loss

A Polygon is cyclic.

The same polygon can be represented as:

```text
[P0, P1, ..., P31]
```

or:

```text
[P1, P2, ..., P31, P0]
```

A cyclic-invariant loss computes:

```text
min over all cyclic shifts
```

Conceptually:

```text
L_cyclic(P, G)
=
min_k L1(P, roll(G, k))
```

Optionally also compare reversed order:

```text
min(
    forward cyclic shifts,
    reversed cyclic shifts
)
```

---

## 12. Why Cyclic-Invariant Loss Is Important

V1 solves point ordering by canonical GT normalization.

However, model predictions may still learn a geometrically correct polygon with another cyclic phase.

Canonical GT start-point rules can also be unstable near symmetric edges.

Therefore, cyclic-invariant loss may be more natural for polygons.

---

## 13. Cyclic-Invariant Loss Trade-Off

Advantages:

- respects polygon cyclic symmetry
- reduces arbitrary start-point dependence
- remains vector-based
- directly supervises ordered polygon geometry

Disadvantages:

- increases matching/loss computation
- 32 shifts per pred/GT pair
- may interact with Hungarian matching
- reverse-direction handling adds complexity

Recommended research question:

> Is canonical point ordering sufficient, or does cyclic-invariant supervision significantly improve Polygon OCC?

This is a strong V2 ablation.

---

## 14. V2 Matching Options

V1 matching:

```text
classification cost
+
direct vertex L1 cost
```

V2 can explore:

### Option A

Keep matching unchanged.

Add Chamfer only to training loss.

This is the cleanest V2 extension.

### Option B

Use Chamfer in Hungarian cost.

Matching becomes more shape-aware.

### Option C

Use cyclic-invariant vertex cost.

This directly addresses polygon symmetry.

Recommended order:

```text
A first
then B/C as separate experiments
```

Do not change loss and matching simultaneously in the first V2 experiment.

---

# Part II — V3: Rasterization and Area Supervision

## 15. Why Rasterization Is Valuable

Polygon OCC predicts area geometry.

Vertex losses supervise boundary points.

However, occupancy-style quality is fundamentally area-oriented.

Example:

```text
GT Polygon:
large semantic region

Prediction:
similar vertices but region slightly distorted
```

Vertex L1 may be small.

Yet:

```text
occupied-area overlap
```

may be significantly worse.

Rasterization converts polygons into masks.

```text
Polygon
    ↓
Rasterization
    ↓
BEV semantic mask
```

This enables region-level supervision.

---

## 16. Core V3 Loss

The main V3 design is:

```text
L_total
=
L_cls
+
λ_vertex L_vertex
+
λ_raster L_raster
```

This is the previously discussed:

> **Vertex Loss + λ · Raster Loss**

The two losses supervise complementary properties.

---

## 17. Role of Vertex Loss

Vertex Loss supervises:

```text
ordered geometry
boundary coordinates
local point correspondence
```

It encourages:

```text
precise vector output
```

This is important because Polygon OCC should remain useful as a vector representation.

---

## 18. Role of Raster Loss

Raster Loss supervises:

```text
occupied area
region overlap
overall shape consistency
```

It encourages:

```text
correct semantic region coverage
```

This makes Polygon OCC more occupancy-like.

---

## 19. Why Vertex + Raster Is Stronger Than Raster Alone

Raster-only supervision may permit many different polygon vertex configurations that generate similar masks.

Example:

```text
Polygon A:
well-distributed vertices

Polygon B:
irregular clustered vertices
```

Both may rasterize similarly.

Raster Loss does not directly preserve good vector geometry.

Vertex Loss provides explicit geometry structure.

Therefore:

```text
Vertex + Raster
```

is preferred over:

```text
Raster only
```

for Polygon OCC.

---

## 20. Why Vertex + Raster Is Stronger Than Vertex Alone

Vertex-only supervision does not directly measure region overlap.

Two polygons with moderate boundary errors may have a large area mismatch.

Raster Loss directly supervises the semantic region.

Therefore:

```text
Vertex + Raster
```

provides:

```text
vector geometry
+
occupancy area
```

This is the main V3 research hypothesis.

---

## 21. Differentiable Rasterization Requirement

For Raster Loss to train the polygon vertices:

```text
predicted vertices
    ↓
rasterizer
    ↓
mask
    ↓
loss
```

the rasterization path must be differentiable with respect to vertex coordinates.

A normal OpenCV call such as:

```python
cv2.fillPoly(...)
```

is not differentiable.

It is suitable for:

```text
visualization
evaluation
```

but not Raster Loss training.

V3 requires a differentiable rasterization strategy.

---

## 22. Candidate Differentiable Rasterization Strategies

Potential approaches include:

### Soft Edge Distance

For every BEV cell:

```text
distance to polygon edges
```

Use a smooth function to estimate inside probability.

### Soft Winding / Inside Test

Approximate point-in-polygon membership using differentiable angular or winding formulations.

### Triangle-Based Rasterization

Triangulate the polygon and use a differentiable mesh rasterizer.

### Existing Differentiable Rasterizer

Adopt a tested differentiable vector-to-raster implementation if compatible.

The implementation choice should be evaluated for:

```text
stability
speed
memory
polygon validity behavior
```

---

## 23. V3 Raster Resolution

Raster resolution is a new hyperparameter.

Example:

```text
100 × 100
200 × 200
400 × 400
```

Higher resolution:

```text
better boundary precision
higher memory
higher compute
```

Lower resolution:

```text
faster
more aliasing
weaker small-object supervision
```

Raster resolution must be tied to the BEV ROI.

Example:

```text
ROI = 100 m × 100 m
resolution = 0.5 m
→ 200 × 200
```

The project should report metric resolution, not only tensor size.

---

## 24. Candidate Raster Losses

### Dice Loss

Good for region overlap.

Useful when positive regions are sparse.

### Binary Cross Entropy

Simple per-pixel supervision.

### Focal Loss

Useful for severe positive/negative imbalance.

### Semantic Cross Entropy

Useful if generating one exclusive semantic raster.

However, Polygon OCC uses overlapping per-class regions.

Therefore, multi-label per-class raster losses may be more natural.

---

## 25. Preferred V3 Raster Supervision

Because Polygon OCC allows overlap:

```text
road polygon
+
vehicle polygon
```

the preferred raster target is:

```text
C independent binary semantic masks
```

Output for evaluation/training:

```text
[C, H, W]
```

Each class channel is independent.

Recommended first V3 loss:

```text
Dice
```

or:

```text
Dice + BCE
```

per class.

Do not force exclusive softmax semantics unless the representation design changes.

---

## 26. Per-Query Raster vs Aggregated Per-Class Raster

There are two major designs.

### Design A: Per-Query Raster Loss

Each matched predicted polygon is rasterized and compared with its matched GT polygon.

```text
pred query i
↔
GT polygon j
```

Raster Loss is instance-level.

Advantages:

- compatible with Hungarian matching
- direct matched supervision

Disadvantages:

- region decomposition errors matter
- duplicate predictions not directly penalized globally

### Design B: Aggregated Per-Class Raster Loss

Rasterize all predicted polygons of one class.

Aggregate them into one semantic mask.

Compare with class GT mask.

Advantages:

- directly supervises scene occupancy
- tolerant to region decomposition differences

Disadvantages:

- differentiable aggregation required
- classification confidence integration required
- duplicate polygons may still overlap harmlessly
- harder attribution to individual queries

---

## 27. Recommended V3 Sequence

Start with:

```text
Per-query matched Raster Loss
```

because it fits the existing Hungarian pipeline.

Then explore:

```text
Aggregated class-level Raster Loss
```

as a stronger occupancy-oriented extension.

Recommended experiments:

```text
V3-A:
Vertex + matched Polygon Dice

V3-B:
Vertex + aggregated class Dice

V3-C:
Vertex + both
```

---

## 28. Aggregating Polygon Masks

For class `c`, predicted query masks may be:

```text
M_1
M_2
...
M_k
```

A differentiable union can use:

```text
1 - Π_i (1 - p_i)
```

where each `p_i` is a soft occupancy probability.

This is analogous to probabilistic union.

Classification score may modulate each mask:

```text
p_i = score_i(c) × raster_mask_i
```

Then:

```text
class_mask_c
=
1 - Π_i (1 - p_i)
```

This is a future design direction.

It should not be introduced in the first Raster Loss experiment without ablation.

---

## 29. Raster Loss and Polygon Invalidity

Differentiable rasterization may behave unpredictably for:

```text
self-intersecting polygons
```

Possible responses:

- soft rasterizer still produces a mask
- rasterizer fails
- inside/outside is ambiguous

Therefore, V3 should first measure V1/V2 invalid polygon ratio.

If invalidity is frequent, V3 may require:

```text
topology regularization
```

or a representation change.

---

# Part III — Polygon Topology and Validity

## 30. Why Topology May Become a Future Problem

A sequence of 32 points does not guarantee a simple polygon.

Possible predictions:

```text
bow-tie
edge crossing
point collapse
duplicate vertices
very thin loops
```

Vertex L1 may reduce these problems near good GT.

However, early training can produce invalid geometry.

---

## 31. Candidate Topology Loss: Edge Intersection Penalty

For non-adjacent polygon edges:

```text
edge_i
edge_j
```

detect or softly estimate intersection.

Penalize crossing edges.

Conceptually:

```text
L_intersection
=
sum crossing_probability(edge_i, edge_j)
```

Challenge:

```text
O(N²) edge pairs
```

With 32 vertices, this may still be manageable.

But differentiable segment intersection is nontrivial.

---

## 32. Candidate Topology Loss: Edge-Length Regularization

Penalize extreme adjacent edge lengths.

Example:

```text
L_edge
=
variance(edge_lengths)
```

This may encourage more uniform vertex distribution.

However, semantic polygons may naturally contain unequal local curvature.

Uniform perimeter GT already provides roughly equal spacing.

A mild edge-length regularizer may be reasonable.

Do not overconstrain shape.

---

## 33. Candidate Topology Loss: Area Regularization

Compare predicted and GT polygon area.

Using shoelace formula:

```text
A =
1/2 |Σ x_i y_{i+1} - x_{i+1} y_i|
```

Loss:

```text
L_area
=
|A_pred - A_gt|
```

Advantages:

- differentiable
- vector-based
- cheap

Limitations:

- same area does not imply same shape
- self-intersection may produce misleading signed area

Area Loss may be a lightweight intermediate step between V2 and V3.

---

## 34. Candidate Boundary Smoothness Loss

Penalize rapid direction changes.

Using consecutive edges:

```text
e_i = P_{i+1} - P_i
```

Measure angular variation.

This may help noisy vegetation contours.

But it can oversmooth:

```text
vehicle corners
barriers
sharp road boundaries
```

Therefore, class-aware or adaptive smoothness may be needed.

Not recommended as an early default.

---

# Part IV — Expanded Semantic Polygon OCC

## 35. V4 Objective

V1 should begin with a controlled semantic class subset.

V4 expands Polygon OCC toward a broader occupancy representation.

Potential categories:

```text
drivable_surface
sidewalk
terrain
vegetation
vehicle
pedestrian
barrier
bicycle
motorcycle
truck
bus
construction
other semantic regions
```

Class selection should follow GT quality and polygon suitability.

---

## 36. Class Granularity Decision

Vehicle categories may be:

### Merged

```text
car
truck
bus
trailer
→ vehicle
```

Advantages:

- more samples per class
- simpler first semantic representation

### Separated

```text
car
truck
bus
trailer
```

Advantages:

- richer semantics
- closer to occupancy taxonomy

V1/V4 should compare class complexity with available query capacity.

---

## 37. Which Classes Are Naturally Polygon-Friendly

Highly suitable:

```text
road
sidewalk
terrain
building footprint
vehicle footprint
barrier region
```

Moderately suitable:

```text
vegetation
pedestrian
bicycle
```

Potentially difficult:

```text
thin poles
traffic signs
very sparse structures
unknown
free space
```

Polygon OCC does not need to represent every occupancy class equally well.

The class taxonomy may be adapted to the representation.

---

## 38. Representation-Specific Class Taxonomy

Dense OCC taxonomy is voxel-centric.

Polygon OCC is region-centric.

Therefore, a direct one-to-one label copy is not mandatory.

A future Polygon OCC taxonomy may merge classes that have similar BEV region semantics.

Example:

```text
car
truck
bus
trailer
→ dynamic_vehicle_region
```

The scientific question is:

> What semantic taxonomy is appropriate for sparse region primitives?

This may become its own research contribution.

---

# Part V — V5: Sparse 3D Polygon OCC

## 39. Limitation of 2D Polygon OCC

V1-V4 represent:

```text
(x, y)
```

regions.

They cannot distinguish vertical geometry.

Example:

```text
building footprint
```

does not specify:

```text
height
```

A vehicle and a truck may have similar BEV footprint but different vertical occupancy.

Therefore, 2D Polygon OCC cannot fully replace 3D semantic occupancy.

---

## 40. Polygon + Height Representation

A natural extension is:

```text
Query
├── class
├── score
├── polygon [32, 2]
├── bottom_z
└── height
```

Equivalent:

```text
z_min
z_max
```

Then:

```text
2D polygon
    ↓
extrude between z_min and z_max
    ↓
3D occupied volume
```

This is a sparse prism representation.

---

## 41. Suggested Name for V5

Possible terms:

```text
3D Polygon OCC
Prismatic OCC
Sparse Semantic Prism Representation
Polygon Prism Occupancy
```

Project-level naming can still remain:

```text
Polygon OCC
```

with:

```text
2D Polygon OCC
3D Polygon OCC
```

as variants.

---

## 42. 3D GT Generation

From 3D semantic occupancy:

```text
connected semantic 3D component
        ↓
BEV footprint polygon
        ↓
z_min
z_max
```

For a semantic region:

```text
footprint = BEV projection
bottom_z = lowest occupied voxel
top_z = highest occupied voxel
height = top_z - bottom_z
```

This is simple but assumes roughly column-consistent occupancy.

---

## 43. Limitation of Single Height Per Polygon

Consider:

```text
tree
sloped terrain
building with multiple roof levels
```

One height value is insufficient.

Possible extensions:

### Per-Vertex Height

```text
32 × (x, y, z_top)
```

plus bottom surface.

### Height Grid Inside Polygon

Hybrid sparse-dense representation.

### Multiple Vertical Layers

```text
polygon + several z intervals
```

### 3D Mesh

Much more complex.

V5 should start with:

```text
polygon + z_min + z_max
```

as the minimal 3D extension.

---

## 44. 3D Rasterization / Voxelization

A Polygon Prism can be voxelized.

```text
Polygon
+
z range
    ↓
Voxelization
    ↓
3D semantic occupancy
```

This enables comparison with Occ3D-style metrics.

The model remains sparse in native representation.

Dense voxels are generated only for:

```text
evaluation
optional supervision
downstream use
```

This is a strong long-term research story.

---

# Part VI — V6: Temporal Polygon OCC

## 45. Why Temporal Modeling Matters

SparseDrive already uses temporal sparse representations.

Polygon OCC semantic regions may evolve across frames.

Examples:

```text
vehicle moves
pedestrian moves
road region remains static
vegetation remains static
```

Temporal consistency can improve stability.

---

## 46. Static and Dynamic Polygon Regions

A future semantic taxonomy may distinguish:

```text
static regions
dynamic regions
```

Static:

```text
road
sidewalk
terrain
vegetation
building
```

Dynamic:

```text
vehicle
pedestrian
bicycle
```

Temporal update behavior may differ.

---

## 47. Dynamic Polygon Motion

A dynamic Polygon query may predict:

```text
velocity
yaw / orientation
future polygon sequence
```

Conceptually:

```text
Polygon_t
+
motion
    ↓
Polygon_{t+1}
```

This connects Polygon OCC with motion forecasting.

---

## 48. Temporal Polygon Consistency Loss

For a matched static region:

```text
Polygon_t
```

transform it using ego motion into frame `t+1`.

Compare with:

```text
Polygon_{t+1}
```

Possible loss:

```text
temporal Chamfer
temporal raster IoU
vertex consistency
```

This may stabilize static semantic regions.

---

## 49. Split and Merge Problem

Semantic regions may split or merge between frames.

Example:

```text
vegetation component
road visible region
occupancy truncation at ROI boundary
```

Therefore, persistent Polygon identity is not always well-defined.

A future temporal design must distinguish:

```text
physical instance identity
```

from:

```text
semantic region decomposition identity
```

This is a key research challenge.

---

# Part VII — V7: Unified Scene Representation

## 50. Long-Term Unified Query Idea

SparseDrive currently uses separate task representations.

Conceptually:

```text
Detection Queries
Map Queries
Motion Queries
Planning
```

Polygon OCC introduces:

```text
Semantic Polygon Queries
```

A long-term idea is to unify scene primitives.

---

## 51. Unified Semantic Region Query

One query could predict:

```text
class
polygon
height
motion
confidence
```

Examples:

```text
road
vehicle
pedestrian
vegetation
building
```

The representation becomes:

```text
Sparse Semantic Geometry Set
```

instead of separate:

```text
boxes
polylines
voxels
```

---

## 52. Why Unified Queries Are Attractive

Today:

```text
Detection
→ box

Map
→ polyline

Occupancy
→ voxel
```

Three tasks use different geometric primitives.

Unified Polygon OCC asks:

> Can semantic geometry use one common sparse primitive?

Potential benefits:

- common decoder
- shared query semantics
- easier planning consumption
- reduced redundant representation
- direct geometric reasoning

---

## 53. Why Unified Queries Are Difficult

Different semantic classes have different geometry.

Vehicle:

```text
compact object
```

Road:

```text
large region
```

Pedestrian:

```text
small region
```

Vegetation:

```text
irregular region
```

A fixed 32-point polygon may not be equally efficient for all.

Other problems:

- different query counts
- instance vs region semantics
- class imbalance
- temporal identity
- motion supervision
- matching rules

Therefore, unified queries are a long-term direction.

---

## 54. Possible Hybrid Unified Representation

A future query could predict a geometry type.

```text
Query
├── semantic class
├── geometry type
└── geometry parameters
```

Geometry types:

```text
point
box
polyline
polygon
prism
```

This is more flexible but far beyond V1.

It may evolve into a general sparse scene grammar.

---

# Part VIII — Representation Improvements

## 55. Variable-Length Polygon Decoding

Fixed 32 points are simple but inefficient.

A future model may predict variable vertex count.

Possible design:

```text
autoregressive vertex decoder
```

Output:

```text
P0
P1
...
STOP
```

Advantages:

- adaptive complexity
- fewer vertices for simple regions

Disadvantages:

- sequence decoding latency
- training instability
- vertex-order dependence
- harder Hungarian cost

Not recommended before fixed-length Polygon OCC is validated.

---

## 56. Polygon Vertex Validity Mask

A non-autoregressive compromise:

```text
maximum 64 vertices
+
per-vertex validity score
```

Each query predicts:

```text
64 candidate vertices
64 validity logits
```

The effective polygon length varies.

Challenges:

```text
maintaining sequence continuity
invalid gaps
matching variable-length GT
```

This is a possible future extension.

---

## 57. Bézier Representation

Polygon boundary may be represented using Bézier control points.

Example:

```text
8 control points
    ↓
smooth closed curve
```

Advantages:

- compact
- smooth
- fewer parameters

Risks:

- difficult sharp corners
- curve closure constraints
- control-point correspondence
- vehicle rectangles may be inefficient

Potentially useful for:

```text
road
vegetation
curved semantic regions
```

---

## 58. B-Spline Representation

B-Splines provide flexible smooth closed curves.

Possible output:

```text
control points
+
fixed knot structure
```

Advantages:

- smooth
- compact
- differentiable

Challenges:

- periodic spline setup
- topology
- sharp corners
- more complex decoder semantics

---

## 59. Fourier Descriptor Representation

A closed contour can be represented in the frequency domain.

Conceptually:

```text
Polygon boundary
    ↓
complex sequence x + iy
    ↓
Fourier transform
    ↓
K coefficients
```

The model predicts coefficients.

Advantages:

- compact global shape representation
- naturally closed
- different complexity control through frequency count

Risks:

- less interpretable
- more difficult integration with SparseDrive
- local boundary errors can be hard to control

This is a strong alternative research direction after Polygon OCC.

---

## 60. Signed Distance Function

Instead of explicit vertices, a query may predict an implicit shape function.

```text
f(x, y)
```

with:

```text
inside  < 0
boundary = 0
outside > 0
```

This can represent complex topology.

However, it becomes closer to implicit dense shape modeling.

It weakens the minimal sparse vector interpretation.

Not a near-term extension.

---

# Part IX — Evaluation Roadmap

## 61. V1 Evaluation

V1 should report:

```text
training convergence
vertex regression loss
classification metrics
Polygon GT approximation quality
Polygon IoU
BEV semantic IoU
valid polygon ratio
self-intersection ratio
```

The most important qualitative result is:

```text
predicted semantic polygons
```

---

## 62. V2 Evaluation

Compare:

```text
Vertex
Chamfer
Vertex + Chamfer
cyclic-aware loss
```

Metrics:

```text
Polygon IoU
boundary Chamfer distance
Hausdorff distance
valid polygon ratio
```

The key question:

> Does shape-aware vector supervision improve polygon geometry?

---

## 63. V3 Evaluation

Compare:

```text
Vertex only
Raster only
Vertex + Raster
```

Recommended metrics:

```text
BEV semantic mIoU
Polygon IoU
boundary distance
GT occupancy vs predicted rasterized Polygon
```

The key question:

> Does area supervision improve occupancy quality without damaging vector geometry?

---

## 64. Efficiency Evaluation

Polygon OCC should be evaluated as a sparse representation.

Measure:

```text
model FLOPs
GPU memory
training memory
inference latency
number of queries
output size
```

Compare output representation size.

Traditional BEV occupancy:

```text
H × W × C
```

Polygon OCC:

```text
N × 32 × 2
+
N class logits
```

This representation-size comparison is an important part of the research story.

---

## 65. Output Storage Comparison

Example conceptual comparison:

Dense semantic BEV:

```text
200 × 200 × 16 classes
```

Polygon OCC:

```text
100 queries × 32 × 2
```

The exact storage comparison depends on:

```text
dtype
class logits
confidence
```

A paper should report actual byte size rather than only element counts.

---

## 66. Planning-Oriented Evaluation

Polygon OCC may be more directly useful for planning.

Possible downstream tests:

```text
point-in-polygon drivable query
distance to semantic boundary
collision with vehicle polygon
road-region overlap
trajectory semantic intersection
```

These operations are natural with polygons.

Future work may compare planning utility against dense occupancy.

---

## 67. Boundary Quality Evaluation

Because Polygon OCC is a vector representation, boundary metrics matter.

Potential metrics:

```text
Chamfer distance
Hausdorff distance
Average Symmetric Surface Distance
boundary F-score
```

Raster mIoU alone does not fully capture vector quality.

---

## 68. Region Decomposition Evaluation

Polygon OCC predicts a set of regions.

Possible metrics:

```text
GT region recall
predicted region precision
matched region IoU
number of region splits
number of region merges
```

These may reveal failure modes hidden by aggregate semantic mIoU.

---

# Part X — Ablation Study Plan

## 69. Core Ablation A: Representation

Compare:

```text
A1: Original SparseDrive LineString Map
A2: Polygon OCC 32-point Vertex Loss
```

This is not a direct task-equivalent comparison if classes differ.

Therefore, the comparison should focus on architecture feasibility and representation behavior.

A stronger comparison may use the same region classes represented as:

```text
boundary LineString
vs
area Polygon
```

if a fair GT design is possible.

---

## 70. Core Ablation B: Number of Vertices

Compare:

```text
16 points
32 points
64 points
```

Metrics:

```text
GT approximation IoU
prediction IoU
latency
memory
vertex loss
```

Expected trade-off:

```text
more points
→ better shape capacity
→ harder regression
```

32 is the V1 default.

---

## 71. Core Ablation C: Simplification

Compare:

```text
no simplification
light Douglas-Peucker
strong Douglas-Peucker
```

Measure:

```text
GT polygon approximation
training convergence
prediction quality
```

The goal is to determine whether cleaner contours improve learning.

---

## 72. Core Ablation D: Vertex Ordering

Compare:

```text
canonical start + fixed direction
cyclic-invariant L1
canonical + Chamfer
```

This directly studies Polygon parameterization.

---

## 73. Core Ablation E: Geometry Loss

Compare:

```text
Vertex
Chamfer
Vertex + Chamfer
Vertex + Hausdorff
```

Keep all other settings identical.

---

## 74. Core Ablation F: Raster Supervision

Compare:

```text
Vertex
Raster
Vertex + Raster
Vertex + Chamfer + Raster
```

This is the key V3 ablation.

---

## 75. Core Ablation G: Query Count

Use GT statistics to select values.

Example:

```text
50
100
150
```

Measure:

```text
GT match coverage
recall
latency
memory
```

This studies sparse representation capacity.

---

## 76. Core Ablation H: Class Set

Compare:

```text
static region classes only
static + vehicle
full selected semantic set
```

This shows how Polygon OCC scales with semantic complexity.

---

## 77. Core Ablation I: BEV Projection Policy

Compare:

```text
per-class independent masks
exclusive priority projection
```

The expected benefit of independent masks is preserving layered semantics.

This should be empirically validated.

---

# Part XI — Paper Positioning

## 78. Core Paper Story

A concise research story is:

> Existing semantic occupancy methods typically discretize space into dense grids or voxels. Polygon OCC explores a sparse vectorized alternative that represents semantic occupied regions as a set of closed polygons predicted by sparse queries.

SparseDrive is used as the architectural foundation because it already supports sparse query-based geometric prediction.

---

## 79. Core Representation Contrast

Traditional occupancy:

```text
Cell-Centric Representation

Scene
    ↓
Dense Grid / Voxels
    ↓
Semantic label per cell
```

Polygon OCC:

```text
Region-Centric Representation

Scene
    ↓
Semantic regions
    ↓
Sparse polygon set
```

The key term is:

> **region-centric occupancy representation**

This may be useful in the paper narrative.

---

## 80. Possible Method Description

Conceptual method paragraph:

> Polygon OCC converts dense semantic occupancy annotations into a sparse set of semantic region polygons. Each region is represented by a fixed-length sequence of 32 perimeter-sampled vertices. We reuse SparseDrive's sparse map queries to predict polygon classes and ordered vertices, preserving set-based Hungarian assignment and point regression supervision. The resulting representation remains vectorized while allowing direct rasterization into BEV semantic occupancy.

This is a conceptual description and should be adapted after implementation.

---

## 81. Possible Contribution Structure

Potential contributions:

### Contribution 1

A sparse polygon-based occupancy representation for semantic BEV regions.

### Contribution 2

An occupancy-to-polygon vectorization pipeline that converts dense semantic GT into fixed-length semantic region primitives.

### Contribution 3

A minimal sparse-query adaptation of SparseDrive for semantic polygon prediction.

### Contribution 4

Future V3 only:

A joint vertex and raster supervision strategy bridging vector geometry and occupancy area.

Do not claim Contribution 4 in a V1 paper unless Raster Loss is actually implemented.

---

## 82. Important Paper Honesty

Do not claim:

```text
full 3D occupancy
```

for 2D Polygon OCC.

Do not claim:

```text
voxel-free 3D scene understanding
```

until height or 3D geometry is modeled.

V1 should be described as:

```text
BEV semantic occupancy representation
```

or:

```text
polygon-based semantic BEV occupancy
```

This is more accurate.

---

## 83. Naming Recommendations

Project:

```text
Polygon OCC
```

Method variants:

```text
Polygon OCC V1
Polygon OCC + Chamfer
Polygon OCC + Raster
3D Polygon OCC
```

Representation:

```text
Polygon Occupancy Representation
```

Output:

```text
Semantic Polygon Set
```

Primitive:

```text
Semantic Polygon Region
```

Head:

```text
Polygon Head
```

Internal code may still reuse map naming in V1.

---

## 84. Possible Paper Titles

Conceptual title ideas:

```text
Polygon OCC: Sparse Polygon-Based Semantic Occupancy for Autonomous Driving
```

```text
Polygon OCC: A Sparse Vectorized Representation for Semantic BEV Occupancy
```

```text
From Voxels to Polygons: Sparse Semantic Occupancy via Polygon Queries
```

```text
Region-Centric Semantic Occupancy with Sparse Polygon Queries
```

These are working-title ideas.

Final title should depend on actual experiments and scope.

---

# Part XII — Experimental Sequence Recommendation

## 85. Recommended Research Sequence

### Experiment 0

GT vectorization only.

Measure:

```text
Occ mask
vs
32-point Polygon GT
```

Goal:

Determine representation ceiling.

---

### Experiment 1

Polygon OCC V1.

```text
Vertex Loss only
```

Goal:

Prove trainability.

---

### Experiment 2

16 / 32 / 64 point ablation.

Goal:

Select representation capacity.

---

### Experiment 3

Canonical ordering vs cyclic-invariant loss.

Goal:

Study Polygon parameterization.

---

### Experiment 4

Vertex + Chamfer.

Goal:

Improve shape quality.

---

### Experiment 5

Differentiable rasterizer prototype.

Goal:

Validate stable gradient flow.

---

### Experiment 6

Vertex + Raster.

Goal:

Improve semantic occupancy overlap.

---

### Experiment 7

Expanded class set.

Goal:

Test semantic scalability.

---

### Experiment 8

Polygon + Height.

Goal:

Explore sparse 3D occupancy.

---

## 86. Do Not Skip Experiment 0

The first number that should be known is:

```text
How well can 32-point polygons reconstruct the GT occupancy masks?
```

If GT vectorization quality is poor:

```text
model improvement cannot fix representation capacity
```

Possible responses:

```text
increase point count
split large regions
change simplification
change class set
```

This must be resolved before interpreting model metrics.

---

## 87. Do Not Skip the Overfit Experiment

Before full training:

```text
small subset
+
many iterations
```

The model should memorize Polygon GT.

If it cannot:

```text
do not tune full-training hyperparameters
```

First debug:

```text
ordering
matching
normalization
anchor initialization
```

---

## 88. Research Risk Register

### Risk 1: Polygon Ordering Instability

Symptoms:

```text
high Vertex Loss
geometrically correct but phase-shifted prediction
```

Mitigation:

```text
canonical ordering
cyclic-invariant loss
Chamfer
```

---

### Risk 2: Polygon Self-Intersection

Symptoms:

```text
invalid geometry
rasterizer failure
```

Mitigation:

```text
measure first
topology regularization later
```

---

### Risk 3: Too Many Semantic Regions

Symptoms:

```text
query capacity insufficient
low region recall
```

Mitigation:

```text
GT statistics
increase query count
region filtering
```

---

### Risk 4: 32 Points Insufficient

Symptoms:

```text
low GT approximation IoU
large-region distortion
```

Mitigation:

```text
64 points
region splitting
adaptive representation
```

---

### Risk 5: Occupancy Projection Loses Vertical Semantics

Symptoms:

```text
class overlap ambiguity
incorrect BEV meaning
```

Mitigation:

```text
per-class masks
z filtering
future 3D Polygon OCC
```

---

### Risk 6: Line-Based Map Priors Hurt Polygon Learning

Symptoms:

```text
prediction collapse
slow convergence
line-like polygons
```

Mitigation:

```text
new Polygon anchors
reinitialize map geometry layers
```

---

### Risk 7: Raster Loss Dominates Vertex Geometry

Symptoms:

```text
good mask IoU
poor vertex distribution
unstable polygon boundaries
```

Mitigation:

```text
loss weight tuning
retain Vertex Loss
monitor boundary metrics
```

---

### Risk 8: Class Fragmentation

Symptoms:

```text
hundreds of tiny vegetation polygons
```

Mitigation:

```text
minimum-area filtering
morphology
class selection
region merging policy
```

---

## 89. Future Engineering Principles

Future versions should preserve several rules.

### Rule 1

Keep V1 baseline runnable.

### Rule 2

Add one major change per experiment.

### Rule 3

Use separate experiment configs.

### Rule 4

Never overwrite GT caches without versioning.

### Rule 5

Measure representation error independently from model error.

### Rule 6

Preserve sparse-native output even if rasterization is added.

### Rule 7

Do not call a dense intermediate the primary Polygon OCC representation.

---

## 90. Versioned Documentation Policy

When a future stage is implemented, create a new design note.

Recommended:

```text
06_Polygon_OCC_V2_Geometry_Loss.md
07_Polygon_OCC_V3_Raster_Loss.md
08_Polygon_OCC_3D_Extension.md
```

Do not silently modify V1 assumptions.

The historical design path is useful for:

```text
ablation
debugging
paper writing
reproducibility
```

---

## 91. Recommended Claude Code Behavior for Future Versions

Before implementing a future feature:

1. read V1 documentation
2. identify which V1 constraint is intentionally being relaxed
3. inspect current implementation
4. create a separate experiment config
5. implement only the new feature
6. add focused tests
7. compare against the previous version

Example:

```text
V3 Raster Loss
```

must explicitly relax:

```text
"No Raster Loss in V1"
```

while preserving:

```text
Sparse Polygon native output
```

---

## 92. Final Long-Term Vision

Polygon OCC begins with a simple representation change:

```text
LineString
    ↓
Closed Polygon
```

The long-term vision is broader:

```text
Dense Cell-Centric Occupancy
            ↓
Sparse Region-Centric Occupancy
```

Instead of representing the world as millions of independent cells, the scene is represented as a sparse set of semantic geometric regions.

A future scene may be described as:

```text
Region 0
class = road
polygon = [...]
height = ...

Region 1
class = vehicle
polygon = [...]
height = ...
motion = ...

Region 2
class = vegetation
polygon = [...]
height = ...

...
```

This representation can potentially support:

```text
semantic understanding
occupancy reconstruction
collision reasoning
map understanding
planning
motion reasoning
```

using one sparse geometric scene description.

---

## 93. Final Research Principle

The Polygon OCC project should not begin by trying to solve every problem.

The correct sequence is:

```text
prove representation
        ↓
improve geometry
        ↓
supervise area
        ↓
expand semantics
        ↓
recover 3D
        ↓
add temporal behavior
        ↓
explore unified scene primitives
```

V1 is successful if fixed-length semantic polygons can be learned reliably by SparseDrive sparse map queries.

V2 is successful if vector geometry supervision improves shape quality.

V3 is successful if Raster Loss improves occupancy overlap while preserving vector quality.

Later versions should only be pursued after the previous representation is well understood.

The central long-term hypothesis is:

> Semantic occupancy does not necessarily need to be represented natively as a dense voxel or grid tensor. A sparse set of semantic polygons may provide a compact, geometric, and planning-friendly alternative for representing occupied semantic regions.

Polygon OCC should be developed as a staged investigation of that hypothesis.
