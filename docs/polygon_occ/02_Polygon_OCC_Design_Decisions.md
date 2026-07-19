# 02_Polygon_OCC_Design_Decisions

## 1. Purpose of This Document

This document records the major technical and architectural decisions for the **Polygon OCC** project.

It is intentionally different from a source-code analysis document.

The SparseDrive repository itself is the source of truth for:

- class names
- file names
- call graph
- module dependencies
- exact tensor layouts
- configuration structure

This document instead answers:

- What should be changed?
- Why should it be changed?
- What should remain unchanged?
- Which alternatives were considered?
- Which ideas are intentionally postponed?

Claude Code should use this document as the **design intent and constraint reference** before implementing modifications.

---

## 2. Core Decision Summary

The core V1 decisions are:

> Workspace note: the current active implementation baseline in this repository
> has been reduced to **20 points per polygon** for practical memory reasons.
> The table below records the original V1 design intent.

| Topic | V1 Decision |
|---|---|
| Base model | SparseDrive |
| Modified branch | Existing map branch |
| Representation | Closed semantic polygon |
| Points per polygon | Fixed 32 |
| Output dimensionality | 32 × 2 = 64 |
| Polygon closure | Implicit, connect P31 to P0 |
| GT source | Semantic occupancy annotations, preferably Occ3D/OpenOcc style |
| BEV conversion | Per-class independent masks |
| Region unit | Connected semantic region |
| Matching | Reuse existing Hungarian framework |
| Regression | Reuse point/vertex regression |
| Main loss | Classification + Vertex L1 |
| Rasterization in training | Disabled in V1 |
| Rasterization in evaluation | Allowed |
| Chamfer / Hausdorff | Deferred |
| Dense OCC head | Forbidden in V1 |
| Detection branch changes | Avoid |
| Main research question | Can sparse map queries learn semantic occupied polygons? |

Every implementation decision should be consistent with this table.

---

## 3. Decision: Use Polygon Instead of LineString

### 3.1 Original Representation

The original SparseDrive map branch represents map elements as ordered point sequences:

```text
P0 → P1 → P2 → ... → Pn
```

The sequence is interpreted as a LineString or polyline.

This representation is suitable for line-like map primitives:

```text
divider
boundary
lane line
road divider
```

A line primarily models geometry along one dimension.

---

### 3.2 Target Representation

Polygon OCC changes the primitive to a closed polygon:

```text
P0 → P1 → P2 → ... → P31
↑                       ↓
└───────────────────────┘
```

The polygon represents an area.

The representation changes from:

```text
curve
```

to:

```text
region
```

This is the most fundamental design choice in the project.

---

### 3.3 Why Polygon Is Chosen

Polygon has several advantages.

#### Area Representation

A polygon directly describes a 2D occupied semantic region.

For example:

```text
road
vegetation
vehicle footprint
sidewalk
terrain
```

These classes are naturally region-like.

#### Sparse Representation

A scene can be represented as:

```text
N polygons
```

instead of:

```text
H × W cells
```

This preserves the sparse-query philosophy of SparseDrive.

#### Vector Representation

Polygon output remains continuous in coordinate space.

The model predicts metric geometry rather than only discrete cells.

#### Easy Rasterization

A polygon can be converted into a BEV mask:

```text
Polygon
    ↓
Rasterization
    ↓
Semantic mask
```

This provides a natural bridge to occupancy-style evaluation and future supervision.

#### Compatibility With Existing Map Regression

SparseDrive already predicts ordered points.

The minimal implementation path is therefore:

```text
ordered polyline points
        ↓
ordered polygon vertices
```

rather than introducing a completely new decoding paradigm.

---

### 3.4 Why Not Use Dense BEV Segmentation

A dense BEV head would represent:

```text
H × W × C
```

This may be effective, but it creates a different architecture.

It would require:

- dense prediction head
- dense semantic loss
- potentially different feature requirements
- new memory and latency costs
- new evaluation and post-processing pipeline

This would make the project:

```text
SparseDrive + Dense OCC Head
```

instead of:

```text
Polygon OCC
```

Therefore, dense BEV segmentation is outside V1 scope.

---

## 4. Decision: Keep the Existing Sparse Map Query Branch

### 4.1 Why Reuse the Map Branch

The existing map branch already has the correct high-level structure:

```text
Sparse Queries
    ↓
Classification
+
Point Regression
    ↓
Hungarian Matching
```

Polygon OCC V1 needs:

```text
Sparse Queries
    ↓
Classification
+
Vertex Regression
    ↓
Hungarian Matching
```

The structural similarity is very high.

Therefore, the preferred strategy is to reuse the map branch.

---

### 4.2 What "Reuse" Means

Reuse means:

- preserve query count initially
- preserve transformer interaction
- preserve map query update logic
- preserve classification head
- preserve regression head design where possible
- preserve decoder flow
- preserve Hungarian framework
- preserve training loop

Modify only what is required by:

```text
LineString → Polygon
20 points → 32 vertices
Map GT → Polygon OCC GT
```

---

### 4.3 Why Not Add a New Polygon Head Immediately

Creating a new `polygon_head` may appear cleaner.

However, it introduces:

- duplicated logic
- new registry entries
- new configuration blocks
- new initialization behavior
- possible inconsistency with map branch
- harder debugging

For V1, the preferred strategy is:

```text
reuse first
separate later if necessary
```

A dedicated polygon module should only be created if reuse becomes unsafe or semantically confusing.

---

## 5. Decision: Fixed 32-Point Polygon Representation

### 5.1 Variable-Length Polygon Problem

Real polygons have variable complexity.

Examples:

```text
vehicle footprint      ≈ 4 vertices
simple sidewalk        ≈ 8 vertices
road region            ≈ dozens of vertices
vegetation contour     ≈ hundreds of vertices
```

A neural network regression head requires a fixed output size.

Therefore, Polygon OCC V1 uses a fixed number of vertices.

---

### 5.2 Why 32 Points

The selected value is:

```text
32
```

The reasoning is:

- greater geometric capacity than the original 20-point map representation
- still small enough for efficient sparse regression
- power-of-two style configuration is easy to reason about
- 64 regression values per query remain modest
- sufficient for a proof of concept
- easy to integrate into the existing point-based design

The regression dimension becomes:

```text
32 × 2 = 64
```

---

### 5.3 Why Not 16 Points

16 points may be sufficient for:

```text
vehicle
simple road blocks
simple crossing
```

but may underfit:

```text
curved road regions
vegetation boundaries
irregular semantic regions
```

Since the project intends to move toward occupancy-style region representation, 16 points may be too restrictive.

---

### 5.4 Why Not 64 Points

64 points provide more geometric capacity.

However, they also increase:

- regression dimensionality
- matching cost complexity
- point correspondence sensitivity
- susceptibility to noisy contours
- training difficulty

V1 should first test the representation itself.

Therefore, 32 is the preferred balance.

---

### 5.5 Configurability Requirement

32 must not be hard-coded across the codebase.

Preferred design:

```python
num_polygon_points = 32
```

or reuse the repository's existing point-count parameter.

All dependent dimensions should be derived from this value.

Avoid:

```python
reg_dim = 64
```

if the same value can be written as:

```python
reg_dim = num_polygon_points * 2
```

The implementation should minimize duplicated constants.

---

## 6. Decision: Polygon Closure Is Implicit

A polygon with 32 vertices is stored as:

```text
P0
P1
...
P31
```

The polygon edge sequence is:

```text
P0 → P1
P1 → P2
...
P30 → P31
P31 → P0
```

Do not duplicate `P0` as the final stored point.

Incorrect:

```text
32 unique points + repeated P0
= 33 stored points
```

Preferred:

```text
32 stored points
closure handled by geometry logic
```

This keeps the tensor shape fixed at:

```text
32 × 2
```

Visualization, rasterization, and polygon validation must explicitly close the sequence.

---

## 7. Decision: Derive GT From Semantic Occupancy

### 7.1 Why Not Rely Only on nuScenes HD Map

HD map polygons are useful for static map regions.

However, they do not naturally provide all desired Polygon OCC categories.

For example:

```text
vehicle
pedestrian
vegetation occupancy
dynamic obstacle regions
```

The long-term Polygon OCC idea is broader than vector map prediction.

Using occupancy GT creates a more natural path toward:

```text
semantic region representation
```

rather than:

```text
map-only representation
```

---

### 7.2 Why Semantic Occupancy Is Suitable

Semantic occupancy provides a semantic field over space.

Conceptually:

```text
voxel/grid cell
    ↓
semantic class
```

Polygon OCC converts this dense semantic field into sparse regions:

```text
semantic mask
    ↓
connected region
    ↓
polygon
```

This is a form of vectorization.

The model target becomes:

```text
dense semantic GT
        ↓
sparse polygon GT
```

---

### 7.3 Preferred Dataset Direction

The preferred direction is nuScenes-based semantic occupancy annotations such as:

```text
Occ3D-nuScenes
OpenOccupancy / OpenOcc-style annotations
```

The actual implementation must inspect:

- local annotation format
- semantic label IDs
- voxel range
- voxel size
- coordinate frame
- sample token alignment

The document does not assume one exact annotation file format.

---

## 8. Decision: Convert 3D Occupancy to 2D BEV Semantic Regions

### 8.1 Why V1 Is 2D

The current SparseDrive map branch predicts 2D map geometry.

Reusing this branch naturally suggests:

```text
(x, y)
```

polygon vertices.

Adding height immediately would change the representation to:

```text
(x, y, z)
```

or:

```text
polygon + height
```

This introduces a second major research variable.

V1 therefore focuses on 2D BEV Polygon OCC.

---

### 8.2 Why BEV Projection Is Necessary

Semantic occupancy annotations may be 3D:

```text
X × Y × Z
```

Polygon OCC V1 requires:

```text
X × Y
```

semantic masks.

Therefore, a projection or reduction along the Z dimension is required.

---

### 8.3 Selected Policy: Per-Class Independent BEV Masks

The preferred V1 design is:

```text
one binary BEV mask per semantic class
```

For class `c`:

```text
mask_c(x, y) = 1
if one or more selected voxels along z belong to class c
```

This produces:

```text
road_mask
vehicle_mask
pedestrian_mask
vegetation_mask
...
```

Each mask is processed independently.

---

### 8.4 Why Not Force a Single Exclusive BEV Label Map

Consider:

```text
road polygon
+
vehicle polygon
```

The vehicle occupies space above the road region.

In a single exclusive semantic BEV map, one class must overwrite the other.

For example:

```text
vehicle > road
```

This loses the fact that the underlying region is road.

Polygon OCC does not need this exclusivity.

It can represent:

```text
Polygon A = road
Polygon B = vehicle
```

with overlapping geometry.

This is a feature, not an error.

Therefore, independent per-class masks are preferred.

---

## 9. Decision: Use Semantic Region Instances

Polygon OCC predictions are query-based.

Each query predicts:

```text
one class
+
one polygon
```

This resembles instance prediction.

However, not every polygon represents a physical object.

Examples:

```text
vehicle polygon
    → object-like

pedestrian polygon
    → object-like

road polygon
    → region-like

vegetation polygon
    → region-like
```

Therefore, the correct conceptual unit is:

> Semantic Region Instance

This term should guide GT generation.

A connected semantic region may become one GT polygon.

---

## 10. Decision: Connected Components Define Initial Region Instances

For a class-specific binary mask:

```text
binary mask
    ↓
connected components
```

Each connected component is initially treated as one semantic region.

Then:

```text
connected component
    ↓
contour
    ↓
polygon
    ↓
32-point sampling
```

This is simple and deterministic.

---

### 10.1 Why Connected Components Are a Good V1 Choice

Advantages:

- easy to implement
- no learned clustering
- class-aware
- deterministic
- compatible with contour extraction
- naturally provides multiple polygons per class

---

### 10.2 Known Risk: Object Merging

Two nearby vehicles may occupy adjacent BEV cells.

At coarse occupancy resolution:

```text
Vehicle A + Vehicle B
        ↓
one connected component
```

This may produce one merged polygon.

V1 accepts this risk.

The project is not trying to reproduce object detection instances exactly.

The output primitive is a semantic region instance.

If region merging becomes severe, future work may use:

- 3D box instance IDs
- panoptic occupancy
- watershed separation
- distance transform
- class-specific instance rules

These are not required in V1.

---

## 11. Decision: Simplify Contours Before 32-Point Sampling

Raw raster contours are often noisy.

Example:

```text
stair-step voxel boundary
```

A raw contour may contain hundreds of points.

Directly sampling 32 points from this contour can preserve unnecessary raster artifacts.

The preferred pipeline is:

```text
raw contour
    ↓
polygon simplification
    ↓
uniform perimeter sampling
```

---

### 11.1 Why Simplification Comes Before Sampling

Without simplification:

```text
voxel staircase
    ↓
32 samples may follow noise
```

With simplification:

```text
structural contour
    ↓
32 samples describe actual shape
```

This creates cleaner GT.

---

### 11.2 Recommended Simplification

A practical V1 option is:

```text
Douglas-Peucker
```

For OpenCV:

```python
cv2.approxPolyDP(...)
```

The epsilon value should be configurable.

Avoid hard-coding a large epsilon.

The simplification strength should be related to:

- BEV resolution
- polygon perimeter
- expected semantic geometry

The GT-generation document will define this in more detail.

---

## 12. Decision: Uniformly Sample Along Polygon Perimeter

After simplification, each polygon is converted to 32 vertices.

The selected sampling method is:

> Uniform sampling by cumulative perimeter distance.

Do not sample by raw contour index.

---

### 12.1 Why Raw Index Sampling Is Wrong

Contour points from raster extraction are not guaranteed to be uniformly spaced in metric space.

For example:

```text
straight edge
    → many neighboring pixel points

diagonal or simplified edge
    → fewer points
```

Index-based sampling may overrepresent one region of the contour.

---

### 12.2 Preferred Sampling Logic

Given a closed polygon:

```text
P0, P1, ..., Pm-1
```

Compute edge lengths:

```text
d_i = ||P_{i+1} - P_i||
```

including:

```text
P_m = P_0
```

Compute total perimeter:

```text
L = Σ d_i
```

Create 32 target distances:

```text
0
L / 32
2L / 32
...
31L / 32
```

Interpolate points along the closed perimeter.

The result is:

```text
32 approximately equally spaced vertices
```

This reduces sampling bias.

---

## 13. Decision: Enforce Deterministic Vertex Ordering

This is one of the most important V1 decisions.

Polygon geometry is cyclic.

The same polygon may be represented as:

```text
[P0, P1, P2, P3]
```

or:

```text
[P1, P2, P3, P0]
```

These are geometrically identical.

However:

```text
L1(sequence_A, sequence_B)
```

may be large.

The same applies to reverse ordering.

Therefore, vertex ordering must be normalized.

---

### 13.1 Direction Normalization

Use one consistent orientation:

```text
clockwise
```

or:

```text
counter-clockwise
```

The project must choose exactly one.

The current recommended V1 choice is:

```text
clockwise
```

The direction should be verified using the actual SparseDrive coordinate convention.

---

### 13.2 Starting Vertex Normalization

After direction normalization, choose a deterministic starting vertex.

Recommended V1 rule:

```text
minimum x
```

Tie-break using:

```text
minimum y
```

Conceptually:

```python
start_idx = lexicographic_argmin(x, y)
```

Then cyclically rotate the 32-point sequence so this point becomes index 0.

---

### 13.3 Why This Is Necessary Even With Hungarian Matching

Hungarian matching matches:

```text
predicted query
↔
GT polygon
```

It does not automatically resolve:

```text
cyclic vertex permutation
```

Therefore:

```text
polygon-level matching
```

does not solve:

```text
vertex-level correspondence
```

Without ordering normalization, the matching cost itself may become unreliable.

---

## 14. Decision: Reuse Hungarian Matching

The existing SparseDrive map branch already performs set prediction.

Predicted queries are matched to GT map instances.

This is structurally compatible with Polygon OCC.

V1 should reuse the same framework.

Conceptually:

```text
cost
=
classification cost
+
vertex regression cost
```

The main change is:

```text
polyline point target
        ↓
polygon vertex target
```

---

### 14.1 Why Not Use Polygon IoU Matching in V1

Polygon IoU matching is attractive.

However, it requires:

- polygon rasterization or geometric intersection
- valid polygons
- handling self-intersection
- differentiability is not required for Hungarian cost, but implementation complexity increases
- new cost weight tuning

This changes the matching behavior and the representation simultaneously.

V1 should avoid this.

---

### 14.2 Why Not Use Chamfer Matching in V1

Chamfer distance reduces point-order sensitivity.

However, the project intentionally tests whether minimal point-based adaptation is sufficient.

Adding Chamfer to matching would introduce another major change.

Therefore, Chamfer is deferred.

---

## 15. Decision: Use Vertex L1 as the Main Geometry Loss in V1

The V1 loss should remain conceptually close to SparseDrive map point regression.

The geometry loss is:

```text
L_vertex
=
L1(pred_vertices, gt_vertices)
```

The total map/Polygon OCC loss remains conceptually:

```text
L
=
L_cls
+
λ_vertex L_vertex
```

The exact existing SparseDrive loss module and weights should be reused where possible.

---

### 15.1 Why Vertex L1 Is Chosen

Advantages:

- minimal modification
- compatible with existing regression
- easy to debug
- deterministic gradient
- preserves original training philosophy
- suitable for fixed ordered vertices

---

### 15.2 Limitation of Vertex L1

Vertex L1 optimizes point correspondence.

It does not directly optimize:

```text
area overlap
polygon IoU
region coverage
shape occupancy
```

Example:

Two polygons may have relatively small vertex errors but noticeably different area overlap.

This is a known limitation.

V1 accepts it deliberately.

---

## 16. Decision: Do Not Add Raster Loss in V1

The long-term preferred supervision may be:

```text
L
=
L_cls
+
λ_vertex L_vertex
+
λ_raster L_raster
```

The raster term may use:

```text
Dice
Focal
BCE
semantic CE
```

This is likely to complement vertex supervision.

However, Raster Loss is postponed.

---

### 16.1 Why Vertex + Raster Is Still Considered Strong

The two losses supervise different properties.

Vertex Loss:

```text
exact ordered geometry
```

Raster Loss:

```text
region area and overlap
```

They are complementary.

Therefore, postponing Raster Loss does **not** mean Raster Loss is considered unnecessary.

It means the project follows a staged research strategy.

---

### 16.2 Why Raster Loss Is Deferred

Raster Loss introduces:

- raster grid resolution
- raster coordinate conversion
- anti-aliasing
- differentiable polygon rasterizer
- invalid polygon behavior
- self-intersection handling
- overlap behavior
- class aggregation
- new loss weights

If performance improves, it would be difficult to determine whether the gain comes from:

```text
Polygon representation
```

or:

```text
Raster supervision
```

V1 must isolate the representation change.

---

## 17. Decision: Rasterization Is Allowed for Evaluation

The V1 rule is:

```text
No Raster Loss
```

The V1 rule is **not**:

```text
No Rasterization
```

Rasterization is useful for:

- visualization
- polygon IoU
- BEV semantic IoU
- debugging
- qualitative comparison

The following is valid in V1:

```text
Predicted Polygon
        ↓
Rasterize
        ↓
Evaluation Mask
```

No gradient is required.

---

## 18. Decision: Keep Original Classification Design Initially

The existing map branch already predicts map classes.

Polygon OCC requires a new semantic class mapping.

The classification head structure should be reused initially.

Only the number of classes should change as required.

Avoid redesigning the classifier.

---

### 18.1 Class Set Must Be Controlled

Do not immediately use every occupancy class.

The first class set should favor stable 2D semantic regions.

Recommended candidates:

```text
drivable_surface
sidewalk
terrain
vegetation
vehicle
pedestrian
barrier
```

This is not a fixed final list.

The exact list should be selected after GT statistics.

---

### 18.2 Why Not Include All Classes

Some occupancy classes may be:

- extremely sparse
- vertically dominant
- noisy in BEV
- heavily fragmented
- unsuitable for region contours

A large class set makes it harder to determine whether the Polygon representation itself works.

V1 should start with a stable subset.

---

## 19. Decision: Do Not Merge Detection and Polygon OCC Queries in V1

A long-term idea is:

```text
one unified semantic polygon query set
```

representing:

```text
road
vehicle
pedestrian
vegetation
building
...
```

This could potentially unify:

```text
detection
map
occupancy
```

However, this is a much larger architecture change.

V1 should only replace the map branch representation.

The detection branch remains unchanged.

---

### 19.1 Why This Is Important

If detection queries and map queries are unified immediately, the experiment changes:

- representation
- query semantics
- branch structure
- assignment
- class taxonomy
- losses

This would make debugging and ablation difficult.

Therefore:

```text
V1 = map branch replacement only
```

---

## 20. Decision: Do Not Add Height in V1

A 3D Polygon OCC representation could use:

```text
polygon
+
bottom_z
+
height
```

or:

```text
polygon
+
z_min
+
z_max
```

This would allow polygon extrusion into voxels.

Example:

```text
vehicle footprint + height
    ↓
3D occupancy volume
```

This is an important future direction.

However, V1 remains 2D.

---

### 20.1 Why Height Is Deferred

Adding height introduces:

- new GT generation
- new regression targets
- new normalization
- 3D evaluation
- occlusion interpretation
- vertically overlapping classes

The initial research question is specifically about sparse polygon region prediction.

Height is postponed.

---

## 21. Decision: Keep Polygon GT Generation Modular

GT conversion should be implemented as a modular pipeline.

Preferred conceptual modules:

```text
occupancy loader
    ↓
BEV projector
    ↓
class mask generator
    ↓
connected component extractor
    ↓
contour extractor
    ↓
polygon simplifier
    ↓
32-point resampler
    ↓
vertex order normalizer
```

Each stage should be testable independently.

---

### 21.1 Why Modularity Matters

Polygon GT errors may come from:

```text
wrong occupancy alignment
wrong z projection
wrong class mask
wrong contour extraction
wrong coordinate conversion
wrong sampling
wrong ordering
```

A monolithic function makes these failures difficult to isolate.

Claude Code should prefer small pure functions for the offline conversion pipeline.

---

## 22. Decision: Validate GT Before Model Integration

The model must not be modified before the Polygon GT pipeline has been visually validated.

Required pre-integration checks:

```text
sample alignment correct
semantic classes correct
BEV orientation correct
ROI correct
polygon contours correct
32-point sampling correct
clockwise order correct
starting point normalization correct
```

At least several scenes and multiple classes should be visualized.

---

### 22.1 Why This Is a Hard Requirement

A model can train with incorrect GT.

Typical failure examples:

```text
x/y swapped
left/right mirrored
90-degree rotation
wrong ego transform
wrong voxel origin
wrong class IDs
wrong point normalization
```

The training pipeline may still run.

Therefore, a successful forward/backward pass does not validate GT correctness.

---

## 23. Decision: Require a Small-Subset Overfit Test

Before full training, run an overfit experiment on a very small subset.

For example:

```text
a few frames
or
a very small scene subset
```

The goal is not generalization.

The goal is to verify:

```text
model can memorize Polygon GT
```

Expected signs:

```text
classification loss decreases
vertex loss decreases strongly
predicted polygons approach GT
```

If the model cannot overfit a tiny subset, do not begin full training.

---

## 24. Decision: Evaluate Polygon Geometry Separately From Training Loss

The training loss alone is insufficient.

Recommended evaluation metrics include:

```text
Polygon IoU
BEV semantic IoU
valid polygon ratio
self-intersection ratio
polygon area error
```

These metrics may require non-differentiable rasterization or geometry libraries.

That is acceptable.

Training and evaluation do not need identical representations.

---

## 25. Decision: Treat Self-Intersection as an Observed Risk, Not an Immediate Loss Term

A free 32-point regression may produce:

```text
bow-tie polygons
crossing edges
collapsed regions
```

These are invalid polygon geometries.

However, V1 should first measure the problem.

Recommended metrics:

```text
invalid polygon ratio
self-intersection ratio
near-zero area ratio
```

Only add topology constraints if invalid geometry is a major failure mode.

---

### 25.1 Why Not Add Topology Loss Immediately

Potential topology losses introduce:

- edge-edge intersection logic
- ordering constraints
- angle regularization
- convexity assumptions
- differentiable geometry complexity

These may overconstrain valid semantic shapes.

V1 should observe first.

---

## 26. Decision: Preserve Existing Coordinate Convention

Polygon OCC GT must be converted into the same local BEV coordinate system used by the original SparseDrive map branch.

Do not create a new coordinate convention.

Claude Code should inspect:

```text
map ROI
coordinate normalization
ego frame
x/y axis
map point denormalization
decoder output conversion
```

Then convert Polygon GT into the same convention.

---

### 26.1 Why This Is Preferred

Reusing the map coordinate convention allows:

- reuse of normalization
- reuse of regression range
- reuse of decoder logic
- simpler visualization
- lower implementation risk

The representation changes.

The coordinate system should not.

---

## 27. Decision: Prefer Runtime Integration Only After Offline GT Prototype

There are two possible GT-generation strategies.

### Offline Preprocessing

```text
Occ GT
    ↓
Polygon conversion script
    ↓
saved Polygon GT
    ↓
training loader
```

### Runtime Conversion

```text
training loader
    ↓
load Occ GT
    ↓
generate Polygon GT on the fly
```

The preferred V1 development order is:

```text
offline prototype first
```

Then decide whether production training should use offline or runtime conversion.

---

### 27.1 Why Offline First

Offline generation is easier to:

- visualize
- inspect
- debug
- collect statistics
- reproduce
- cache

Runtime generation hides preprocessing errors inside DataLoader logic.

---

### 27.2 Final Choice May Depend on Dataset Size

After validation:

- offline saved Polygon GT may improve training speed
- runtime conversion may reduce storage
- cache-based conversion may balance both

The final implementation can be chosen after measuring cost.

---

## 28. Decision: Keep Query Count Unchanged Initially

The number of semantic polygons per frame may differ from the number of vector map elements.

However, changing query count immediately adds another variable.

V1 should first try the existing map query count.

Collect GT statistics:

```text
mean polygons per frame
P95 polygons per frame
max polygons per frame
per-class polygon counts
```

Only increase query count if GT coverage is insufficient.

---

### 28.1 Why Query Statistics Matter

If:

```text
GT polygons = 150
queries = 100
```

some GT regions cannot be matched.

If:

```text
GT polygons = 20
queries = 100
```

the existing query count may be more than sufficient.

Therefore, query count should be data-driven.

---

## 29. Decision: Do Not Rename Every Internal "map" Symbol in V1

The project concept is now Polygon OCC.

However, aggressively renaming:

```text
map_head
gt_map_pts
map_decoder
map_loss
```

may create large code churn.

V1 should prefer minimal changes.

A practical strategy is:

```text
keep internal map interfaces when reuse is convenient
document that geometry is now Polygon OCC
```

Only rename interfaces where the old name becomes actively misleading or causes confusion.

---

### 29.1 Why This Matters for Claude Code

Mass renaming creates:

- import changes
- config changes
- registry changes
- checkpoint incompatibility
- merge conflicts
- unnecessary bugs

The first implementation should prioritize behavior over naming purity.

---

## 30. Decision: Document Every Non-Minimal Change

If Claude Code determines that a major change is necessary, it should explicitly document:

```text
1. existing behavior
2. reason minimal reuse is insufficient
3. proposed change
4. affected modules
5. tensor shape changes
6. compatibility risks
```

This is especially required for:

- assigner changes
- decoder changes
- head replacement
- new loss modules
- new data containers

Do not silently redesign components.

---

## 31. Rejected or Deferred Alternatives

### 31.1 Dense Semantic BEV Head

Status:

```text
Rejected for V1
```

Reason:

Breaks sparse representation goal.

---

### 31.2 3D Voxel OCC Head

Status:

```text
Rejected for V1
```

Reason:

Different research direction.

---

### 31.3 Variable-Length Polygon Decoder

Status:

```text
Deferred
```

Reason:

Requires sequence stopping or autoregressive decoding.

---

### 31.4 Bézier Polygon

Status:

```text
Deferred
```

Reason:

Higher representation complexity.

---

### 31.5 B-Spline Representation

Status:

```text
Deferred
```

Reason:

Requires control-point and knot design.

---

### 31.6 Fourier Descriptor

Status:

```text
Deferred
```

Reason:

Less directly compatible with SparseDrive point regression.

---

### 31.7 Signed Distance Function

Status:

```text
Deferred
```

Reason:

Changes the representation to implicit geometry.

---

### 31.8 Chamfer Loss

Status:

```text
Deferred to V2
```

Reason:

Would change supervision in addition to representation.

---

### 31.9 Hausdorff Loss

Status:

```text
Deferred to V2
```

Reason:

Can be sensitive to outliers and introduces another geometry objective.

---

### 31.10 Raster Loss

Status:

```text
Deferred to V3
```

Reason:

Strong but adds a second representation domain during training.

---

### 31.11 Polygon IoU Matching

Status:

```text
Deferred
```

Reason:

More complex Hungarian cost and invalid-polygon handling.

---

### 31.12 Unified Detection + Map + OCC Query

Status:

```text
Long-term research direction
```

Reason:

Too many simultaneous architecture changes for V1.

---

## 32. Primary Research Hypothesis

The V1 hypothesis is:

> SparseDrive's sparse map query framework can be repurposed from line-like vector map prediction to fixed-length semantic polygon region prediction with only minimal architectural changes.

The experiment should test this hypothesis directly.

The V1 contribution is not:

```text
new rasterizer
new dense OCC decoder
new topology loss
new matching algorithm
```

The V1 contribution is the representation transition:

```text
Sparse LineString Map
        ↓
Sparse Polygon OCC
```

---

## 33. Secondary Hypothesis

A secondary hypothesis is:

> A semantic scene may be represented efficiently as a sparse set of semantic region polygons rather than only as a dense BEV or voxel grid.

V1 does not fully prove this broad hypothesis.

It provides the first implementation step.

Later versions may compare:

```text
Polygon OCC
vs
Dense BEV semantic occupancy
```

on:

- memory
- latency
- geometric quality
- semantic IoU
- planning usefulness

---

## 34. Design Review Checklist

Before implementation, verify:

```text
[ ] Polygon OCC is understood as sparse region representation.
[ ] No dense OCC head will be added.
[ ] Existing map branch is the starting point.
[ ] Polygon point count is fixed at 32.
[ ] Polygon closure is implicit.
[ ] GT source is semantic occupancy.
[ ] BEV masks are per-class independent.
[ ] Connected components define initial regions.
[ ] Contours are simplified.
[ ] Sampling is uniform by perimeter distance.
[ ] Vertex orientation is normalized.
[ ] Starting vertex is deterministic.
[ ] Hungarian matching is reused.
[ ] Vertex L1 is reused.
[ ] Raster Loss is disabled in V1.
[ ] Rasterization is allowed for evaluation.
[ ] Detection branch remains unchanged.
[ ] Height prediction is deferred.
[ ] Tiny-subset overfit test is required.
```

If any item changes, the design document should be updated before full implementation.

---

## 35. Final Decision Statement

Polygon OCC V1 deliberately prioritizes:

```text
minimal change
clear ablation
sparse representation
fixed geometry
debuggability
```

over:

```text
maximum immediate accuracy
full 3D occupancy
complex geometric losses
unified multi-task architecture
```

The implementation should remain conservative.

The project should first prove that:

```text
SparseDrive Map Query
        ↓
32-point Closed Polygon
```

is trainable and meaningful.

Only after that result is established should the project introduce:

```text
Chamfer
Hausdorff
Differentiable Rasterization
Raster Loss
Height
Unified Scene Queries
```

The most important engineering rule is:

> Do not solve future versions inside V1.

The most important research rule is:

> Keep the first representation experiment clean enough that its result can be interpreted.
