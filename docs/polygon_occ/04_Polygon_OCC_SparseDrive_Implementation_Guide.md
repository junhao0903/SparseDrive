# 04_Polygon_OCC_SparseDrive_Implementation_Guide

## 1. Purpose of This Document

This document is the implementation guide for integrating **Polygon OCC V1** into SparseDrive.

The implementation objective is:

> Replace the original LineString-based map representation with a fixed 32-point semantic polygon representation using the smallest safe set of code changes.

This document is written for Claude Code.

It does not assume exact repository file names.

Before modifying code, Claude Code must inspect the checked-out SparseDrive repository and derive the actual:

- dataset classes
- map annotation preprocessing path
- map data pipeline
- map head
- point encoder
- point decoder
- Hungarian assigner
- regression cost
- regression loss
- visualization
- evaluation path

The source tree is the source of truth.

This document defines:

- implementation order
- required behavior
- tensor-shape changes
- compatibility constraints
- verification steps
- failure checkpoints

The implementation must preserve SparseDrive architecture whenever possible.

---

## 2. V1 Implementation Scope

Polygon OCC V1 modifies only the existing map-task path.

Conceptually:

```text
Current SparseDrive Map Path

Map GT
    ↓
Polyline / LineString targets
    ↓
Map Query
    ↓
Point regression
    ↓
Hungarian Matching
    ↓
Map Loss
    ↓
Polyline Decoder
```

Target:

```text
Polygon OCC V1 Path

Semantic Polygon GT
    ↓
32-point closed polygon targets
    ↓
Existing Map Query
    ↓
32-point vertex regression
    ↓
Existing Hungarian Matching
    ↓
Existing point/vertex regression loss
    ↓
Polygon-aware Decoder / Visualization
```

The required representation change is:

```text
20 × 2 polyline points
        ↓
32 × 2 polygon vertices
```

The implementation should avoid unrelated architectural changes.

---

## 3. Hard Implementation Constraints

Claude Code must obey the following constraints.

### Constraint 1: Do Not Add a Dense OCC Head

Do not create:

```text
occ_head
voxel_decoder
dense_bev_head
3d_occ_decoder
```

unless explicitly instructed in a future version.

Polygon OCC V1 is not a conventional dense OCC implementation.

---

### Constraint 2: Preserve Sparse Queries

The model must continue to use the existing sparse map-query framework.

Do not convert the task into:

```text
BEV feature
    ↓
dense semantic segmentation
```

---

### Constraint 3: Modify the Map Branch Only

Do not intentionally change:

```text
detection head
motion head
planning head
object decoder
object Hungarian matching
```

Shared utility changes are allowed only when required and must preserve existing behavior.

---

### Constraint 4: Fixed 32 Vertices

Every Polygon OCC regression target must use:

```text
32 × 2
```

The number of points should be configuration-driven.

---

### Constraint 5: Keep Polygon Closure Implicit

The tensor stores:

```text
P0 ... P31
```

Do not append `P0` again.

Geometry code must connect:

```text
P31 → P0
```

---

### Constraint 6: Reuse Hungarian Matching

V1 should reuse the original map Hungarian matching framework.

Do not introduce Polygon IoU, raster IoU, Dice cost, or Chamfer cost in V1.

---

### Constraint 7: Reuse Existing Regression Loss

If the current map regression loss supports arbitrary regression dimensions, reuse it.

V1 should use point/vertex L1-style supervision.

Do not create a new geometry loss unless technically required.

---

### Constraint 8: No Raster Loss in V1

Rasterization may be used for:

```text
debug
visualization
evaluation
```

It must not contribute gradients during V1 training.

---

## 4. Required Implementation Order

Do not modify the model first.

The required order is:

```text
1. Repository source inspection
2. Existing map data-flow summary
3. Polygon GT offline validation
4. Dataset integration
5. Configuration update
6. Regression-shape update
7. Assigner verification
8. Loss verification
9. Decoder/output verification
10. Visualization update
11. DataLoader test
12. Forward smoke test
13. Backward smoke test
14. Tiny-subset overfit test
15. Full training
```

Each stage has an acceptance checkpoint.

Do not continue to full training if an earlier checkpoint fails.

---

## 5. Stage 1: Inspect the Existing Map Data Flow

Claude Code must first trace the current map branch.

The goal is to answer:

```text
Where does map GT originate?
Where is it stored?
Which pipeline transforms collect it?
What tensor/object shape enters the model?
Which head predicts map points?
Where is num_sample defined?
How is regression dimension derived?
Which assigner matches map queries?
Which cost compares map points?
Which loss supervises map regression?
Which decoder converts normalized points to metric coordinates?
Which evaluator consumes map predictions?
Which visualizer draws map polylines?
```

Recommended search keywords:

```text
gt_map_pts
gt_map_labels
map_annos
num_sample
SparsePoint3D
map_head
map_decoder
map_assigner
reg_weights
MapTR
LineString
polyline
```

Search actual repository symbols rather than assuming names from this document.

---

## 6. Required Source-Inspection Deliverable

Before code modification, create a short local implementation note.

Example:

```text
Map GT source:
    file/class/function

Dataset output:
    key
    type
    shape

Map head:
    file/class

Point count config:
    config variable

Regression shape:
    current dimension

Assigner:
    file/class

Regression cost:
    file/class

Regression loss:
    file/class

Decoder:
    file/class

Visualization:
    file/function
```

This note can be temporary.

Do not produce a 50-page source analysis.

The purpose is to identify the exact modification surface.

---

## 7. Stage 2: Integrate the Polygon GT Source

The GT generation behavior is defined in:

```text
03_Polygon_OCC_GT_Generation.md
```

Before SparseDrive integration, Polygon GT must already satisfy:

```text
gt polygon count = N
polygon shape = [N, 32, 2]
label shape = [N]
finite values
deterministic ordering
clockwise orientation
deterministic start vertex
```

The integration task is to expose this data through the existing map GT data path.

---

## 8. Preferred Dataset Integration Strategy

Prefer reusing the existing map target interface.

For example, if SparseDrive currently expects conceptually:

```python
gt_map_pts
gt_map_labels
```

the minimal V1 implementation may keep these internal keys and replace their semantic content:

```text
gt_map_pts
    previously: polyline points
    V1: polygon vertices

gt_map_labels
    previously: map classes
    V1: Polygon OCC semantic classes
```

This approach reduces code churn.

The conceptual project terms remain:

```text
Polygon GT
Polygon labels
```

but internal field names do not need immediate global renaming.

---

## 9. When to Introduce New GT Keys

New keys such as:

```text
gt_polygon_pts
gt_polygon_labels
```

should be introduced only if:

- existing map transforms assume LineString semantics
- original map evaluation must remain operational in parallel
- field reuse creates ambiguous or unsafe behavior
- both original map and Polygon OCC need to coexist

If V1 fully replaces the map task, reusing map keys may be simpler.

Claude Code should choose based on the actual code path.

The choice must be documented.

---

## 10. Dataset Output Shape

The required per-sample Polygon target is:

```text
polygon points:
[N, 32, 2]

polygon labels:
[N]
```

Where `N` varies by sample.

For an empty sample:

```text
polygon points:
[0, 32, 2]

polygon labels:
[0]
```

Do not allow the empty polygon tensor to collapse to:

```text
[0]
```

The DataLoader and DataContainer behavior must be verified.

---

## 11. Dataset Pipeline Audit

Inspect all transforms that touch map GT.

Potential transform categories include:

```text
loading
formatting
normalization
augmentation
rotation
flip
scaling
collection
batch collation
```

For each transform, answer:

> Does this transform work on arbitrary ordered 2D points, or does it assume an open LineString?

Examples of likely reusable transforms:

```text
2D rotation
translation
scaling
normalization
```

Examples requiring inspection:

```text
line reversal augmentation
shifted point patterns
polyline direction variants
map-specific permutation targets
line clipping
```

Do not assume all existing map point transforms are valid for polygons.

---

## 12. Important Audit: Shifted Point Sequences

Vector-map methods often generate multiple point-order variants.

Examples:

```text
forward sequence
reverse sequence
shifted sequence
```

This may exist to handle direction or permutation ambiguity.

Polygon OCC has cyclic symmetry.

Therefore, any existing logic related to:

```text
shift_fixed_num_sampled_points
shift_fixed_num_sampled_points_v1
shift_fixed_num_sampled_points_v2
reverse points
```

must be inspected carefully.

Do not blindly reuse LineString sequence-shift logic.

---

## 13. V1 Point-Ordering Strategy

The Polygon OCC GT pipeline already enforces:

```text
fixed orientation
+
fixed starting vertex
```

Therefore, the preferred V1 training target is:

```text
one canonical 32-point sequence per polygon
```

If the current map training pipeline generates multiple shifted variants for matching, Claude Code must determine whether that behavior should be disabled for Polygon OCC.

Preferred V1 philosophy:

```text
canonical target
+
plain vertex regression
```

This keeps the first experiment interpretable.

---

## 14. Preserve Geometric Augmentations

If the current map target is transformed under data augmentation, Polygon GT must receive the same geometric transformation.

Examples:

```text
rotation
horizontal flip
BEV flip
scaling
translation
```

A polygon is still a set of 2D metric points.

Therefore, standard point-coordinate transformations should remain valid.

After augmentation, verify whether canonical vertex ordering must be recomputed.

This is important.

---

## 15. Re-Normalize Vertex Ordering After Reflection

A reflection or flip can reverse polygon orientation.

Example:

```text
clockwise polygon
    ↓
mirror flip
    ↓
counter-clockwise polygon
```

If vertex L1 assumes canonical clockwise order, post-augmentation orientation may become inconsistent.

Therefore, if geometric augmentation includes reflections, the pipeline must either:

### Option A

Re-run orientation and starting-point normalization after augmentation.

Preferred.

### Option B

Transform canonical targets and adjust vertex sequence explicitly.

Option A is simpler and safer.

This check is mandatory.

---

## 16. Stage 3: Update Polygon Point Count Configuration

Locate the current map point-count configuration.

Possible conceptual variable:

```text
num_sample
fixed_num
num_pts_per_vec
num_map_points
```

Change the active Polygon OCC V1 configuration to:

```text
32
```

Prefer:

```python
num_polygon_points = 32
```

if the project configuration permits a clearer semantic alias.

However, do not duplicate the value unnecessarily.

---

## 17. Derive Regression Dimension From Point Count

The regression dimension should become:

```text
32 × 2 = 64
```

Do not hard-code `64` in multiple places.

Preferred:

```python
reg_dim = num_points * 2
```

or let the existing encoder/head derive the shape automatically.

Search for current assumptions such as:

```text
40
20 * 2
reshape(..., 20, 2)
view(..., 20, 2)
```

Every hard-coded 20 or 40 in the map path must be reviewed.

Do not globally replace all `20` values.

Only modify values semantically tied to map point count.

---

## 18. Shape-Change Checklist

Search the map path for:

```text
20
40
num_sample
num_points
2 * num_sample
reshape
view
flatten
unflatten
reg_weights
output_dim
anchor dimensions
```

For every match, classify it as:

```text
point-count dependent
unrelated
```

Required target shapes may include:

```text
[B, num_queries, 32, 2]
[B, num_queries, 64]
[N_gt, 32, 2]
[N_gt, 64]
```

Document the actual repository shapes.

---

## 19. Stage 4: Inspect and Update the Map Point Encoder

SparseDrive may encode map anchors or points into query embeddings.

If the map branch contains a point encoder, inspect whether it assumes:

```text
num_points = 20
```

or accepts arbitrary point count.

Possible behaviors:

```text
flatten [num_points, 2]
MLP over all points
point-wise encoding
positional encoding
anchor encoding
```

The V1 change should preserve the encoder architecture where possible.

Only input dimensionality should change if required.

---

## 20. Anchor Initialization

Inspect how map query anchors are initialized.

Potential forms:

```text
learned points
fixed anchors
anchor bank
normalized point sets
```

If the anchor shape is tied to:

```text
20 × 2
```

it must become:

```text
32 × 2
```

Do not initialize polygon anchors as random unrelated vectors without understanding the current anchor design.

Preferred approach:

```text
reuse the same anchor mechanism
with updated point count
```

---

## 21. Pretrained Checkpoint Compatibility

Changing map regression dimension from 40 to 64 may break map-branch checkpoint loading.

Expected mismatches may include:

```text
map regression output layer
map anchor embedding
point encoder input layer
map anchor file
regression branch weights
```

The detection branch and shared backbone may still be loadable.

The implementation should support partial checkpoint loading.

Do not force strict checkpoint loading if only map-shape layers differ.

---

## 22. Required Checkpoint-Loading Behavior

When loading a SparseDrive pretrained checkpoint:

```text
shared compatible weights
    → load

map-shape incompatible weights
    → skip / reinitialize
```

Log all skipped keys.

The expected skipped keys should be reviewed.

Unexpected mismatches in:

```text
backbone
image neck
detection branch
shared transformer
```

must be investigated.

---

## 23. Map Anchor Files or Cached Anchors

Sparse vector-map models may use pre-generated anchors.

Search for:

```text
anchor file
.npy
.pkl
map_anchor
instance_bank
```

If a map anchor file stores:

```text
[num_anchors, 20, 2]
```

it is incompatible with:

```text
[num_anchors, 32, 2]
```

Do not reshape the old anchor file arbitrarily.

Possible V1 options:

### Option A: Regenerate 32-Point Anchors

Preferred if anchor generation tooling exists.

### Option B: Resample Existing 20-Point Anchors to 32 Points

Possible only if anchor geometry is meaningful and ordered.

### Option C: Initialize New Learned Anchors

Use only if compatible with the original design.

Claude Code must inspect the real anchor mechanism.

---

## 24. Resampling Existing Polyline Anchors Is Not Automatically Correct

An original map anchor may be an open LineString.

Closing it creates:

```text
last point → first point
```

which may form an arbitrary edge.

Therefore:

```text
20-point LineString anchor
```

cannot automatically be treated as:

```text
Polygon anchor
```

This is an important semantic incompatibility.

If the map branch relies strongly on geometric anchor priors, Polygon OCC may require new anchors.

This should be documented as a major implementation decision.

---

## 25. Recommended V1 Anchor Strategy

First inspect whether the current map anchor is:

```text
essential geometric prior
```

or primarily:

```text
learned sparse query reference
```

If anchor geometry is not strongly hand-designed, use the simplest compatible 32-point initialization.

If anchor geometry encodes real line shapes, do not preserve incorrect line priors for Polygon prediction merely to minimize code changes.

Minimal modification does not mean preserving semantically wrong initialization.

---

## 26. Stage 5: Update the Map Regression Head

Locate the final regression branch for map points.

Current conceptual output:

```text
num_queries × (20 × 2)
```

Target:

```text
num_queries × (32 × 2)
```

The classification branch only changes class count.

The regression branch output dimension becomes:

```text
64
```

if flattened.

---

## 27. Preferred Regression-Head Modification

If current code uses:

```python
Linear(embed_dims, num_points * 2)
```

change the configuration point count.

Avoid adding a new network block.

If current code hard-codes:

```python
Linear(embed_dims, 40)
```

replace with a derived dimension.

Example:

```python
self.reg_dim = self.num_points * 2
```

The exact code style should match the repository.

---

## 28. Regression Output Semantics

The output tensor remains ordered 2D coordinates.

Original:

```text
point 0
point 1
...
point 19
```

Target:

```text
vertex 0
vertex 1
...
vertex 31
```

The model does not need an explicit "closed" flag.

Polygon closure is a semantic interpretation of the ordered sequence.

---

## 29. Stage 6: Verify Coordinate Encoding and Normalization

SparseDrive map points may be normalized relative to a map ROI.

Conceptually:

```text
metric coordinate
    ↓
normalize to [0, 1]
```

or another range.

The Polygon OCC vertices should reuse this normalization.

Inspect:

```text
encode
decode
normalize
denormalize
sigmoid
inverse_sigmoid
roi_size
pc_range
```

Do not directly feed raw metric coordinates if the existing map regression expects normalized coordinates.

---

## 30. Required Normalization Test

Create a reference Polygon GT.

Example:

```text
rectangle in metric BEV
```

Run:

```text
metric
→ map target normalization
→ map target denormalization
```

Verify:

```text
reconstructed coordinates ≈ original coordinates
```

Test all ROI edges.

This should be a small unit/integration test.

---

## 31. Stage 7: Verify Hungarian Matching

The existing map Hungarian assigner should remain in use.

The main verification is shape compatibility.

Conceptually:

```text
pred:
[num_queries, 32, 2]

gt:
[num_gt, 32, 2]
```

The regression cost may flatten these tensors.

Expected pairwise cost:

```text
[num_queries, num_gt]
```

---

## 32. Inspect the Map Regression Cost

Find the exact cost implementation.

Questions:

```text
Does it flatten point dimensions?
Does it explicitly assume 20 points?
Does it support multiple GT point shifts?
Does it compute min over point-order variants?
Does it use L1?
Does it use Chamfer?
```

Do not claim Hungarian is unchanged until this code is inspected.

The framework should remain Hungarian.

The regression-cost implementation may require a small Polygon-specific adjustment if it assumes LineString variants.

---

## 33. Preferred V1 Matching Cost

Conceptually:

```text
cost
=
cls_cost
+
λ_reg × L1(
    pred_vertices,
    canonical_gt_vertices
)
```

The GT sequence is canonical.

Do not add new geometric matching terms in V1.

If the existing cost can already compute this with arbitrary point count, no new cost class is needed.

---

## 34. Disable LineString-Specific Shift Matching When Necessary

Some vector-map assigners compare predictions against multiple shifted versions of a GT line.

For Polygon OCC V1, canonical ordering is already defined.

If the current matching performs:

```text
min over shifted GT point sequences
```

there are two possible choices.

### Preferred V1 Choice

Use one canonical sequence only.

Reason:

```text
clean vertex correspondence experiment
```

### Alternative

Reuse cyclic-shift invariant matching.

This may improve optimization but changes the V1 supervision assumption.

Do not choose the alternative silently.

The implementation note must document the decision.

---

## 35. Recommended First Implementation: Canonical Sequence

For the cleanest V1 experiment:

```text
GT preprocessing
    ↓
clockwise
    ↓
canonical start point
    ↓
one fixed 32-point target
```

Then:

```text
Hungarian cost = direct L1
```

This is the preferred baseline.

If training fails due to cyclic correspondence instability, a cyclic-invariant variant can become a separate experiment.

---

## 36. Stage 8: Verify Regression Loss

Inspect the current map regression loss.

Questions:

```text
Is it L1Loss?
SmoothL1?
weighted L1?
custom map loss?
does it flatten points?
does it use reg_weights?
does it assume 40 regression weights?
```

The preferred V1 strategy is to reuse the current loss.

---

## 37. Update Regression Weights

A common configuration pattern may contain:

```python
reg_weights = [1.0] * 40
```

For 32 2D vertices:

```python
reg_weights = [1.0] * 64
```

Prefer a derived expression:

```python
reg_weights = [1.0] * (num_polygon_points * 2)
```

if the config system supports it.

Search all map-task `reg_weights`.

Do not modify detection regression weights.

---

## 38. Geometry Loss Definition

The intended V1 geometry loss is:

```text
L_vertex =
mean or weighted L1 over 32 × 2 coordinates
```

The exact reduction should follow the original map loss.

Do not change:

```text
mean → sum
loss normalization
positive sample normalization
decoder-stage weighting
```

unless required.

The goal is to preserve the original training behavior.

---

## 39. Multi-Decoder Losses

SparseDrive may supervise predictions at multiple decoder stages.

If map regression loss is applied at each stage, Polygon OCC should preserve this behavior.

Conceptually:

```text
decoder stage 0 → polygon loss
decoder stage 1 → polygon loss
...
final stage → polygon loss
```

Do not accidentally supervise only the final stage.

Inspect current map loss aggregation.

---

## 40. Temporal Map Behavior

SparseDrive may propagate map queries temporally.

Polygon OCC V1 should preserve this behavior if it is already part of the map branch.

However, semantic region identity across frames may differ from original map-line identity.

Potential issue:

```text
connected components change shape
split
merge
```

V1 does not redesign temporal association.

The implementation should first preserve existing temporal mechanics.

Document temporal instability if observed.

---

## 41. Stage 9: Update the Map Decoder

The decoder may currently output:

```text
scores
labels
pts
```

The minimal V1 strategy is to preserve the structure.

For example:

```python
{
    "scores": ...,
    "labels": ...,
    "pts": ...,
}
```

where:

```text
pts.shape = [N, 32, 2]
```

The semantic interpretation changes from polyline to polygon.

Do not rename decoder output keys unless necessary.

---

## 42. Polygon Closure in Decoder

The decoder should not append the first point.

Output:

```text
[N, 32, 2]
```

Downstream Polygon consumers must treat the sequence as closed.

Avoid returning:

```text
[N, 33, 2]
```

This would make training and output shapes inconsistent.

---

## 43. Decoder Filtering

Preserve the existing map score-threshold and top-k filtering initially.

Do not introduce:

```text
polygon NMS
mask NMS
polygon IoU suppression
```

in V1.

The original set-prediction design may already avoid heavy NMS.

Polygon overlap between semantic classes is allowed.

---

## 44. Same-Class Overlapping Polygon Predictions

V1 may produce duplicate or overlapping same-class polygons.

Do not immediately add post-processing.

First measure:

```text
duplicate polygon frequency
same-class high-IoU pairs
```

If severe, future work may add:

```text
polygon NMS
set-level overlap regularization
```

Not V1.

---

## 45. Stage 10: Update Visualization

The existing map visualizer likely draws open polylines.

Polygon OCC visualization must close each sequence.

Current conceptual logic:

```text
draw P0-P1
draw P1-P2
...
draw P30-P31
```

Target:

```text
draw P0-P1
...
draw P30-P31
draw P31-P0
```

---

## 46. Required Polygon Visualization Modes

Implement:

### Mode 1: GT Polygon

Draw:

```text
boundary
vertices
index 0 marker
class
```

### Mode 2: Predicted Polygon

Draw:

```text
boundary
vertices
class
score
```

### Mode 3: GT + Prediction Overlay

Use the project's normal visualization conventions.

### Mode 4: Filled Polygon

Optional transparency fill.

### Mode 5: Rasterized Comparison

Debug/evaluation only.

---

## 47. Vertex-Index Visualization

During debugging, support drawing vertex indices:

```text
0
1
2
...
31
```

At minimum, highlight:

```text
vertex 0
```

This is important for detecting:

```text
wrong start-point normalization
reversed orientation
unexpected cyclic shifts
```

Do not remove this debug mode before the GT pipeline is validated.

---

## 48. Stage 11: Update or Replace Map Evaluation

Original map evaluation may compute vector-map metrics.

These metrics may assume:

```text
LineString
divider
boundary
ped_crossing
```

They may not be meaningful for Polygon OCC.

Do not report original map metrics as Polygon OCC quality without analysis.

---

## 49. Recommended V1 Evaluation

Use:

```text
Polygon IoU
BEV semantic IoU
valid polygon ratio
self-intersection ratio
area error
classification metrics
```

Rasterization is allowed for evaluation.

Conceptual evaluation:

```text
Predicted Polygons
        ↓
Rasterize per class
        ↓
Predicted BEV masks

GT Polygons
        ↓
Rasterize per class
        ↓
GT BEV masks

        ↓
IoU / mIoU
```

---

## 50. Evaluation Rasterization Policy

Use the same:

```text
ROI
resolution
coordinate transform
class mapping
```

for GT and prediction.

Do not rasterize GT directly from original Occ GT while rasterizing predictions from polygons if the metric is intended to measure model-vs-Polygon-GT quality.

Recommended metrics:

### Representation Metric

```text
original Occ mask
vs
rasterized Polygon GT
```

### Model Metric

```text
rasterized Polygon GT
vs
rasterized predicted Polygon
```

### End-to-End OCC Approximation Metric

```text
original Occ mask
vs
rasterized predicted Polygon
```

Keep these metrics distinct.

---

## 51. Stage 12: DataLoader Validation

Before running the model, inspect several dataset samples.

Print or assert:

```text
number of polygons
points shape
labels shape
dtype
min coordinate
max coordinate
```

Expected:

```text
pts: [N, 32, 2]
labels: [N]
```

For batched data, verify variable `N` handling.

Do not proceed if any sample has:

```text
NaN
Inf
wrong last dimension
unexpected 33 points
```

---

## 52. Recommended Dataset Assertions

During development:

```python
assert gt_pts.ndim == 3
assert gt_pts.shape[1] == num_polygon_points
assert gt_pts.shape[2] == 2
assert len(gt_pts) == len(gt_labels)
assert np.isfinite(gt_pts).all()
```

Use equivalent tensor checks if the data is already converted to PyTorch.

Assertions may later be reduced after stabilization.

---

## 53. Stage 13: Forward Smoke Test

Run one or a few batches.

The test should verify:

```text
dataset load
collation
model forward
map/polygon head output
assigner
loss construction
decoder
```

Required checks:

```text
prediction shape
GT shape
pairwise cost shape
matched indices
loss finite
```

Do not start a distributed full training job for the first test.

---

## 54. Forward Smoke-Test Logging

Temporarily log one batch:

```text
num queries
num GT polygons
pred regression shape
GT regression shape
classification shape
cost matrix shape
number of matches
loss values
```

Example conceptual output:

```text
queries: 100
gt polygons: 37
pred pts: [100, 32, 2]
gt pts: [37, 32, 2]
cost: [100, 37]
matches: 37
```

Remove or gate verbose logging after validation.

---

## 55. Stage 14: Backward Smoke Test

After forward succeeds:

```text
loss.backward()
```

Verify:

```text
no NaN gradients
no shape error
regression head receives gradient
map query branch receives gradient
```

Inspect gradient norms for:

```text
map classification output
map regression output
map query embedding / instance bank
```

Do not assume finite loss means gradients are correct.

---

## 56. Expected Initial Loss Behavior

Changing the map regression output dimension invalidates map-specific pretrained layers.

Initial Polygon regression loss may be high.

This is expected.

Warning signs include:

```text
NaN from first iteration
loss immediately zero
no matched GT
regression gradient always zero
classification dominates completely
```

These require debugging.

---

## 57. Stage 15: Tiny-Subset Overfit Test

This test is mandatory.

Select:

```text
a few samples
or
a tiny scene subset
```

Train repeatedly.

The goal is to verify memorization.

Expected behavior:

```text
classification loss decreases
vertex loss decreases significantly
predicted polygons approach GT
```

Visualize predictions regularly.

---

## 58. Why the Overfit Test Matters

Polygon OCC adds several possible failure sources:

```text
GT alignment
coordinate conversion
point ordering
shape mismatch
assignment mismatch
checkpoint initialization
```

A tiny-subset overfit test isolates basic learnability.

If the model cannot memorize a few Polygon GT samples, full training should not begin.

---

## 59. Overfit Failure Diagnosis

### Failure: Loss Does Not Decrease

Check:

```text
vertex ordering
Hungarian cost
target normalization
regression branch gradient
```

### Failure: Polygons Are Mirrored

Check:

```text
x/y axes
BEV row/column conversion
flip transforms
```

### Failure: Polygon Shape Is Right but Start Vertex Is Wrong

Check:

```text
post-augmentation ordering normalization
cyclic target shifts
```

### Failure: Classification Learns but Geometry Does Not

Check:

```text
regression loss weight
reg_weights length
target normalization
regression output activation
```

### Failure: Predictions Collapse to Tiny Region

Check:

```text
coordinate normalization
anchor initialization
sigmoid/inverse-sigmoid logic
```

---

## 60. Stage 16: Full Training

Only begin full training after:

```text
GT visual validation passed
DataLoader validation passed
forward passed
backward passed
tiny-subset overfit passed
```

Use the original training recipe initially.

Do not simultaneously change:

```text
learning rate
optimizer
decoder layers
query count
loss weights
```

unless required.

The first full experiment should isolate the Polygon representation change.

---

## 61. Learning Rate and Optimizer Policy

V1 should preserve SparseDrive optimizer and learning-rate settings initially.

The map-specific output layers are reinitialized, but shared features may use pretrained weights.

Do not automatically increase learning rate because the Polygon head is new.

If optimization issues appear, tune in a separate experiment.

---

## 62. Loss Weight Policy

Reuse original map classification and regression loss weights initially.

If the original regression loss magnitude changes significantly because:

```text
20 points → 32 points
```

first inspect the loss reduction.

If the loss averages across elements, scale may remain similar.

If it sums across coordinates, scale may increase.

Do not tune weights blindly.

Measure actual loss magnitude.

---

## 63. Query Count Policy

Keep original map query count for the first model smoke test.

Before full training, compare with GT statistics from the preprocessing pipeline.

Required statistics:

```text
P95 polygons per frame
P99 polygons per frame
max polygons per frame
```

If GT count exceeds query capacity frequently, increase query count.

Document the change as data-driven.

---

## 64. Semantic Class Count Update

The Polygon OCC class count will differ from the original map class count.

Update:

```text
num_classes
class names
label mapping
visualization labels
evaluation labels
```

Search for hard-coded original map class lists:

```text
divider
ped_crossing
boundary
```

Do not leave stale label-name mappings.

---

## 65. Class Weighting

V1 should not add class-specific loss weights unless severe imbalance is observed.

First collect:

```text
GT polygons per class
matched positives per class
classification recall per class
```

If road-like regions dominate, class weighting may become a later experiment.

Avoid adding it before the baseline.

---

## 66. Recommended Minimal-Change Dependency Map

Conceptually, the expected modification surface is:

```text
Polygon GT preprocessing
        ↓
dataset annotation loading
        ↓
map GT formatting
        ↓
map point-count config
        ↓
map anchor / point encoding if shape-dependent
        ↓
map regression output dimension
        ↓
map regression cost shape
        ↓
map regression loss weights
        ↓
map decoder point shape
        ↓
visualization
        ↓
evaluation
```

The actual code may require fewer or more files.

Do not create new modules merely to match this diagram.

---

## 67. Likely Minimal Configuration Changes

Conceptual example:

```python
polygon_occ_classes = (
    "drivable_surface",
    "sidewalk",
    "terrain",
    "vegetation",
    "vehicle",
    "pedestrian",
    "barrier",
)

num_polygon_points = 32
num_polygon_classes = len(polygon_occ_classes)
```

Then update the existing map task config:

```text
num_sample / num_points → 32
num_classes → Polygon OCC class count
reg_weights → length 64
GT source → Polygon GT
```

Actual key names must come from the repository.

---

## 68. Do Not Hard-Code Design Constants

Centralize:

```text
num polygon points
class names
class mapping
ROI
minimum area
simplification ratio
```

Avoid duplicated literals in:

```text
dataset
head
decoder
visualizer
evaluation
```

For model point count, use the existing config propagation path if possible.

---

## 69. Checkpoint Initialization Strategy

Recommended sequence:

```text
load SparseDrive pretrained checkpoint
        ↓
load all compatible shared weights
        ↓
skip incompatible Polygon/map shape layers
        ↓
initialize modified map layers
```

Log:

```text
missing keys
unexpected keys
shape mismatch keys
```

Review the list manually.

Expected map-related mismatches are acceptable.

Unexpected shared-module mismatches are not.

---

## 70. Potential Incompatible Modules

Depending on the actual source, likely shape-sensitive components include:

```text
map regression final linear layer
map anchor tensor/file
point encoder first layer
map instance bank anchor
map regression weights
decoder reshape logic
```

The implementation guide does not assume all are present.

Claude Code must inspect actual dependencies.

---

## 71. Pretrained Map Branch: Reuse or Reinitialize

The original map branch learns line-like geometry.

Some internal map-query features may still transfer.

V1 recommendation:

```text
reuse compatible map branch weights
reinitialize only shape-incompatible layers
```

Do not discard the entire map branch unless experiments show negative transfer.

This is the minimal-change approach.

---

## 72. New Anchor Initialization Verification

If Polygon anchors are newly initialized, visualize or inspect them if geometrically meaningful.

Check:

```text
coordinate range
distribution
NaN
collapse
```

If anchor tensors are learned parameters without direct geometric visualization, verify their shapes and initialization statistics.

---

## 73. Handling Original Map Dataset Fields

If the original dataset metadata always includes map annotations, the Polygon OCC dataset may no longer need them.

However, do not delete original map fields from preprocessing immediately.

Prefer:

```text
ignore unused map annotation
```

during the first implementation.

Large dataset-format refactors should be postponed.

---

## 74. Dataset Compatibility Strategy

Recommended:

```text
new Polygon OCC dataset config
```

rather than changing the default SparseDrive training config in place.

For example, create a new experiment config based on the original config.

Conceptually:

```text
original SparseDrive config
        ↓ inherit / copy
Polygon OCC V1 config
```

This preserves a baseline for comparison.

---

## 75. Experiment Config Naming

Use an explicit name.

Example concept:

```text
sparsedrive_polygon_occ_v1_32pts.py
```

The exact naming should follow repository style.

The config should make clear:

```text
Polygon OCC
V1
32 points
```

Avoid overwriting the official baseline config.

---

## 76. Required Baseline Preservation

Before changes:

```text
git status clean
baseline config preserved
```

Recommended:

```text
new git branch
```

Example:

```text
feature/polygon-occ-v1
```

Claude Code should not destructively edit the only working baseline.

---

## 77. Recommended Commit Stages

Use small commits.

### Commit 1

```text
Add Polygon OCC GT preprocessing
```

### Commit 2

```text
Integrate Polygon GT into dataset pipeline
```

### Commit 3

```text
Update map branch to 32 polygon vertices
```

### Commit 4

```text
Add Polygon OCC visualization and evaluation
```

### Commit 5

```text
Add smoke/overfit validation fixes
```

Do not combine the entire project into one opaque commit.

---

## 78. Required Unit Tests

At minimum, add or run tests for:

```text
polygon point-count configuration
empty GT shape
target normalization round-trip
regression output reshape
decoder output shape
polygon closure visualization logic
```

Geometry preprocessing tests are specified in the GT-generation document.

---

## 79. Required Shape Assertions During Development

Recommended temporary assertions:

```python
assert pred_pts.shape[-2:] == (num_polygon_points, 2)
assert gt_pts.shape[-2:] == (num_polygon_points, 2)
assert len(reg_weights) == num_polygon_points * 2
```

Use the equivalent repository style.

These assertions can catch silent 20/32-point mismatches.

---

## 80. Common Failure: `view` or `reshape` Still Uses 20

Symptom:

```text
invalid shape
```

or worse:

```text
tensor reshapes successfully into the wrong semantic grouping
```

Search all map-path `view` / `reshape` calls.

Use:

```text
num_points
```

rather than literal `20`.

---

## 81. Common Failure: Regression Weights Still Have Length 40

Symptom:

```text
broadcasting error
```

or:

```text
partial weighting
```

Fix map regression weight length to:

```text
64
```

Do not modify detection `reg_weights`.

---

## 82. Common Failure: Decoder Returns 32 Points but Evaluator Assumes 20

Symptom:

```text
visualization errors
metric failure
unexpected slicing
```

Audit all post-processing consumers.

Model training may succeed while evaluation fails.

---

## 83. Common Failure: Original Map Shift Logic Produces Multiple Polygon Targets

Symptom:

```text
unexpected target dimension
matching chooses strange cyclic variants
vertex loss unstable
```

Inspect line-shift augmentation/matching logic.

For canonical Polygon OCC V1, use one normalized target sequence.

---

## 84. Common Failure: Flip Augmentation Reverses Vertex Orientation

Symptom:

```text
same polygon geometry
high vertex L1
```

Re-run:

```text
orientation normalization
start-point normalization
```

after reflective augmentation.

---

## 85. Common Failure: GT and Prediction Use Different ROI Normalization

Symptom:

```text
predictions collapse
loss remains huge
visualization offset/scaled
```

Verify normalization round-trip.

Reuse original map ROI logic.

---

## 86. Common Failure: New Class Count Only Updated in Head

Symptom:

```text
label out of range
wrong visualization names
evaluation mismatch
```

Update all class-dependent components:

```text
dataset mapping
head
decoder
visualizer
evaluator
config
```

---

## 87. Common Failure: Pretrained Checkpoint Strict Load

Symptom:

```text
size mismatch for map regression layer
```

Use compatible partial loading.

Review skipped map-shape keys.

Do not disable all checkpoint loading.

---

## 88. Common Failure: Empty Polygon Sample Crashes Hungarian

Symptom:

```text
zero-size tensor error
```

Test explicit `N = 0` samples.

Follow the original map branch's empty-GT handling if available.

---

## 89. Common Failure: Self-Intersecting Predictions Break Shapely Evaluation

Training does not require valid polygons for Vertex L1.

Evaluation geometry libraries may fail or repair them unexpectedly.

Evaluation should:

```text
check validity
attempt documented repair
count invalid predictions
```

Do not silently discard invalid predictions from metrics without reporting them.

---

## 90. Common Failure: Same-Class Region Count Exceeds Queries

Symptom:

```text
many GT polygons never matched
low recall
```

Compare query count with GT P95/P99 statistics.

Increase queries only after confirming capacity limitation.

---

## 91. Recommended Runtime Debug Information

For the first training iterations, optionally report:

```text
num GT polygons per batch
matched polygon count
positive query count
mean vertex loss
class loss
pred coordinate min/max
GT coordinate min/max
```

This is more useful than only observing total loss.

---

## 92. Training Logging Recommendations

Track:

```text
loss_polygon_cls
loss_polygon_reg
matched_polygon_count
gt_polygon_count
```

If multiple decoder stages exist, preserve the normal per-stage loss names.

Do not rename all map loss keys immediately if existing training tools depend on them.

A comment or experiment documentation can clarify that the map branch now represents Polygon OCC.

---

## 93. Polygon-Specific Debug Metrics During Training

Optional non-gradient metrics:

```text
valid predicted polygon ratio
mean predicted area
mean GT area
mean rasterized IoU on matched pairs
```

These can be expensive.

Do not compute them every iteration.

Use:

```text
periodic validation
```

or a dedicated debug hook/script.

---

## 94. Recommended Evaluation Script Separation

Do not overload the training loss module with Polygon evaluation geometry.

Preferred:

```text
training:
classification + vertex loss

evaluation script/module:
polygon validity
rasterization
IoU
area metrics
```

This keeps V1 loss minimal.

---

## 95. Minimal Implementation Decision Tree

When Claude Code finds a map module, ask:

### Question 1

Does it depend on number of map points?

If no:

```text
leave unchanged
```

If yes:

```text
make it use 32 / config value
```

### Question 2

Does it assume an open LineString?

If no:

```text
reuse
```

If yes:

```text
adapt for closed polygon semantics
```

### Question 3

Does it implement map-specific topology or shifted sequences?

If no:

```text
reuse
```

If yes:

```text
compare with canonical Polygon target design
```

### Question 4

Can existing behavior be parameterized?

If yes:

```text
parameterize
```

If no:

```text
add minimal Polygon-specific branch
```

Do not rewrite first.

---

## 96. Recommended Claude Code Work Log

For every modified file, record:

```text
File:
Function/Class:

Original role:

Polygon OCC change:

Tensor shape before:

Tensor shape after:

Why modification is required:

Risk:
```

This can be maintained in a temporary implementation note.

It helps review accidental over-modification.

---

## 97. Definition of "Minimal Modification"

Minimal modification does not mean:

```text
change only one line regardless of correctness
```

It means:

> Preserve all architectural behavior that is still semantically valid and change only behavior incompatible with polygon region prediction.

Examples:

### Valid Minimal Change

```text
20 points → configurable 32 points
```

### Valid Required Change

```text
disable LineString-specific reverse/shift target logic
```

### Invalid Over-Modification

```text
replace Hungarian with custom Polygon matcher without evidence
```

### Invalid Under-Modification

```text
keep open-polyline anchor geometry even though it corrupts Polygon initialization
```

Correctness has priority over superficial line-count minimization.

---

## 98. V1 Acceptance Criteria

The implementation is complete only when all conditions are satisfied.

### Data

```text
[ ] Polygon GT aligns with SparseDrive samples.
[ ] Per-sample shape is [N, 32, 2].
[ ] Empty GT is supported.
[ ] Class labels are correct.
[ ] Polygon ordering is deterministic.
```

### Model

```text
[ ] Existing sparse map queries are reused.
[ ] Regression output supports 32 × 2.
[ ] No dense OCC head exists.
[ ] Detection branch remains functionally unchanged.
```

### Matching and Loss

```text
[ ] Hungarian matching runs.
[ ] Pairwise cost shape is correct.
[ ] Vertex regression loss is finite.
[ ] Regression weights match 64 values if required.
[ ] No Raster Loss is used.
```

### Decoder

```text
[ ] Decoder outputs [N, 32, 2] polygons.
[ ] Polygon closure is implicit.
[ ] Score/label filtering works.
```

### Visualization

```text
[ ] GT polygons close correctly.
[ ] Predicted polygons close correctly.
[ ] Vertex 0 can be debugged.
[ ] Coordinate orientation is visually correct.
```

### Training

```text
[ ] Forward smoke test passes.
[ ] Backward smoke test passes.
[ ] Regression branch receives gradient.
[ ] Tiny-subset overfit succeeds.
```

### Evaluation

```text
[ ] Polygon rasterization is available for evaluation.
[ ] Polygon IoU / BEV semantic IoU can be computed.
[ ] Invalid polygon ratio is reported.
```

---

## 99. Recommended Implementation Summary for Claude Code

Claude Code should implement Polygon OCC V1 using the following strategy:

```text
Step 1
Trace the existing SparseDrive map path.

Step 2
Integrate validated 32-point Polygon GT into the map target interface.

Step 3
Audit map-specific point transforms and disable LineString-only sequence logic.

Step 4
Change the active map point count to 32.

Step 5
Update every map regression dimension derived from point count.

Step 6
Inspect map anchors / point encoders and adapt only shape-dependent components.

Step 7
Reuse existing map Hungarian matching with direct canonical vertex regression cost.

Step 8
Reuse existing map regression loss.

Step 9
Update map decoder to interpret 32 points as a closed polygon.

Step 10
Update visualization and add rasterized evaluation.

Step 11
Run DataLoader, forward, backward, and tiny-subset overfit tests.

Step 12
Only after all checks pass, run full training.
```

---

## 100. Final Implementation Principle

The implementation should preserve the following identity:

```text
SparseDrive
+
minimal map representation change
=
Polygon OCC V1
```

It should not become:

```text
SparseDrive
+
new dense OCC architecture
+
new matcher
+
new losses
+
new decoder
```

The first experiment must remain simple enough to answer:

> Can the original SparseDrive sparse map-query framework learn semantic occupied regions as fixed 32-point closed polygons?

Every code modification should be justified against that question.

When uncertain, prefer:

```text
inspect
reuse
parameterize
test
```

before:

```text
duplicate
rewrite
redesign
```

V1 is a representation experiment first and an architecture experiment only when unavoidable.
