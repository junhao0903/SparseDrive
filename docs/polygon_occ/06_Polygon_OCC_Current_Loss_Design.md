# 06_Polygon_OCC_Current_Loss_Design

## 1. Purpose

This document explains the **current Polygon OCC loss design** used in this repository.

The goal is to make clear:

- what losses are currently used,
- what each loss supervises,
- how line-related and face-related supervision are combined,
- and what limitation still remains in the current design.

---

## 2. Overall Design

The current `polygon_occ` training objective can be understood as:

```text
classification loss
+ line-level geometry loss
+ shape-aware auxiliary loss
+ rasterized face-related auxiliary loss
```

In practice, the current loss structure is approximately:

```text
L_total
=
L_cls
+ L_line
+ λ_chamfer L_chamfer
+ λ_raster L_raster
```

This means each Polygon OCC query is supervised from four angles:

1. semantic category,
2. ordered polygon boundary,
3. global polygon shape,
4. polygon rasterized occupancy region.

---

## 3. Classification Loss

### `loss_cls`

The classification loss is:

- **FocalLoss**

Current configuration:

```python
loss_cls=dict(
    type="FocalLoss",
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0,
)
```

Purpose:

- supervise the semantic occupancy class of each polygon query,
- reduce the impact of positive/negative imbalance.

---

## 4. Line-Related Main Loss

### `loss_line`

The main polygon regression loss is:

- **LinesL1Loss**

This is the current ordered polygon boundary supervision.

It is applied after:

1. Hungarian matching,
2. permutation-aware GT selection,
3. matched positive filtering.

So the effective path is:

```text
GT polygon points
→ permutation-expanded target
→ Hungarian matching with permute=True
→ best GT ordering selected
→ LinesL1Loss
```

This loss supervises:

- ordered vertex coordinates,
- local point-to-point boundary correspondence,
- canonical polygon representation stability.

It is the core geometry loss in the current implementation.

---

## 5. Shape-Aware Auxiliary Loss

### `loss_chamfer`

The current auxiliary shape loss is:

- **PolygonChamferLoss**

Chamfer loss compares the predicted polygon vertices and GT polygon vertices as two point sets.

Conceptually:

```text
CD(P, G)
=
Σ_{p∈P} min_{g∈G} ||p-g||
+
Σ_{g∈G} min_{p∈P} ||g-p||
```

Purpose:

- reduce sensitivity to cyclic shift or start-point misalignment,
- improve global boundary-shape supervision,
- complement the ordered `loss_line` term.

Role in the current design:

- `loss_line` supervises **ordered polygon boundary expression**,
- `loss_chamfer` supervises **overall boundary shape similarity**.

So it is not intended to replace `loss_line`, but to supplement it.

---

## 6. Face-Related Auxiliary Loss

### `loss_raster`

The current face-related auxiliary term is:

- **PolygonRasterLoss**

This loss rasterizes the predicted polygon and GT polygon into a small BEV occupancy grid and compares the resulting masks.

```text
pred polygon → soft raster mask
gt polygon   → soft raster mask
→ soft Dice loss
```

Purpose:

- add a face-level constraint,
- ensure the predicted polygon does not only have a reasonable boundary,
  but also a reasonable occupied region.

This is the current rasterized face-related supervision in the codebase.

---

## 7. Why This Is Called “Line + Face” Supervision

The current design combines:

### Line-related supervision

- `loss_line`
- `loss_chamfer`

These losses mainly constrain the polygon **boundary**.

### Face-related supervision

- `loss_raster`

This loss constrains the polygon **enclosed region** through occupancy-mask similarity.

Therefore, the current design already includes both:

- **line-related loss**, and
- **face-related loss**.

This gives the current design both:

- **line-related supervision**, and
- **face-related occupancy supervision**.

---

## 8. Current Limitation

Although `loss_raster` provides actual occupancy-overlap supervision, it still has some limitations.

It improves:

- occupied-region overlap,
- filled occupancy support,
- and overall face-level shape consistency.

However, it still does **not** directly guarantee:

- polygon validity,
- clean ordered vector geometry,
- or correct topology in the vector domain.

That is why it is kept alongside:

- `loss_line`, and
- `loss_chamfer`

rather than replacing them.

---

## 9. Current Practical Interpretation

At the current stage, the Polygon OCC loss design is:

```text
classification supervision
+ precise ordered boundary supervision
+ global shape supervision
+ rasterized occupancy supervision
```

More concretely:

- **FocalLoss** teaches semantic class prediction,
- **LinesL1Loss** teaches ordered polygon boundaries,
- **PolygonChamferLoss** teaches overall polygon shape,
- **PolygonRasterLoss** teaches occupancy-region overlap.

---

## 10. Recommended Next Step

The current design already includes a rasterized occupancy loss. The next natural extensions would be:

```text
stronger raster formulations
e.g. Dice + BCE
or more advanced differentiable rasterization variants
```

At the current stage, the design should still be viewed as a staged compromise:

- keep the sparse polygon representation trainable,
- retain strong line-level supervision,
- and introduce true occupancy-style face supervision without rewriting the full architecture.
