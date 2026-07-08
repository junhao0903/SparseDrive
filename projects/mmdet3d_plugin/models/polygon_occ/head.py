from mmdet.models import HEADS

from ..detection3d.detection3d_head import Sparse4DHead


@HEADS.register_module()
class PolygonOccHead(Sparse4DHead):
    """Thin Sparse4DHead wrapper for Polygon OCC configs.

    This keeps the existing Sparse4DHead execution flow untouched while
    allowing configs to select a dedicated Polygon OCC head type instead of
    reusing the map_head entry directly.
    """

    pass
