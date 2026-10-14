"""Paired RGB-SAR datasets."""

from mmseg.datasets.basesegdataset import BaseCDDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class PairedDDHRSKDataset(BaseCDDataset):
    """DDHRSK semantic segmentation with paired RGB and SAR images.

    ``img_path`` is the RGB path and ``img_path2`` is the SAR path.  Pairing
    is performed by relative filename, as implemented by ``BaseCDDataset``.
    """

    METAINFO = dict(
        classes=('Building', 'Road', 'Greenery', 'Water', 'Farmland'),
        palette=[[79, 100, 127], [148, 160, 190], [163, 193, 165],
                 [20, 99, 191], [200, 197, 164]])

    def __init__(self,
                 img_suffix='.png',
                 img_suffix2='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            img_suffix2=img_suffix2,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

@DATASETS.register_module()
class PairedGEONRWDataset(BaseCDDataset):
    """DDHRSK semantic segmentation with paired RGB and SAR images.

    ``img_path`` is the RGB path and ``img_path2`` is the SAR path.  Pairing
    is performed by relative filename, as implemented by ``BaseCDDataset``.
    """

    METAINFO = dict(classes=('forest', 'water', 'agricultural', 'residential,commercial,industrial',
                             'grassland,swamp,shrubbery', 'railway,trainstation', 'highway,squares',
                             'airport,shipyard', 'roads', 'buildings'),
                    palette=[[0, 255, 0], [0, 0, 255], [255, 195, 128], [79, 100, 127],
                             [189,255,0], [148, 160, 190], [195,176,58],
                            [249,255,159], [91,255,129], [165,165,165]])

    def __init__(self,
                 img_suffix='.png',
                 img_suffix2='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            img_suffix2=img_suffix2,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

@DATASETS.register_module()
class PairedWHUDataset(BaseCDDataset):

    METAINFO = dict(classes=('farmland', 'city', 'village', 'water', 'forest',
                             'road', 'other'),
                    palette=[[255, 195, 128], [255, 0, 0], [159, 129, 183],
                             [0, 0, 255], [0, 255, 0], [255, 255, 0],
                             [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 img_suffix2='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            img_suffix2=img_suffix2,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

@DATASETS.register_module()
class PairedNYUDataset(BaseCDDataset):
    METAINFO = dict(
        classes=(
            'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
            'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds',
            'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
            'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator',
            'television', 'paper', 'towel', 'shower curtain', 'box',
            'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp',
            'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop',
        ),
        # NYU40 colors for semantic IDs 1..40, shifted to train IDs 0..39.
        # The black color for disk ID 0 is deliberately omitted: it is ignore.
        palette=[
            [174, 199, 232], [152, 223, 138], [31, 119, 180],
            [255, 187, 120], [188, 189, 34], [140, 86, 75],
            [255, 152, 150], [214, 39, 40], [197, 176, 213],
            [148, 103, 189], [196, 156, 148], [23, 190, 207],
            [178, 76, 76], [247, 182, 210], [66, 188, 102],
            [219, 219, 141], [140, 57, 197], [202, 185, 52],
            [51, 176, 203], [200, 54, 131], [92, 193, 61],
            [78, 71, 183], [172, 114, 82], [255, 127, 14],
            [91, 163, 138], [153, 98, 156], [140, 153, 101],
            [158, 218, 229], [100, 125, 154], [178, 127, 135],
            [120, 185, 128], [146, 111, 194], [44, 160, 44],
            [112, 128, 144], [96, 207, 209], [227, 119, 194],
            [213, 92, 176], [94, 106, 211], [82, 84, 163],
            [100, 85, 144],
        ],
    )

    def __init__(self,
                 img_suffix='.png',
                 img_suffix2='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            img_suffix2=img_suffix2,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class PairedSUNRGBDDataset(BaseCDDataset):
    METAINFO = dict(
        classes=(
            'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
            'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves',
            'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes',
            'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel', 'shower_curtain',
            'box', 'whiteboard', 'person', 'night_stand', 'toilet', 'sink', 'lamp',
            'bathtub', 'bag',
        ),
        palette=[
            [174, 199, 232], [152, 223, 138], [31, 119, 180],
            [255, 187, 120], [188, 189, 34], [140, 86, 75],
            [255, 152, 150], [214, 39, 40], [197, 176, 213],
            [148, 103, 189], [196, 156, 148], [23, 190, 207],
            [178, 76, 76], [247, 182, 210], [66, 188, 102],
            [219, 219, 141], [140, 57, 197], [202, 185, 52],
            [51, 176, 203], [200, 54, 131], [92, 193, 61],
            [78, 71, 183], [172, 114, 82], [255, 127, 14],
            [91, 163, 138], [153, 98, 156], [140, 153, 101],
            [158, 218, 229], [100, 125, 154], [178, 127, 135],
            [120, 185, 128], [146, 111, 194], [44, 160, 44],
            [112, 128, 144], [96, 207, 209], [227, 119, 194],
            [213, 92, 176],
        ],
    )

    def __init__(self,
                 img_suffix='.png',
                 img_suffix2='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            img_suffix2=img_suffix2,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)