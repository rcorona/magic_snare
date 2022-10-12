from models.single_cls import SingleClassifier
from models.zero_shot_cls import ZeroShotClassifier
from models.rotator import Rotator
from models.transformer import TransformerClassifier
from models.pixelnerf import PixelNeRFClassifier
from models.lfn import LFNClassifier

names = {
    # classifiers
    'single_cls': SingleClassifier,
    'zero_shot_cls': ZeroShotClassifier,

    # rotators
    'rotator': Rotator,

    # Transformer
    'transformer': TransformerClassifier,

    # PixelNeRF
    'pixelnerf': PixelNeRFClassifier,

    'lfn': LFNClassifier
}
