"""
Checkpoint Utils for Models

Author: Zongze Wu (zongze0907@gmail.com)
Please cite our work if the code is helpful to you.
"""

from utils.registry import Registry

MODELS = Registry("models")
MODULES = Registry("modules")


def build_model(cfg):
    """Build models."""
    return MODELS.build(cfg)
