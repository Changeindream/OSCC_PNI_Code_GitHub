# models/__init__.py

from .swin_transformer import (
    swin_base_patch4_window7_224,
    swin_base_patch4_window12_384,
    swin_large_patch4_window12_384,
    swin_large_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_tiny_patch4_window7_224,
    # 如果有其他模型，也可以在这里添加
)

__all__ = [
    'swin_base_patch4_window7_224',
    'swin_base_patch4_window12_384',
    'swin_large_patch4_window12_384',
    'swin_large_patch4_window7_224',
    'swin_small_patch4_window7_224',
    'swin_tiny_patch4_window7_224',
    # 其他模型名称
]
