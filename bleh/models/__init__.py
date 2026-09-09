"""Model package initialization."""

from .fast_scnn import FastSCNN


def build_model(model_name: str = "fast_scnn", num_classes: int = 4, **kwargs) -> FastSCNN:
    """
    Factory function to build real-time segmentation models.

    Args:
        model_name: Name of architecture ('fast_scnn').
        num_classes: Number of semantic output classes.
        **kwargs: Additional parameters passed to model constructor.

    Returns:
        torch.nn.Module instance.
    """
    if model_name.lower() == "fast_scnn":
        return FastSCNN(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model architecture: {model_name}. Supported: ['fast_scnn']")


__all__ = ["FastSCNN", "build_model"]
