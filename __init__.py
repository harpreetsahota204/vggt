import os
import torch
import logging

from fiftyone.operators import types

from vggt.models.vggt import VGGT

from .zoo import VGGTModelConfig, VGGTModel, VGGTOutputProcessor

logger = logging.getLogger(__name__)

# Model variants and their configurations
MODEL_VARIANTS = {
    "facebook/VGGT-1B": {
        "model_name": "facebook/VGGT-1B",
        "description": "VGGT 1B parameter model for 3D scene reconstruction"
    },
}

def download_model(model_name, model_path):
    """Downloads the VGGT model.
    
    Args:
        model_name: the name of the model to download
        model_path: the absolute filename or directory to which to download the model
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    model_info = MODEL_VARIANTS[model_name]
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    logger.info(f"Downloading VGGT model {model_name}...")
    
    # Load model from HuggingFace
    model = VGGT.from_pretrained(model_name)
    
    # Save the COMPLETE model (architecture + weights) to disk
    torch.save(model, model_path)
    
    logger.info(f"VGGT model {model_name} saved to {model_path}")


def load_model(
    model_name, 
    model_path, 
    confidence_threshold=51.0,
    mode="pad",
    **kwargs
):
    """Loads the VGGT model.
    
    Args:
        model_name: the name of the model to load
        model_path: the absolute filename to which the model was downloaded
        confidence_threshold: confidence threshold for point cloud filtering (percentile)
        **kwargs: additional keyword arguments
        
    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    config_dict = {
        "model_name": model_name,
        "model_path": model_path,
        "confidence_threshold": confidence_threshold,
        "mode": mode,
        **kwargs
    }
    
    # Set up single output processor for all outputs
    config_dict["output_processor_cls"] = VGGTOutputProcessor
    config_dict["output_processor_args"] = {
        "confidence_threshold": confidence_threshold,
    }
    
    config = VGGTModelConfig(config_dict)
    return VGGTModel(config)


def resolve_input(model_name, ctx):
    """Defines any necessary properties to collect the model's custom
    parameters from a user during prompting.
    
    Args:
        model_name: the name of the model
        ctx: an :class:`fiftyone.operators.ExecutionContext`
        
    Returns:
        a :class:`fiftyone.operators.types.Property`, or None
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    inputs = types.Object()
    
    inputs.float(
        "confidence_threshold",
        default=51.0,
        label="Confidence Threshold",
        description="Confidence threshold percentile for point cloud filtering (0-100)"
    )
    
    
    return types.Property(inputs)
