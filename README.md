# VGGT: Visual Geometry Grounded Transformer FiftyOne Remote Source Zoo Model Integration

This repository provides a FiftyOne Zoo Model for **VGGT (Visual Geometry Grounded Transformer)**, enabling seamless 3D scene reconstruction from single images with integrated visualization capabilities.

<img src="vggt-lq.gif">

## Overview

VGGT takes a single RGB image as input and produces:
- **Dense depth maps** with confidence scores
- **Camera pose estimation** (extrinsic and intrinsic parameters)  
- **Dense 3D point clouds** from depth map unprojection
- **Dynamic camera orientation** for optimal 3D visualization

## Installation

```bash
# Install FiftyOne
pip install fiftyone

```

You also need to install the following:

```bash
pip install vggt@git+https://github.com/facebookresearch/vggt.git
pip install open3d
```

### Register the VGGT Zoo Model source

```import fiftyone.zoo as foz
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/vggt",
    overwrite=True
)
```

## Quick Start

### 1. Load the Model

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Load VGGT model from the zoo
model = foz.load_zoo_model("facebook/VGGT-1B")
```

### 2. Apply to Your Dataset

```python
# Load your dataset
dataset = fo.load_dataset("your_dataset")

# Apply VGGT model to generate depth maps and 3D reconstructions
dataset.apply_model(model, "depth_map_path")
```

### 3. Create Grouped Dataset for Multi-Modal Visualization

**Important**: The VGGT model runs and saves results alongside the original input files, but these are not automatically added to your dataset. Use the following code to create a grouped dataset for comprehensive visualization:

```python
import fiftyone as fo
import os
from pathlib import Path

# Get filepaths from your existing dataset
filepaths = dataset.values("filepath")

# Create a new grouped dataset
grouped_dataset = fo.Dataset("vggt_results", overwrite=True)
grouped_dataset.add_group_field("group", default="rgb")

# Process each filepath and create the group structure
samples = []
for filepath in filepaths:
    # Extract base information from the filepath
    path = Path(filepath)
    base_dir = path.parent
    base_name = path.stem
    
    # Create paths for each modality following your pattern
    rgb_path = filepath  # Original filepath (RGB)
    depth_path = os.path.join(base_dir, f"{base_name}_depth.png")  # Depth map
    threed_path = os.path.join(base_dir, f"{base_name}.fo3d")  # 3D point cloud
    
    # Create a group for these related samples
    group = fo.Group()
    
    # Create a sample for each modality with the appropriate group element
    rgb_sample = fo.Sample(filepath=rgb_path, group=group.element("rgb"))
    depth_sample = fo.Sample(filepath=depth_path, group=group.element("depth"))
    threed_sample = fo.Sample(filepath=threed_path, group=group.element("threed"))
    
    # Add samples to the list
    samples.extend([rgb_sample, depth_sample, threed_sample])

# Add all samples to the dataset
grouped_dataset.add_samples(samples)

# Display the dataset in the FiftyOne App
fo.launch_app(grouped_dataset)
```

## Output Files

For each input image VGGT generates:

- **`image_depth.png`**: Colorized depth map for heatmap visualization
- **`image.pcd`**: 3D point cloud in PCD format  
- **`image.fo3d`**: FiftyOne 3D scene with dynamic camera orientation

## Configuration Options

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence_threshold` | float | 51.0 | Percentile threshold (0-100) for point filtering |
| `mode` | str | "pad" | Image preprocessing mode |

### Preprocessing Modes

The `mode` parameter controls how images are preprocessed for VGGT:

- **`mode="crop"`**: Ensures width=518px while maintaining aspect ratio. Height is center-cropped if larger than 518px
- **`mode="pad"`**: Ensures the largest dimension is 518px while maintaining aspect ratio. The smaller dimension is padded to reach a square shape (518x518)

### Example Configuration

```python
# Load model with custom configuration
model = foz.load_zoo_model(
    "facebook/VGGT-1B",
    confidence_threshold=75.0,  # More aggressive filtering
    mode="crop"                 # Use crop instead of pad
)
```

## Camera Orientation

The implementation **Sets FiftyOne camera orientation statically to `up="Z"`**

## Roadmap

### Current Release
- ✅ Depth map generation and visualization
- ✅ 3D point cloud reconstruction  
- ✅ FiftyOne Zoo Model integration

### Future Releases
- 🔄 Camera parameter extraction and export
- 🔄 Positions camera at the actual VGGT camera location with correct viewing direction
- 🔄 Dynamic camera orientation
- 🔄 Multi-view reconstruction support
- 🔄 Camera animation for video sequences
- 🔄 Advanced scene analysis tools


## License

This integration wrapper follows the same license as the underlying VGGT model. [Please refer to the original VGGT repository for licensing details](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request with clear description


## Citation

If you use this FiftyOne integration in your research, please cite both VGGT and FiftyOne:

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

@misc{fiftyone,
  title={FiftyOne},
  author={Voxel51},
  year={2020},
  url={https://github.com/voxel51/fiftyone}
}
```