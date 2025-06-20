import os
import logging
import numpy as np
import torch
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torchvision.transforms.functional as TF
import torchvision
import open3d as o3d
from PIL import Image
from skimage.transform import resize

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout

logger = logging.getLogger(__name__)


class VGGTModelConfig(fout.TorchImageModelConfig):
    """Configuration for running a VGGT model for 3D scene reconstruction.
    
    This configuration class defines all the parameters needed to set up and run
    VGGT (Video-Grounded Generation and Tracking) models for single-image 3D
    reconstruction tasks.

    Args:
        model_name (str): The VGGT model identifier (e.g., "facebook/VGGT-1B").
            This corresponds to the HuggingFace model repository name.
        model_path (str): Absolute path to the saved model file on disk.
            The model should be saved as a complete PyTorch model using torch.save().
        confidence_threshold (float): Percentile threshold (0-100) for filtering
            point cloud points based on depth confidence. Higher values result in
            fewer but more reliable points. Default is 51.0.
    """

    def __init__(self, d):
        super().__init__(d)
        self.model_name = self.parse_string(d, "model_name", default="facebook/VGGT-1B")
        self.model_path = self.parse_string(d, "model_path")
        self.confidence_threshold = self.parse_number(d, "confidence_threshold", default=51.0)
        self.mode = self.parse_string(d, "mode", default="pad")


class VGGTModel(fout.TorchImageModel, fout.TorchSamplesMixin):
    """VGGT model wrapper for 3D scene reconstruction from single images.
    
    This class implements a FiftyOne-compatible wrapper around VGGT (Video-Grounded
    Generation and Tracking) models. VGGT takes a single RGB image as input and
    produces:
    
    1. Dense depth maps with confidence scores
    2. Camera pose estimation (extrinsic and intrinsic parameters)
    3. Dense 3D point clouds from depth map unprojection
    
    The model outputs are saved as:
    - Depth map PNG: Colorized depth visualization for FiftyOne heatmap display
    - PCD file: Dense 3D point cloud in standard format
    - fo3d file: 3D point cloud for FiftyOne 3D visualization
    
    The primary FiftyOne output is a Heatmap label pointing to the depth PNG,
    while auxiliary files are co-located with the original images for post-processing.

    Args:
        config (VGGTModelConfig): Configuration object containing model parameters.
    """

    def __init__(self, config):
        super().__init__(config)
        
        # Explicitly set device
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine optimal precision based on GPU capabilities
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        
        # Initialize fields dict for SamplesMixin
        self._fields = {}

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields

    def _load_model(self, config):
        """Load the pre-downloaded VGGT model from disk.
        
        This method loads a complete VGGT model that was previously downloaded
        and saved using torch.save(). The model includes both architecture and
        trained weights.
        
        Args:
            config (VGGTModelConfig): Configuration containing model_path
            
        Returns:
            torch.nn.Module: The loaded VGGT model ready for inference
            
        Raises:
            FileNotFoundError: If the model file doesn't exist at the specified path
        """
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model file not found: {config.model_path}")
        
        logger.info(f"Loading VGGT model from {config.model_path}")
        
        # Load the complete model (architecture + weights) from disk
        # weights_only=False allows loading the full model object
        model = torch.load(config.model_path, map_location='cpu', weights_only=False)
        model = model.to(self._device)
        model.eval()

        return model

    def predict_all(self, imgs, samples=None):
        """Perform VGGT inference on images (FiftyOne processes one at a time).
        
        Args:
            imgs: Single image tensor from FiftyOne preprocessing
            samples: Single sample object containing filepath and metadata
                
        Returns:
            Single prediction dict for the output processor
        """
        if samples is None:
            raise ValueError("VGGT model requires sample objects to access filepaths")
        
        # FiftyOne passes containers with single items, extract them
        if isinstance(imgs, list):
            img = imgs[0]  # Extract single image from list
        else:
            img = imgs
            
        if isinstance(samples, (tuple, list)):
            sample = samples[0]  # Extract single sample from tuple/list
        else:
            sample = samples
        
        try:
            # Extract file path information for saving auxiliary outputs
            original_path = Path(sample.filepath)
            base_dir = original_path.parent
            base_name = original_path.stem
            
            # Define output paths for all auxiliary files
            depth_png_path = base_dir / f"{base_name}_depth.png"      # Colorized depth for heatmap
            fo3d_path = base_dir / f"{base_name}.fo3d"                # 3D point cloud
            
            # Preprocess image using VGGT-specific requirements
            img_tensor, original_size = self._preprocess_vggt_image(sample)
            vggt_size = (img_tensor.shape[-1], img_tensor.shape[-2])  # (width, height)
            
            # Add batch dimension for model inference
            img_batch = img_tensor.unsqueeze(0)
            
            # Run VGGT inference with automatic mixed precision
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    # Run full VGGT model prediction
                    vggt_predictions = self._model(img_batch)
                    
                    # Convert VGGT's pose encoding to standard camera matrices
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        vggt_predictions["pose_enc"], img_batch.shape[-2:]
                    )
            
            # Package all VGGT outputs for file saving and processing
            vggt_output = {
                "depth": vggt_predictions["depth"].cpu().numpy().squeeze(0),
                "depth_conf": vggt_predictions["depth_conf"].cpu().numpy().squeeze(0),
                "world_points_conf": vggt_predictions["world_points_conf"].cpu().numpy().squeeze(0),
                "extrinsic": extrinsic.cpu().numpy().squeeze(0),
                "intrinsic": intrinsic.cpu().numpy().squeeze(0),
                "images": img_tensor.cpu().numpy(),
                "original_size": original_size,
                "vggt_size": vggt_size,
            }
            
            # Save auxiliary files for post-processing
            self._save_depth_png(vggt_output, depth_png_path)
            self._save_fo3d(vggt_output, fo3d_path)
            
            # Package prediction data for output processor
            prediction_data = {
                'depth_map_path': str(depth_png_path),  # Primary output for heatmap
                'vggt_output': vggt_output,             # Complete results for debugging
            }
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error processing sample {sample.filepath}: {e}")
            return None

    def _preprocess_vggt_image(self, sample):
        """Preprocess image using VGGT's built-in preprocessing function."""
        # Use the original file path directly - much more efficient!
        original_path = sample.filepath
        
        # Get original size efficiently
        with Image.open(original_path) as img_pil:
            original_size = img_pil.size  # (width, height)
        
        # Use VGGT's built-in preprocessing directly on original file
        images = load_and_preprocess_images([original_path], mode=self.config.mode).to(self._device)
        
        # Remove batch dimension since we're processing single images
        img_tensor = images.squeeze(0)  # Remove batch dim: [1, C, H, W] -> [C, H, W]
        
        vggt_size = (img_tensor.shape[-1], img_tensor.shape[-2])  # (width, height)
        
        return img_tensor, original_size

    def _save_depth_png(self, vggt_output: Dict, output_path: Path):
        """Save depth map as a colorized PNG at VGGT's native resolution.
        
        Converts VGGT's raw depth predictions into a visually interpretable
        colorized image that can be displayed as a heatmap in FiftyOne. The depth
        values are normalized and colored using OpenCV's JET colormap.
        
        Processing steps:
        1. Keep depth map at VGGT's native resolution (no resizing)
        2. Handle invalid/infinite depth values
        3. Normalize to [0, 255] range using percentile scaling
        4. Apply JET colormap for visualization
        5. Save as PNG file
        
        Args:
            vggt_output (Dict): Complete VGGT inference results
            output_path (Path): Destination path for depth PNG file
        """
        try:
            # Extract depth map and remove any extra dimensions
            depth_vis = vggt_output["depth"].squeeze()
            
            # Keep depth at VGGT's native resolution - no resizing
            # This preserves the quality and accuracy of the depth predictions
            vggt_w, vggt_h = vggt_output["vggt_size"]
            logger.debug(f"Keeping depth map at VGGT resolution: {depth_vis.shape} (VGGT size: {vggt_w}x{vggt_h})")
            
            # Handle invalid depth values (NaN, infinity, negative)
            valid_mask = np.isfinite(depth_vis) & (depth_vis > 0)
            
            if np.any(valid_mask):
                # Use robust percentile-based normalization to handle outliers
                depth_min = np.percentile(depth_vis[valid_mask], 5)   # Ignore bottom 5%
                depth_max = np.percentile(depth_vis[valid_mask], 95)  # Ignore top 5%
                
                # Normalize to [0, 1] range
                if depth_max > depth_min:
                    depth_normalized = np.clip(
                        (depth_vis - depth_min) / (depth_max - depth_min), 0, 1
                    )
                else:
                    # Handle edge case where all valid depths are the same
                    depth_normalized = np.zeros_like(depth_vis)
            else:
                # No valid depth values found
                depth_normalized = np.zeros_like(depth_vis)
            
            # Set invalid regions to zero (will appear as dark blue in JET colormap)
            depth_normalized[~valid_mask] = 0
            
            # Convert to 8-bit integer for image saving
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            
            # Apply JET colormap: blue (near) -> green -> yellow -> red (far)
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
            
            # Save as PNG (lossless compression)
            cv2.imwrite(str(output_path), depth_colored)
            logger.debug(f"Saved depth map at native resolution to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving depth PNG to {output_path}: {e}")

    def _extract_camera_parameters(self, vggt_output):
        """Extract camera position, rotation, and other parameters from VGGT output."""
        extrinsic = np.squeeze(vggt_output["extrinsic"])
        intrinsic = np.squeeze(vggt_output["intrinsic"])
        
        # Handle different extrinsic matrix shapes
        if extrinsic.shape == (3, 4):
            R = extrinsic[:3, :3]  # Rotation matrix (world-to-camera)
            t = extrinsic[:, 3]    # Translation vector
        elif extrinsic.shape == (4, 4):
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
        else:
            # Fallback - no camera positioning
            return None
        
        # Camera position in world coordinates (for world-to-camera transform)
        camera_position = -R.T @ t
        
        # Camera orientation vectors in world coordinates
        R_cam_to_world = R.T
        camera_forward = R_cam_to_world @ np.array([0, 0, 1])  # Camera Z-axis (forward)
        camera_up = R_cam_to_world @ np.array([0, -1, 0])     # Camera -Y-axis (up in OpenCV)
        
        return {
            "position": camera_position.tolist(),
            "forward": camera_forward.tolist(),
            "up": camera_up.tolist(),
            "rotation_matrix": R_cam_to_world.tolist()
        }

    def _determine_camera_up_direction(self, extrinsic_matrix):
        """Determine the best up direction for FiftyOne's 3D viewer based on camera pose."""
        
        # Handle different extrinsic matrix shapes
        extrinsic = np.squeeze(extrinsic_matrix)
        
        if extrinsic.shape == (3, 4):
            R = extrinsic[:3, :3]  # Rotation matrix (world-to-camera)
        elif extrinsic.shape == (4, 4):
            R = extrinsic[:3, :3]
        elif extrinsic.shape == (3, 3):
            R = extrinsic  # Already just rotation
        else:
            # Fallback to default
            return "Y"
        
        # Camera-to-world rotation matrix  
        R_cam_to_world = R.T
        
        # Assume world Y-axis points up (standard gravity direction)
        world_up = np.array([0, 1, 0])
        
        # Find which camera axis best aligns with world up (FiftyOne only supports X, Y, Z)
        camera_axes = {
            "X": np.array([1, 0, 0]),
            "Y": np.array([0, 1, 0]),
            "Z": np.array([0, 0, 1])
        }
        
        # Transform world up to camera coordinates
        world_to_cam_R = R_cam_to_world.T
        world_up_in_cam = world_to_cam_R @ world_up
        
        best_alignment = -1
        best_axis = "Y"  # Default fallback
        
        for axis_name, axis_vec in camera_axes.items():
            # Check both positive and negative alignment, but only return positive axes
            alignment = abs(np.dot(world_up_in_cam, axis_vec))
            if alignment > best_alignment:
                best_alignment = alignment
                best_axis = axis_name
        
        return best_axis

    def _determine_scene_camera_setup(self, vggt_output):
        """Determine both camera position and up direction for the scene."""
        
        # Extract camera parameters
        camera_params = self._extract_camera_parameters(vggt_output)
        
        if camera_params is None:
            # Fallback to just up direction
            up_direction = self._determine_camera_up_direction(vggt_output["extrinsic"])
            return {"up": up_direction, "center": None}
        
        # Determine up direction
        up_direction = self._determine_camera_up_direction(vggt_output["extrinsic"])
        
        # Camera position for scene viewing
        camera_position = camera_params["position"]
        
        return {
            "up": up_direction,
            "center": camera_position,  # Position the scene camera at the VGGT camera location
            "camera_params": camera_params
        }

    def _save_fo3d(self, vggt_output: Dict, output_path: Path):
        """Save 3D point cloud with VGGT camera positioning.
        
        Converts VGGT's depth map and camera parameters into a colored 3D point
        cloud suitable for visualization in FiftyOne's 3D viewer. The scene camera
        is positioned at the actual VGGT camera location for authentic viewpoint.
        
        Processing pipeline:
        1. Unproject depth map to 3D world coordinates using camera parameters
        2. Extract corresponding colors from the input image
        3. Apply confidence-based filtering to remove unreliable points
        4. Create Open3D point cloud with colors
        5. Position FiftyOne scene camera at VGGT camera location
        6. Save as both PCD and fo3d formats
        
        Args:
            vggt_output (Dict): Complete VGGT inference results including depth,
                camera parameters, and confidence maps
            output_path (Path): Destination path for fo3d file (PCD will use same base name)
        """
        try:
            # Convert depth map to 3D world coordinates using camera geometry
            world_points = unproject_depth_map_to_point_map(
                vggt_output["depth"],      # Depth values [H, W]
                vggt_output["extrinsic"],  # Camera extrinsic matrix [4, 4]
                vggt_output["intrinsic"]   # Camera intrinsic matrix [3, 3]
            )
            
            # Extract color information from preprocessed image
            # Convert from CHW (Channel-Height-Width) to HWC format
            colors = vggt_output["images"].transpose(1, 2, 0)  # Shape: [H, W, 3]
            confidence = vggt_output["depth_conf"]              # Shape: [H, W]
            
            # Reshape data for point cloud creation
            H, W = world_points.shape[:2]
            points = world_points.reshape(-1, 3)                        # [N, 3] world coordinates
            colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)  # [N, 3] RGB values [0, 255]
            conf_flat = confidence.reshape(-1)                          # [N] confidence values
            
            # Apply confidence-based filtering to remove unreliable points
            threshold_val = np.percentile(conf_flat, self.config.confidence_threshold)
            mask = (conf_flat >= threshold_val) & (conf_flat > 0.1)  # Additional minimum threshold
            
            # Create Open3D point cloud with filtered points and colors
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[mask].astype(np.float32))
            pcd.colors = o3d.utility.Vector3dVector(colors_flat[mask].astype(np.float32) / 255.0)
            
            # Generate PCD path using the same base name as fo3d
            pcd_path = output_path.with_suffix('.pcd')
            
            # Save PCD file directly
            o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False)
            logger.debug(f"Saved PCD file with {np.sum(mask):,} points to {pcd_path}")
            
            # Get camera setup using VGGT camera information
            camera_setup = self._determine_scene_camera_setup(vggt_output)
            
            # Convert to FiftyOne's fo3d format for 3D visualization
            scene = fo.Scene()
            
            # Set up camera with VGGT camera position and orientation
            if camera_setup["center"] is not None:
                scene.camera = fo.PerspectiveCamera(
                    center=camera_setup["center"],  # Position camera at VGGT camera location
                    up=camera_setup["up"]           # Set up direction based on camera pose
                )
                logger.debug(f"Positioned camera at {camera_setup['center']} with up={camera_setup['up']}")
            else:
                # Fallback to just up direction
                scene.camera = fo.PerspectiveCamera(up=camera_setup["up"])
                logger.debug(f"Set camera up direction to {camera_setup['up']}")
            
            # Add point cloud to scene
            mesh = fo.PointCloud("pointcloud", str(pcd_path.name))  # Use relative path
            scene.add(mesh)
            scene.write(str(output_path))
            
            logger.debug(f"Saved fo3d file with VGGT camera positioning to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving 3D files to {output_path}: {e}")


class VGGTOutputProcessor(fout.OutputProcessor):
    """Output processor for VGGT model predictions.
    
    Converts VGGT's rich 3D reconstruction outputs into FiftyOne-compatible labels.
    The primary output is a Heatmap label that references the saved depth PNG file,
    allowing users to visualize depth information directly in the FiftyOne App.
    
    While VGGT produces multiple types of valuable data (3D points, camera
    parameters), FiftyOne's apply_model() accepts only a single label_field. This
    processor focuses on the depth map as the primary visualization, with all other
    data saved as auxiliary files for post-processing.
    
    Design rationale:
    - Depth maps provide immediate visual value in FiftyOne's interface
    - Heatmap labels integrate seamlessly with FiftyOne's visualization tools  
    - Auxiliary files (fo3d, PCD) enable advanced 3D workflows
    - Co-location with original images simplifies data management
    
    Args:
        classes (None): Unused parameter for compatibility with base OutputProcessor
    """
    
    def __init__(self, classes=None, **kwargs):
        super().__init__(classes, **kwargs)
    
    def __call__(self, predictions):
        """Convert VGGT prediction into FiftyOne Heatmap label.
        
        Note: FiftyOne processes one prediction at a time, not batches.
        
        Args:
            predictions: Single prediction dict from VGGTModel.predict_all()
            frame_size: Image dimensions (width, height) - unused for VGGT
            confidence_thresh: Unused parameter for compatibility
                
        Returns:
            fol.Heatmap: Single heatmap label or None for failed predictions
        """
        # Handle failed prediction
        if predictions is None:
            return None
            
        try:
            return predictions['depth_map_path']
            
        except Exception as e:
            logger.error(f"Error creating heatmap label: {e}")
            return None