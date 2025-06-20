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


class VGGTModel(fout.TorchImageModel, fout.TorchSamplesMixin):
    """VGGT model wrapper for 3D scene reconstruction from single images.
    
    This class implements a FiftyOne-compatible wrapper around VGGT (Video-Grounded
    Generation and Tracking) models. VGGT takes a single RGB image as input and
    produces:
    
    1. Dense depth maps with confidence scores
    2. 3D point tracking for automatically generated query points
    3. Camera pose estimation (extrinsic and intrinsic parameters)
    4. Dense 3D point clouds from depth map unprojection
    
    The model outputs are saved as:
    - Depth map PNG: Colorized depth visualization for FiftyOne heatmap display
    - fo3d file: 3D point cloud for FiftyOne 3D visualization
    - Keypoints JSON: Complete tracking data with 2D/3D correspondences
    
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
            keypoints_path = base_dir / f"{base_name}_keypoints.json" # Tracking data
            
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
            # Note: Removed keypoints saving since point tracking is disabled
            
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
        images = load_and_preprocess_images([original_path], mode="pad").to(self._device)
        
        # Remove batch dimension since we're processing single images
        img_tensor = images.squeeze(0)  # Remove batch dim: [1, C, H, W] -> [C, H, W]
        
        vggt_size = (img_tensor.shape[-1], img_tensor.shape[-2])  # (width, height)
        
        return img_tensor, original_size

    def _generate_query_points(self, height: int, width: int, num_points: int) -> List[Tuple[float, float]]:
        """Generate a grid of query points for 3D tracking in original image coordinates.
        
        Creates a roughly uniform distribution of points across the image for tracking,
        avoiding edges where tracking might be unreliable. Points are arranged in a
        grid pattern with small random offsets for natural distribution.
        
        Args:
            height (int): Image height in pixels
            width (int): Image width in pixels  
            num_points (int): Target number of points to generate
            
        Returns:
            List[Tuple[float, float]]: Query points as (x, y) coordinates in pixels
        """
        # Define safety margin to avoid edge artifacts
        margin = 50  # pixels from image edges
        
        # Calculate grid dimensions to approximate target point count
        # Maintain aspect ratio in point distribution
        h_points = int(np.sqrt(num_points * height / width))
        w_points = int(num_points / h_points)
        
        query_points = []
        
        # Generate grid points with random jitter
        for i in range(h_points):
            for j in range(w_points):
                # Calculate base grid position (center of grid cell)
                y = margin + (height - 2 * margin) * (i + 0.5) / h_points
                x = margin + (width - 2 * margin) * (j + 0.5) / w_points
                
                # Add small random offset to avoid perfect grid alignment
                y += np.random.uniform(-20, 20)
                x += np.random.uniform(-20, 20)
                
                # Ensure points stay within image bounds (with margin)
                y = np.clip(y, margin, height - margin)
                x = np.clip(x, margin, width - margin)
                
                query_points.append((float(x), float(y)))
        
        # Return exact number of requested points
        return query_points[:num_points]

    def _get_point_tracks(self, aggregated_tokens_list, ps_idx, images: torch.Tensor, 
                         query_points_original: List[Tuple[float, float]], 
                         original_size: Tuple[int, int], vggt_size: Tuple[int, int]) -> Dict:
        """Perform 3D point tracking with coordinate space conversion.
        
        This method handles the complex coordinate transformations required for VGGT
        tracking. Query points are specified in original image coordinates but must
        be converted to VGGT's preprocessed coordinate space for inference.
        
        The tracking process:
        1. Convert 2D query points from original to VGGT coordinate space
        2. Run VGGT's tracking pipeline using pre-computed aggregated features
        3. Return 2D tracks in original coordinates and 3D tracks in world space
        
        Args:
            aggregated_tokens_list: Pre-computed aggregated features from VGGT aggregator
            ps_idx: Patch indices from VGGT aggregator
            images (torch.Tensor): Preprocessed image batch [N, C, H, W] 
            query_points_original (List[Tuple[float, float]]): Query points in original
                image pixel coordinates
            original_size (Tuple[int, int]): Original image (width, height)
            vggt_size (Tuple[int, int]): VGGT preprocessed image (width, height)
            
        Returns:
            Dict: Tracking results containing:
                - tracks_2d: 2D points in original coordinates
                - tracks_3d: 3D points in world coordinates  
                - visibility_scores: Per-point visibility confidence
                - confidence_scores: Per-point tracking confidence
        """
        # Handle empty query points case
        if not query_points_original:
            return {"tracks_2d": [], "tracks_3d": [], "query_points_original": [], 
                   "visibility_scores": [], "confidence_scores": []}
        
        # Ensure all tensors are on the same device to avoid CUDA errors
        images = images.to(self._device)
        
        # Convert query points from original image space to VGGT preprocessing space
        orig_w, orig_h = original_size
        vggt_w, vggt_h = vggt_size
        
        query_points_vggt = []
        for x_orig, y_orig in query_points_original:
            # Apply the same scaling transformation used in preprocessing
            x_vggt = x_orig * vggt_w / orig_w
            y_vggt = y_orig * vggt_h / orig_h
            query_points_vggt.append((x_vggt, y_vggt))
        
        # Convert query points to tensor format for VGGT
        query_points_tensor = torch.FloatTensor(query_points_vggt).to(self._device)
        query_points_tensor = query_points_tensor[None]  # Add batch dimension [1, N, 2]
        
        # Run VGGT's tracking pipeline using pre-computed features
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                # Debug: Check what track_head actually returns
                print("DEBUG: Calling track_head...")
                tracking_output = self._model.track_head(
                    aggregated_tokens_list, images, ps_idx, 
                    query_points=query_points_tensor
                )
                
                # Print debug information
                print(f"DEBUG: track_head returned {len(tracking_output)} values")
                print(f"DEBUG: Types: {[type(x) for x in tracking_output]}")
                
                # Handle different return value counts
                if len(tracking_output) == 3:
                    track_list, vis_score, conf_score = tracking_output
                    print("DEBUG: Successfully unpacked 3 values from track_head")
                elif len(tracking_output) == 4:
                    track_list, vis_score, conf_score, extra = tracking_output
                    print(f"DEBUG: Got 4 values, extra type: {type(extra)}")
                elif len(tracking_output) == 5:
                    track_list, vis_score, conf_score, extra1, extra2 = tracking_output
                    print(f"DEBUG: Got 5 values, extra types: {type(extra1)}, {type(extra2)}")
                else:
                    raise ValueError(f"Unexpected number of return values from track_head: {len(tracking_output)}")
        
        
        def _convert_to_list(data):
            """Convert various tensor formats to Python lists for serialization."""
            if isinstance(data, torch.Tensor):
                # Handle mixed precision by converting to float32 first
                return data.cpu().float().numpy().squeeze(0).tolist()
            elif isinstance(data, list):
                return data
            elif isinstance(data, np.ndarray):
                return data.squeeze(0).tolist() if data.ndim > 1 else data.tolist()
            else:
                return data
        
        # Convert tracking outputs to serializable format
        tracks_3d = _convert_to_list(track_list)
        
        return {
            "tracks_2d": query_points_original,                    # 2D in original coordinates
            "tracks_3d": tracks_3d,                               # 3D in world coordinates
            "query_points_original": query_points_original,       # For reference
            "query_points_vggt": query_points_vggt,              # For debugging
            "visibility_scores": _convert_to_list(vis_score),     # Per-point visibility
            "confidence_scores": _convert_to_list(conf_score),    # Per-point confidence
        }

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

    def _save_fo3d(self, vggt_output: Dict, output_path: Path):
        """Save 3D point cloud in both PCD and FiftyOne's fo3d formats.
        
        Converts VGGT's depth map and camera parameters into a colored 3D point
        cloud suitable for visualization in FiftyOne's 3D viewer. Points are
        filtered by confidence to remove noisy regions.
        
        Processing pipeline:
        1. Unproject depth map to 3D world coordinates using camera parameters
        2. Extract corresponding colors from the input image
        3. Apply confidence-based filtering to remove unreliable points
        4. Create Open3D point cloud with colors
        5. Save as both PCD and fo3d formats
        
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
            
            # Convert to FiftyOne's fo3d format for 3D visualization
            scene = fo.Scene()
            scene.camera = fo.PerspectiveCamera(up="Z")  # Set camera orientation
            mesh = fo.PointCloud("pointcloud", str(pcd_path))  # Reference the saved PCD
            scene.add(mesh)
            scene.write(str(output_path))
            
            logger.debug(f"Saved fo3d file to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving 3D files to {output_path}: {e}")


class VGGTOutputProcessor(fout.OutputProcessor):
    """Output processor for VGGT model predictions.
    
    Converts VGGT's rich 3D reconstruction outputs into FiftyOne-compatible labels.
    The primary output is a Heatmap label that references the saved depth PNG file,
    allowing users to visualize depth information directly in the FiftyOne App.
    
    While VGGT produces multiple types of valuable data (3D points, tracking, camera
    parameters), FiftyOne's apply_model() accepts only a single label_field. This
    processor focuses on the depth map as the primary visualization, with all other
    data saved as auxiliary files for post-processing.
    
    Design rationale:
    - Depth maps provide immediate visual value in FiftyOne's interface
    - Heatmap labels integrate seamlessly with FiftyOne's visualization tools  
    - Auxiliary files (fo3d, keypoints) enable advanced 3D workflows
    - Co-location with original images simplifies data management
    
    Args:
        classes (None): Unused parameter for compatibility with base OutputProcessor
    """
    
    def __init__(self, classes=None, **kwargs):
        super().__init__(classes, **kwargs)
    
    def __call__(self, predictions, frame_size, confidence_thresh=None):
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
            # Create FiftyOne Heatmap label pointing to depth PNG
            heatmap = fol.Heatmap(
                label="depth_map",                      # Descriptive label for UI
                map_path=predictions['depth_map_path']  # Path to colorized depth PNG
            )
            return heatmap
            
        except Exception as e:
            logger.error(f"Error creating heatmap label: {e}")
            return None