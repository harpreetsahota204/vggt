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
            fewer but more reliable points. Default is 25.0.
        num_query_points (int): Number of 2D points to automatically generate
            for 3D tracking. These points are distributed in a grid pattern
            across the image. Default is 50.
    """

    def __init__(self, d):
        super().__init__(d)
        self.model_name = self.parse_string(d, "model_name", default="facebook/VGGT-1B")
        self.model_path = self.parse_string(d, "model_path")
        self.confidence_threshold = self.parse_number(d, "confidence_threshold", default=51.0)
        self.num_query_points = self.parse_int(d, "num_query_points", default=1551)


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
        
        # Determine optimal precision based on GPU capabilities
        # Ampere GPUs (RTX 30xx+) support bfloat16 for better numerical stability
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

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

        return model

    def predict_all(self, imgs, samples=None):
        """Perform VGGT inference on a batch of images and save outputs.
        
        This is the main inference method that:
        1. Preprocesses each image for VGGT (518px, proper aspect ratio)
        2. Runs VGGT inference to get depth, pose, and confidence data
        3. Generates automatic query points for 3D tracking
        4. Saves auxiliary files (depth PNG, fo3d, keypoints JSON) alongside originals
        5. Returns prediction data for the output processor
        
        All auxiliary files are saved in the same directory as the original image
        with descriptive suffixes (_depth.png, .fo3d, _keypoints.json).

        Args:
            imgs (List[PIL.Image]): Batch of input images from FiftyOne
            samples (List[fiftyone.core.sample.Sample]): Corresponding sample objects
                containing filepath and other metadata. Required for file path access.
                
        Returns:
            List[Dict]: Prediction data for each image, containing:
                - depth_map_path: Path to saved depth PNG for heatmap display
                - vggt_output: Complete VGGT inference results
                None entries indicate failed predictions.
        """
        if samples is None:
            raise ValueError("VGGT model requires sample objects to access filepaths")
        
        predictions = []
        
        for img, sample in zip(imgs, samples):
            try:
                # Extract file path information for saving auxiliary outputs
                original_path = Path(sample.filepath)
                base_dir = original_path.parent
                base_name = original_path.stem
                
                # Define output paths for all auxiliary files
                # These will be co-located with the original image
                depth_png_path = base_dir / f"{base_name}_depth.png"      # Colorized depth for heatmap
                fo3d_path = base_dir / f"{base_name}.fo3d"                # 3D point cloud
                keypoints_path = base_dir / f"{base_name}_keypoints.json" # Tracking data
                
                # Preprocess image using VGGT-specific requirements
                img_tensor, original_size = self._preprocess_vggt_image(img)
                vggt_size = (img_tensor.shape[-1], img_tensor.shape[-2])  # (width, height)
                
                # Add batch dimension for model inference
                img_batch = img_tensor.unsqueeze(0)
                
                # Run VGGT inference with automatic mixed precision
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=self.dtype):
                        # Forward pass through VGGT model
                        vggt_predictions = self._model(img_batch)
                        
                        # Convert VGGT's pose encoding to standard camera matrices
                        extrinsic, intrinsic = pose_encoding_to_extri_intri(
                            vggt_predictions["pose_enc"], img_batch.shape[-2:]
                        )
                
                # Generate query points for 3D tracking in original image coordinates
                query_points_original = self._generate_query_points(
                    original_size[1], original_size[0], self.config.num_query_points
                )
                
                # Perform 3D point tracking with coordinate space conversion
                track_data = self._get_point_tracks(
                    img_batch, query_points_original, original_size, vggt_size
                )
                
                # Package all VGGT outputs for file saving and processing
                vggt_output = {
                    "depth": vggt_predictions["depth"].cpu().numpy().squeeze(0),
                    "depth_conf": vggt_predictions["depth_conf"].cpu().numpy().squeeze(0),
                    "world_points_conf": vggt_predictions["world_points_conf"].cpu().numpy().squeeze(0),
                    "extrinsic": extrinsic.cpu().numpy().squeeze(0),
                    "intrinsic": intrinsic.cpu().numpy().squeeze(0),
                    "images": img_tensor.cpu().numpy(),
                    "point_tracks": track_data,
                    "original_size": original_size,
                    "vggt_size": vggt_size,
                }
                
                # Save all auxiliary files for post-processing
                self._save_depth_png(vggt_output, depth_png_path)
                self._save_fo3d(vggt_output, fo3d_path)
                self._save_keypoints(vggt_output, keypoints_path)
                
                # Package prediction data for output processor
                prediction_data = {
                    'depth_map_path': str(depth_png_path),  # Primary output for heatmap
                    'vggt_output': vggt_output,             # Complete results for debugging
                }
                
                predictions.append(prediction_data)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample.filepath}: {e}")
                # Return None for failed predictions to maintain batch consistency
                predictions.append(None)
        
        return predictions

def _preprocess_vggt_image(self, img):
    """Preprocess PIL image for VGGT model using VGGT's preprocessing logic."""
    # Ensure we have a PIL Image, not a tensor
    if isinstance(img, torch.Tensor):
        raise TypeError("Expected PIL Image, got torch.Tensor. Make sure the input to _preprocess_vggt_image is a PIL Image.")
    
    # Store original size
    original_size = img.size  # (width, height)
    
    target_size = 518
    
    # Handle RGBA images by blending onto white background
    if img.mode == "RGBA":
        img = Image.alpha_composite(
            img.convert("RGBA"), 
            Image.new("RGBA", img.size, (255, 255, 255, 255))
        ).convert("RGB")
    elif img.mode != "RGB":
        # Convert to RGB if not already
        img = img.convert("RGB")
    
    width, height = img.size
    
    # Set width to 518px
    new_width = target_size
    # Calculate height maintaining aspect ratio, divisible by 14
    new_height = round(height * (new_width / width) / 14) * 14
    
    # Resize with new dimensions
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img_tensor = TF.to_tensor(img)  # Convert to tensor (0, 1), shape [C, H, W]
    
    # Center crop height if it's larger than 518
    if new_height > target_size:
        start_y = (new_height - target_size) // 2
        img_tensor = img_tensor[:, start_y : start_y + target_size, :]
    
    # Move to device
    img_tensor = img_tensor.to(self._device)
    
    logger.debug(f"Preprocessed image shape: {img_tensor.shape}, original size: {original_size}")
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

    def _get_point_tracks(self, images: torch.Tensor, query_points_original: List[Tuple[float, float]], 
                         original_size: Tuple[int, int], vggt_size: Tuple[int, int]) -> Dict:
        """Perform 3D point tracking with coordinate space conversion.
        
        This method handles the complex coordinate transformations required for VGGT
        tracking. Query points are specified in original image coordinates but must
        be converted to VGGT's preprocessed coordinate space for inference.
        
        The tracking process:
        1. Convert 2D query points from original to VGGT coordinate space
        2. Run VGGT's tracking pipeline (aggregator + track_head)
        3. Return 2D tracks in original coordinates and 3D tracks in world space
        
        Args:
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
        
        # Run VGGT's tracking pipeline with mixed precision
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                # Extract features using VGGT's aggregator module
                aggregated_tokens_list, ps_idx = self._model.aggregator(images)
                
                # Perform tracking using VGGT's track head
                track_list, vis_score, conf_score = self._model.track_head(
                    aggregated_tokens_list, images, ps_idx, 
                    query_points=query_points_tensor
                )
        
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
        """Save depth map as a colorized PNG for FiftyOne heatmap visualization.
        
        Converts VGGT's raw depth predictions into a visually interpretable
        colorized image that can be displayed as a heatmap in FiftyOne. The depth
        values are normalized and colored using OpenCV's JET colormap.
        
        Processing steps:
        1. Resize depth map to match original image dimensions
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
            
            # Resize depth map to match original image dimensions
            # This ensures the heatmap aligns perfectly with the original image
            orig_w, orig_h = vggt_output["original_size"]
            depth_resized = resize(
                depth_vis,
                (orig_h, orig_w),                    # Target size (height, width)
                preserve_range=True,                 # Keep original value range
                anti_aliasing=True                   # Smooth resizing
            )
            
            # Handle invalid depth values (NaN, infinity, negative)
            valid_mask = np.isfinite(depth_resized) & (depth_resized > 0)
            
            if np.any(valid_mask):
                # Use robust percentile-based normalization to handle outliers
                depth_min = np.percentile(depth_resized[valid_mask], 5)   # Ignore bottom 5%
                depth_max = np.percentile(depth_resized[valid_mask], 95)  # Ignore top 5%
                
                # Normalize to [0, 1] range
                if depth_max > depth_min:
                    depth_normalized = np.clip(
                        (depth_resized - depth_min) / (depth_max - depth_min), 0, 1
                    )
                else:
                    # Handle edge case where all valid depths are the same
                    depth_normalized = np.zeros_like(depth_resized)
            else:
                # No valid depth values found
                depth_normalized = np.zeros_like(depth_resized)
            
            # Set invalid regions to zero (will appear as dark blue in JET colormap)
            depth_normalized[~valid_mask] = 0
            
            # Convert to 8-bit integer for image saving
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            
            # Apply JET colormap: blue (near) -> green -> yellow -> red (far)
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
            
            # Save as PNG (lossless compression)
            cv2.imwrite(str(output_path), depth_colored)
            logger.debug(f"Saved depth map to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving depth PNG to {output_path}: {e}")

    def _save_fo3d(self, vggt_output: Dict, output_path: Path):
        """Save 3D point cloud in FiftyOne's fo3d format for 3D visualization.
        
        Converts VGGT's depth map and camera parameters into a colored 3D point
        cloud suitable for visualization in FiftyOne's 3D viewer. Points are
        filtered by confidence to remove noisy regions.
        
        Processing pipeline:
        1. Unproject depth map to 3D world coordinates using camera parameters
        2. Extract corresponding colors from the input image
        3. Apply confidence-based filtering to remove unreliable points
        4. Create Open3D point cloud with colors
        5. Convert to FiftyOne's fo3d format
        
        Args:
            vggt_output (Dict): Complete VGGT inference results including depth,
                camera parameters, and confidence maps
            output_path (Path): Destination path for fo3d file
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
            
            # Save temporary PCD file (required by FiftyOne's fo3d converter)
            temp_pcd_path = output_path.with_suffix('.pcd')
            o3d.io.write_point_cloud(str(temp_pcd_path), pcd, write_ascii=False)
            
            # Convert to FiftyOne's fo3d format for 3D visualization
            scene = fo.Scene()
            scene.camera = fo.PerspectiveCamera(up="Z")  # Set camera orientation
            mesh = fo.PointCloud("pointcloud", str(temp_pcd_path))
            scene.add(mesh)
            scene.write(str(output_path))
            
            # Clean up temporary PCD file
            temp_pcd_path.unlink()
            
            logger.debug(f"Saved fo3d file with {np.sum(mask):,} points to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving fo3d file to {output_path}: {e}")

    def _save_keypoints(self, vggt_output: Dict, output_path: Path):
        """Save complete tracking and 3D reconstruction data as JSON.
        
        Serializes all the tracking results, 3D coordinates, and camera parameters
        to a JSON file for post-processing. This includes both pixel and normalized
        coordinates for maximum flexibility in downstream applications.
        
        Saved data includes:
        - 2D tracking points in both pixel and normalized [0,1] coordinates
        - 3D world coordinates for each tracked point
        - Visibility and confidence scores per point
        - Camera intrinsic and extrinsic parameters
        - Image size information for coordinate conversion
        
        Args:
            vggt_output (Dict): Complete VGGT inference results
            output_path (Path): Destination path for JSON file
        """
        try:
            # Extract tracking data from VGGT output
            track_data = vggt_output["point_tracks"]
            
            # Convert pixel coordinates to normalized [0, 1] coordinates
            # This format is required by many FiftyOne label types
            orig_w, orig_h = vggt_output["original_size"]
            normalized_points = []
            for x, y in track_data["tracks_2d"]:
                normalized_points.append([x / orig_w, y / orig_h])
            
            # Package all tracking and reconstruction data
            keypoint_data = {
                # 2D tracking points in different coordinate systems
                "points_2d_normalized": normalized_points,           # [0, 1] range for FiftyOne
                "points_2d_pixel": track_data["tracks_2d"],         # Pixel coordinates
                
                # 3D tracking and confidence data
                "tracks_3d": track_data["tracks_3d"],               # World coordinates
                "visibility_scores": track_data["visibility_scores"], # Per-point visibility
                "confidence_scores": track_data["confidence_scores"], # Per-point confidence
                
                # Image and preprocessing metadata
                "original_size": vggt_output["original_size"],       # Original image dimensions
                "vggt_size": vggt_output["vggt_size"],              # Preprocessed dimensions
                
                # Camera parameters for 3D reconstruction
                "camera_extrinsic": vggt_output["extrinsic"].tolist(),  # 4x4 pose matrix
                "camera_intrinsic": vggt_output["intrinsic"].tolist(),  # 3x3 calibration matrix
            }
            
            # Save as formatted JSON for human readability
            with open(output_path, 'w') as f:
                json.dump(keypoint_data, f, indent=2)
            
            logger.debug(f"Saved {len(normalized_points)} keypoints to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving keypoints to {output_path}: {e}")


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
        """Convert VGGT predictions into FiftyOne Heatmap labels.
        
        Processes the prediction data from VGGTModel.predict_all() and creates
        Heatmap labels that reference the saved depth PNG files. Each heatmap
        provides a direct visual representation of the scene's depth structure.
        
        The depth maps are automatically displayed as color-coded overlays in
        FiftyOne's image viewer, where:
        - Blue regions represent near objects (small depth values)
        - Red regions represent far objects (large depth values)
        - The color scale is automatically normalized per image
        
        Args:
            predictions (List[Dict]): Prediction data from VGGTModel.predict_all(),
                where each dict contains:
                - depth_map_path: Path to the saved depth PNG file
                - vggt_output: Complete VGGT results (for debugging)
                None entries indicate failed predictions.
            frame_size (Tuple[int, int]): Image dimensions (width, height).
                Unused for VGGT since depth maps are pre-sized to match originals.
            confidence_thresh (float, optional): Unused parameter for compatibility
                with other model types that support confidence filtering.
                
        Returns:
            List[fol.Heatmap]: FiftyOne Heatmap labels, one per input image.
                Each heatmap references a depth PNG file via map_path.
                None entries are returned for failed predictions.
        """
        heatmaps = []
        
        for pred_data in predictions:
            # Handle failed predictions gracefully
            if pred_data is None:
                heatmaps.append(None)
                continue
                
            try:
                # Create FiftyOne Heatmap label pointing to depth PNG
                heatmap = fol.Heatmap(
                    label="depth_map",                    # Descriptive label for UI
                    map_path=pred_data['depth_map_path']  # Path to colorized depth PNG
                )
                heatmaps.append(heatmap)
                
            except Exception as e:
                logger.error(f"Error creating heatmap label: {e}")
                heatmaps.append(None)
        
        return heatmaps