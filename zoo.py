
import logging
import numpy as np
import torch
import cv2
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
    """Configuration for running a :class:`VGGTModel`.

    See :class:`fiftyone.utils.torch.TorchImageModelConfig` for additional
    arguments.

    Args:
        model_name: the VGGT model name to load
        model_path: path to the saved model file on disk
        confidence_threshold: confidence threshold for point cloud filtering
        num_query_points: number of points to automatically select for tracking
    """

    def __init__(self, d):
        super().__init__(d)

        self.model_name = self.parse_string(d, "model_name", default="facebook/VGGT-1B")
        self.model_path = self.parse_string(d, "model_path")
        self.confidence_threshold = self.parse_number(d, "confidence_threshold", default=25.0)
        self.num_query_points = self.parse_int(d, "num_query_points", default=50)


class VGGTModel(fout.TorchImageModel):
    """Wrapper for VGGT models for 3D scene reconstruction.

    Args:
        config: a :class:`VGGTModelConfig`
    """

    def __init__(self, config):
        if not VGGT_AVAILABLE:
            raise ImportError("VGGT is not installed. Please install it first.")
            
        super().__init__(config)
        
        # Load the VGGT model
        self._vggt_model = self._load_vggt_model()
        
        # Set up dtype for mixed precision
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

    def _load_model(self, config):
        """Load the VGGT model from disk."""
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model file not found: {config.model_path}")
        
        logger.info(f"Loading VGGT model from {config.model_path}")
        
        # Load the saved model data
        checkpoint = torch.load(config.model_path, map_location='cpu')
        model_name = checkpoint['model_name']
        
        # Load model architecture
        model = VGGT.from_pretrained(model_name)
        
        # Load the saved state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model

    def _load_vggt_model(self):
        """Load and setup the VGGT model."""
        model = self._load_model(self.config)
        model = model.to(self._device)
        model.eval()
        return model

    def _generate_query_points(self, height: int, width: int, num_points: int) -> List[Tuple[float, float]]:
        """Generate query points for tracking in original image coordinates."""
        margin = 50
        h_points = int(np.sqrt(num_points * height / width))
        w_points = int(num_points / h_points)
        
        query_points = []
        for i in range(h_points):
            for j in range(w_points):
                y = margin + (height - 2 * margin) * (i + 0.5) / h_points
                x = margin + (width - 2 * margin) * (j + 0.5) / w_points
                
                # Add small random offset
                y += np.random.uniform(-20, 20)
                x += np.random.uniform(-20, 20)
                
                # Clamp to image bounds
                y = np.clip(y, margin, height - margin)
                x = np.clip(x, margin, width - margin)
                
                query_points.append((float(x), float(y)))
        
        return query_points[:num_points]

    def _predict_all(self, imgs):
        """Apply VGGT model to batch of images and create FiftyOne outputs."""
        
        # Handle different input types and get filepaths
        filepaths = []
        for img in imgs:
            if hasattr(img, 'filepath'):
                filepaths.append(img.filepath)
            elif isinstance(img, str):
                filepaths.append(img)
            else:
                # For tensor/array inputs, we'd need to save temporarily
                raise ValueError("VGGT FiftyOne integration requires image file paths")
        
        # Process each image individually 
        results = []
        fo3d_paths = []
        
        for i, filepath in enumerate(filepaths):
            try:
                logger.info(f"Processing image {i+1}/{len(filepaths)}: {filepath}")
                
                # Get original image size
                original_image = Image.open(filepath)
                original_size = original_image.size  # (width, height)
                
                # Load and preprocess for VGGT
                images = load_and_preprocess_images([filepath]).to(self._device)
                vggt_size = (images.shape[-1], images.shape[-2])  # (width, height)
                
                # Run VGGT inference
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=self.dtype):
                        predictions = self._vggt_model(images)
                        
                        # Convert pose encoding to camera matrices
                        extrinsic, intrinsic = pose_encoding_to_extri_intri(
                            predictions["pose_enc"], images.shape[-2:]
                        )
                        
                        predictions["extrinsic"] = extrinsic
                        predictions["intrinsic"] = intrinsic
                        predictions["images"] = images
                
                # Convert tensors to numpy and remove batch dimension
                for key, value in predictions.items():
                    if isinstance(value, torch.Tensor):
                        predictions[key] = value.cpu().float().numpy().squeeze(0)
                
                # Generate query points in original image coordinates
                orig_w, orig_h = original_size
                query_points = self._generate_query_points(orig_h, orig_w, self.config.num_query_points)
                
                # Get point tracks with coordinate conversion
                track_data = self._get_point_tracks(images, query_points, original_size, vggt_size)
                
                # Process outputs based on configuration
                result = {
                    "predictions": predictions,
                    "track_data": track_data,
                    "original_size": original_size,
                    "vggt_size": vggt_size,
                    "filepath": filepath,
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                # Add empty result to maintain batch consistency
                results.append({
                    "predictions": None,
                    "track_data": None,
                    "original_size": (224, 224),
                    "vggt_size": (224, 224),
                    "filepath": filepath,
                    "error": str(e)
                })
        
        # Process outputs with single unified processor
        if self._output_processor is not None:
            # Extract frame sizes for processor
            frame_sizes = [result["original_size"] for result in results]
            
            # Use output processor to get all outputs at once
            return self._output_processor(
                results, frame_sizes, confidence_thresh=self.config.confidence_threshold
            )
        
        # Fallback: return raw results if no processor
        return results

    def _get_point_tracks(self, images: torch.Tensor, query_points_original: List[Tuple[float, float]], 
                         original_size: Tuple[int, int], vggt_size: Tuple[int, int]) -> Dict:
        """Get 3D point tracks with coordinate conversion."""
        if not query_points_original:
            return {"tracks_2d": [], "tracks_3d": [], "query_points_original": [], 
                   "visibility_scores": [], "confidence_scores": []}
        
        # Convert query points from original to VGGT coordinates
        orig_w, orig_h = original_size
        vggt_w, vggt_h = vggt_size
        
        query_points_vggt = []
        for x_orig, y_orig in query_points_original:
            x_vggt = x_orig * vggt_w / orig_w
            y_vggt = y_orig * vggt_h / orig_h
            query_points_vggt.append((x_vggt, y_vggt))
        
        # Run VGGT tracking
        query_points_tensor = torch.FloatTensor(query_points_vggt).to(self._device)
        query_points_tensor = query_points_tensor[None]  # Add batch dimension
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                aggregated_tokens_list, ps_idx = self._vggt_model.aggregator(images[None])
                track_list, vis_score, conf_score = self._vggt_model.track_head(
                    aggregated_tokens_list, images[None], ps_idx, 
                    query_points=query_points_tensor
                )
        
        # Convert to numpy/lists
        def _convert_to_list(data):
            if isinstance(data, torch.Tensor):
                return data.cpu().float().numpy().squeeze(0).tolist()
            elif isinstance(data, list):
                return data
            elif isinstance(data, np.ndarray):
                return data.squeeze(0).tolist() if data.ndim > 1 else data.tolist()
            else:
                return data
        
        tracks_3d = _convert_to_list(track_list)
        
        return {
            "tracks_2d": query_points_original,
            "tracks_3d": tracks_3d,
            "query_points_original": query_points_original,
            "visibility_scores": _convert_to_list(vis_score),
            "confidence_scores": _convert_to_list(conf_score),
        }

    def _create_pointcloud_files(self, predictions: Dict, filepath: str) -> Optional[str]:
        """Create PCD and fo3d files for pointcloud output."""
        try:
            # Determine output directory
            if self.config.output_dir:
                output_dir = Path(self.config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = Path(filepath).parent
            
            # Generate output paths
            base_name = Path(filepath).stem
            pcd_path = output_dir / f"{base_name}.pcd"
            fo3d_path = output_dir / f"{base_name}.fo3d"
            
            # Create point cloud from depth
            world_points = unproject_depth_map_to_point_map(
                predictions["depth"],
                predictions["extrinsic"], 
                predictions["intrinsic"]
            )
            
            # Get colors from input image
            colors = predictions["images"].transpose(1, 2, 0)  # (H, W, 3)
            confidence = predictions["depth_conf"]  # (H, W)
            
            # Flatten for point cloud
            H, W = world_points.shape[:2]
            points = world_points.reshape(-1, 3)
            colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
            conf_flat = confidence.reshape(-1)
            
            # Apply confidence threshold
            threshold_val = np.percentile(conf_flat, self.config.confidence_threshold)
            mask = (conf_flat >= threshold_val) & (conf_flat > 0.1)
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[mask].astype(np.float32))
            pcd.colors = o3d.utility.Vector3dVector(colors_flat[mask].astype(np.float32) / 255.0)
            
            # Save PCD
            o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False)
            logger.info(f"Saved {np.sum(mask):,} points to {pcd_path}")
            
            # Convert to fo3d
            scene = fo.Scene()
            scene.camera = fo.PerspectiveCamera(up="Z")
            mesh = fo.PointCloud("pointcloud", str(pcd_path))
            scene.add(mesh)
            scene.write(str(fo3d_path))
            
            logger.info(f"Converted to fo3d: {fo3d_path}")
            return str(fo3d_path)
            
        except Exception as e:
            logger.error(f"Error creating pointcloud files for {filepath}: {e}")
            return None

    def _create_fiftyone_outputs(self, results: List[Dict]) -> Dict:
        """DEPRECATED: Outputs are now handled by VGGTOutputProcessor."""
        # This method is no longer used but kept for compatibility
        return {}

    def _create_grouped_samples(self, results: List[Dict]) -> List[fo.Sample]:
        """DEPRECATED: Grouped dataset creation is now handled by _convert_to_grouped_dataset function."""
        # This method is no longer used but kept for compatibility
        return []


class VGGTOutputProcessor(fout.OutputProcessor):
    """Unified output processor for all VGGT outputs: depth, points, and pointcloud."""
    
    def __init__(self, confidence_threshold=25.0, output_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir
        
    def __call__(self, results, frame_sizes, confidence_thresh=None):
        """Process all VGGT outputs and return labels for FiftyOne apply_model."""
        confidence_threshold = confidence_thresh or self.confidence_threshold
        
        # Initialize output containers
        depth_heatmaps = []
        point_keypoints = []
        fo3d_paths = []
        
        # Process each result
        for i, result in enumerate(results):
            if result["predictions"] is None:
                point_keypoints.append(None)
                fo3d_paths.append(None)
                continue
            
            # Process point tracks → keypoints
            try:
                point_keypoints_obj = self._create_point_keypoints(result)
                point_keypoints.append(point_keypoints_obj)
            except Exception as e:
                logger.error(f"Error creating keypoints for image {i}: {e}")
                point_keypoints.append(None)
            
            # Process point cloud → fo3d AND save depth map as PNG
            try:
                fo3d_path = self._create_pointcloud_files(result, confidence_threshold)
                fo3d_paths.append(fo3d_path)
            except Exception as e:
                logger.error(f"Error creating point cloud for image {i}: {e}")
                fo3d_paths.append(None)
        
        # Create combined predictions for each sample
        combined_predictions = []
        for i, (keypoints, fo3d_path, result) in enumerate(
            zip(point_keypoints, fo3d_paths, results)
        ):
            # Create a combined prediction object for this sample
            prediction = fo.DynamicEmbeddedDocument()
            
            if keypoints is not None:
                prediction.keypoints = keypoints
            if fo3d_path is not None:
                prediction.fo3d_path = fo3d_path
                
            # Add metadata
            if result["predictions"] is not None:
                prediction.metadata = fo.DynamicEmbeddedDocument(
                    confidence_threshold=confidence_threshold,
                    original_size=result["original_size"],
                    vggt_size=result["vggt_size"],
                    num_query_points=len(result["track_data"]["tracks_2d"]) if result["track_data"] else 0,
                )
            
            combined_predictions.append(prediction)
        
        return combined_predictions
    
    def _create_point_keypoints(self, result: Dict) -> Optional[fol.Keypoints]:
        """Create keypoints from point tracks."""
        if result["track_data"] is None:
            return None
            
        tracks_2d = result["track_data"]["tracks_2d"]
        confidence_scores = result["track_data"]["confidence_scores"]
        orig_w, orig_h = result["original_size"]
        
        keypoints = []
        for j, (x, y) in enumerate(tracks_2d):
            # Normalize coordinates to [0, 1]
            x_norm = x / orig_w
            y_norm = y / orig_h
            
            # Get confidence if available
            conf = confidence_scores[j] if j < len(confidence_scores) else 1.0
            
            keypoint = fol.Keypoint(
                points=[(x_norm, y_norm)],
                confidence=[conf],
                label=f"track_{j}"
            )
            keypoints.append(keypoint)
        
        return fol.Keypoints(keypoints=keypoints)
    
    def _create_pointcloud_files(self, result: Dict, confidence_threshold: float) -> Optional[str]:
        """Create PCD, fo3d files, and save depth map as PNG."""
        filepath = result["filepath"]
        if not filepath:
            return None
            
        # Output directory is same as input image directory
        output_dir = Path(filepath).parent
        
        # Generate output paths
        base_name = Path(filepath).stem
        pcd_path = output_dir / f"{base_name}.pcd"
        fo3d_path = output_dir / f"{base_name}.fo3d"
        depth_png_path = output_dir / f"{base_name}_depth.png"
        
        predictions = result["predictions"]
        
        # Save depth map as PNG first
        self._save_depth_map_png(predictions["depth"], depth_png_path, result["original_size"])
        
        # Create point cloud from depth
        world_points = unproject_depth_map_to_point_map(
            predictions["depth"],
            predictions["extrinsic"], 
            predictions["intrinsic"]
        )
        
        # Get colors from input image
        colors = predictions["images"].transpose(1, 2, 0)  # (H, W, 3)
        confidence = predictions["depth_conf"]  # (H, W)
        
        # Flatten for point cloud
        H, W = world_points.shape[:2]
        points = world_points.reshape(-1, 3)
        colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
        conf_flat = confidence.reshape(-1)
        
        # Apply confidence threshold
        threshold_val = np.percentile(conf_flat, confidence_threshold)
        mask = (conf_flat >= threshold_val) & (conf_flat > 0.1)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[mask].astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(colors_flat[mask].astype(np.float32) / 255.0)
        
        # Save PCD
        o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False)
        logger.info(f"Saved {np.sum(mask):,} points to {pcd_path}")
        
        # Convert to fo3d
        try:
            scene = fo.Scene()
            scene.camera = fo.PerspectiveCamera(up="Z")
            mesh = fo.PointCloud("pointcloud", str(pcd_path))
            scene.add(mesh)
            scene.write(str(fo3d_path))
            
            logger.info(f"Converted to fo3d: {fo3d_path}")
            return str(fo3d_path)
            
        except Exception as e:
            logger.error(f"Error converting PCD to fo3d: {e}")
            return None
    
    def _save_depth_map_png(self, depth: np.ndarray, output_path: Path, original_size: Tuple[int, int]):
        """Save depth map as PNG image, resized to original image dimensions."""
        # Remove extra dimensions
        depth_vis = depth.squeeze()
        
        # Resize depth map to original image size
        orig_w, orig_h = original_size
        depth_resized = resize(
            depth_vis,
            (orig_h, orig_w),
            preserve_range=True,
            anti_aliasing=True
        )
        
        # Handle invalid/infinite depths
        valid_mask = np.isfinite(depth_resized) & (depth_resized > 0)
        if np.any(valid_mask):
            depth_min = np.percentile(depth_resized[valid_mask], 5)
            depth_max = np.percentile(depth_resized[valid_mask], 95)
            
            # Normalize to 0-255 range
            if depth_max > depth_min:
                depth_normalized = np.clip((depth_resized - depth_min) / (depth_max - depth_min), 0, 1)
            else:
                depth_normalized = np.zeros_like(depth_resized)
        else:
            depth_normalized = np.zeros_like(depth_resized)
        
        depth_normalized[~valid_mask] = 0
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        
        # Save as PNG
        cv2.imwrite(str(output_path), depth_colored)
        logger.info(f"Saved depth map to {output_path}")
