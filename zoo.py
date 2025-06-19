import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from skimage.transform import resize
import torch
import torchvision
import torchvision.transforms.functional as TF

# FiftyOne imports
import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

logger = logging.getLogger(__name__)

class VGGTModelConfig(fout.TorchImageModelConfig):
    """Configuration for running a :class:`VGGTModel`.

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
        self.confidence_threshold = self.parse_number(d, "confidence_threshold", default=51.0)
        self.num_query_points = self.parse_int(d, "num_query_points", default=5000)


class VGGTModel(fout.TorchImageModel, fout.TorchSamplesMixin):
    """Wrapper for VGGT models for 3D scene reconstruction.

    Args:
        config: a :class:`VGGTModelConfig`
    """

    def __init__(self, config):
        super().__init__(config)
        
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
        model = torch.load(config.model_path, map_location='cpu', weights_only=False)

        return model

    def _build_transforms(self, config):
        """Build VGGT-specific transforms."""
        # VGGT has its own preprocessing, so we'll handle it in predict_all
        # Return minimal transforms for now
        transforms = [fout.ToPILImage()]
        return torchvision.transforms.Compose(transforms), True  # ragged_batches=True

    def predict_all(self, imgs, samples=None):
        """Performs prediction on the given batch of images.

        Args:
            imgs: the batch of images to process (list of PIL images)
            samples: the corresponding sample objects
            
        Returns:
            a list of prediction data for the output processor
        """
        if samples is None:
            raise ValueError("VGGT model requires sample objects to access filepaths")
        
        predictions = []
        
        for img, sample in zip(imgs, samples):
            try:
                # Get original file info
                original_path = Path(sample.filepath)
                base_dir = original_path.parent
                base_name = original_path.stem
                
                # Generate output paths
                depth_png_path = base_dir / f"{base_name}_depth.png"
                fo3d_path = base_dir / f"{base_name}.fo3d"
                keypoints_path = base_dir / f"{base_name}_keypoints.json"
                
                # Preprocess image for VGGT
                img_tensor, original_size = self._preprocess_vggt_image(img)
                vggt_size = (img_tensor.shape[-1], img_tensor.shape[-2])  # (width, height)
                
                # Add batch dimension
                img_batch = img_tensor.unsqueeze(0)
                
                # Run VGGT inference
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=self.dtype):
                        # Run full VGGT pipeline
                        vggt_predictions = self._model(img_batch)
                        
                        # Convert pose encoding to camera matrices
                        extrinsic, intrinsic = pose_encoding_to_extri_intri(
                            vggt_predictions["pose_enc"], img_batch.shape[-2:]
                        )
                
                # Generate automatic query points for tracking in original coordinates
                query_points_original = self._generate_query_points(
                    original_size[1], original_size[0], self.config.num_query_points
                )
                
                # Get point tracks
                track_data = self._get_point_tracks(
                    img_batch, query_points_original, original_size, vggt_size
                )
                
                # Prepare VGGT output data
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
                
                # Save auxiliary files
                self._save_depth_png(vggt_output, depth_png_path)
                self._save_fo3d(vggt_output, fo3d_path)
                self._save_keypoints(vggt_output, keypoints_path)
                
                # Package for output processor
                prediction_data = {
                    'depth_map_path': str(depth_png_path),
                    'vggt_output': vggt_output,
                }
                
                predictions.append(prediction_data)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample.filepath}: {e}")
                # Return None for failed predictions
                predictions.append(None)
        
        return predictions

    def _preprocess_vggt_image(self, img):
        """Preprocess PIL image for VGGT model using VGGT's preprocessing logic.
        
        Args:
            img: PIL Image from FiftyOne
            
        Returns:
            Preprocessed torch tensor [C, H, W] and original size
        """
        # Store original size
        original_size = img.size  # (width, height)
        
        target_size = 518
        
        # Handle RGBA images by blending onto white background
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            img = Image.alpha_composite(img.convert("RGBA"), 
                                      Image.new("RGBA", img.size, (255, 255, 255, 255)))
        
        # Convert to RGB
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
        """Generate query points in original image coordinates."""
        # Create a grid of points, avoiding edges
        margin = 50  # pixels from edge
        h_points = int(np.sqrt(num_points * height / width))
        w_points = int(num_points / h_points)
        
        query_points = []
        for i in range(h_points):
            for j in range(w_points):
                # Grid position with some random offset
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

    def _get_point_tracks(self, images: torch.Tensor, query_points_original: List[Tuple[float, float]], 
                         original_size: Tuple[int, int], vggt_size: Tuple[int, int]) -> Dict:
        """Get 3D point tracks, handling coordinate conversion between original and VGGT resolution."""
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
        
        # Run VGGT tracking with scaled coordinates
        query_points_tensor = torch.FloatTensor(query_points_vggt).to(self._device)
        query_points_tensor = query_points_tensor[None]  # Add batch dimension
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                aggregated_tokens_list, ps_idx = self._model.aggregator(images)
                track_list, vis_score, conf_score = self._model.track_head(
                    aggregated_tokens_list, images, ps_idx, 
                    query_points=query_points_tensor
                )
        
        # Handle different return types from track_head
        def _convert_to_list(data):
            if isinstance(data, torch.Tensor):
                # Convert to float32 first to handle bfloat16
                return data.cpu().float().numpy().squeeze(0).tolist()
            elif isinstance(data, list):
                return data
            elif isinstance(data, np.ndarray):
                return data.squeeze(0).tolist() if data.ndim > 1 else data.tolist()
            else:
                return data
        
        tracks_3d = _convert_to_list(track_list)
        
        return {
            "tracks_2d": query_points_original,  # 2D tracks in original coordinates
            "tracks_3d": tracks_3d,              # 3D tracks from VGGT
            "query_points_original": query_points_original,
            "query_points_vggt": query_points_vggt,
            "visibility_scores": _convert_to_list(vis_score),
            "confidence_scores": _convert_to_list(conf_score),
        }

    def _save_depth_png(self, vggt_output: Dict, output_path: Path):
        """Save depth map as PNG image, resized to original image dimensions."""
        try:
            # Remove extra dimensions
            depth_vis = vggt_output["depth"].squeeze()
            
            # Resize depth map to original image size
            orig_w, orig_h = vggt_output["original_size"]
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
            logger.debug(f"Saved depth map to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving depth PNG to {output_path}: {e}")

    def _save_fo3d(self, vggt_output: Dict, output_path: Path):
        """Save 3D point cloud as fo3d file."""
        try:
            # Generate point cloud from depth
            world_points = unproject_depth_map_to_point_map(
                vggt_output["depth"],
                vggt_output["extrinsic"], 
                vggt_output["intrinsic"]
            )
            
            # Get colors from input image (convert from CHW to HWC)
            colors = vggt_output["images"].transpose(1, 2, 0)  # (H, W, 3)
            confidence = vggt_output["depth_conf"]  # (H, W)
            
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
            
            # Save temporary PCD file
            temp_pcd_path = output_path.with_suffix('.pcd')
            o3d.io.write_point_cloud(str(temp_pcd_path), pcd, write_ascii=False)
            
            # Convert to fo3d
            scene = fo.Scene()
            scene.camera = fo.PerspectiveCamera(up="Z")
            mesh = fo.PointCloud("pointcloud", str(temp_pcd_path))
            scene.add(mesh)
            scene.write(str(output_path))
            
            # Clean up temporary PCD file
            temp_pcd_path.unlink()
            
            logger.debug(f"Saved fo3d file to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving fo3d file to {output_path}: {e}")

    def _save_keypoints(self, vggt_output: Dict, output_path: Path):
        """Save keypoint tracking data as JSON file."""
        try:
            # Prepare keypoint data for serialization
            track_data = vggt_output["point_tracks"]
            
            # Convert to normalized coordinates [0, 1]
            orig_w, orig_h = vggt_output["original_size"]
            normalized_points = []
            for x, y in track_data["tracks_2d"]:
                normalized_points.append([x / orig_w, y / orig_h])
            
            keypoint_data = {
                "points_2d_normalized": normalized_points,  # [0, 1] coordinates
                "points_2d_pixel": track_data["tracks_2d"],  # Pixel coordinates
                "tracks_3d": track_data["tracks_3d"],
                "visibility_scores": track_data["visibility_scores"],
                "confidence_scores": track_data["confidence_scores"],
                "original_size": vggt_output["original_size"],
                "vggt_size": vggt_output["vggt_size"],
                "camera_extrinsic": vggt_output["extrinsic"].tolist(),
                "camera_intrinsic": vggt_output["intrinsic"].tolist(),
            }
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(keypoint_data, f, indent=2)
            
            logger.debug(f"Saved keypoints to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving keypoints to {output_path}: {e}")


class VGGTOutputProcessor(fout.OutputProcessor):
    """Processes VGGT model outputs into FiftyOne heatmap labels.
    
    Args:
        classes (None): unused, for compatibility with base class
    """
    
    def __init__(self, classes=None, **kwargs):
        super().__init__(classes, **kwargs)
    
    def __call__(self, predictions, frame_size, confidence_thresh=None):
        """Process VGGT predictions into FiftyOne heatmap labels.
        
        Args:
            predictions: list of prediction data from VGGTModel.predict_all()
            frame_size: the (width, height) of the frames (unused)
            confidence_thresh: unused
            
        Returns:
            list of fol.Heatmap instances
        """
        heatmaps = []
        
        for pred_data in predictions:
            if pred_data is None:
                # Failed prediction
                heatmaps.append(None)
                continue
                
            try:
                heatmap = fol.Heatmap(
                    label="depth_map",
                    map_path=pred_data['depth_map_path']
                )
                heatmaps.append(heatmap)
                
            except Exception as e:
                logger.error(f"Error creating heatmap: {e}")
                heatmaps.append(None)
        
        return heatmaps