# Model Selection and Justification

## Selected Model: PointNet

### Architecture Overview

PointNet is a pioneering deep learning architecture designed to process unordered point cloud data directly. The model uses shared MLPs (implemented as 1D convolutions) followed by max pooling to extract global features, combined with local point features for per-point segmentation.

**Key Components:**
- **Point-wise feature extraction:** 1D convolutions process each point independently
- **Global feature aggregation:** Max pooling creates a permutation-invariant global descriptor
- **Segmentation head:** Concatenates local and global features for per-point classification

### Justification

**1. Simplicity and Reproducibility**
- Pure PyTorch implementation with no external dependencies requiring compilation
- Easy to install, debug, and modify
- Reproducible across different platforms (Windows, Linux, macOS)

**2. Direct Point Cloud Processing**
- Works on raw XYZ coordinates without voxelization
- No preprocessing overhead or information loss from discretization
- Suitable for outdoor LiDAR data with varying point densities

**3. Established Baseline**
- Well-documented architecture with extensive research validation
- Provides a solid baseline for comparison with more complex methods
- Proven effectiveness on semantic segmentation tasks

**4. Resource Efficiency**
- Lower memory footprint compared to voxel-based approaches
- Faster inference on sparse outdoor scenes
- Suitable for deployment on edge devices

### Alternative Approaches Considered

**Sparse Convolutional U-Net (MinkowskiEngine/spconv):**
- **Pros:** Better accuracy (60-70% mIoU), memory efficient for large scenes
- **Cons:** Complex installation requiring CUDA compilation, platform-specific dependencies
- **Status:** Initially attempted but encountered compilation issues on Python 3.12 with MinkowskiEngine

**Technical Challenges Encountered:**
During development, significant effort was invested in implementing sparse convolutions using MinkowskiEngine:
- MinkowskiEngine lacks Windows support and requires specific CUDA/PyTorch version alignment
- Python 3.12 compatibility issues with numpy.distutils deprecation
- CUDA compilation errors due to version mismatches (system CUDA 12.0 vs PyTorch CUDA 12.1)
- Environmental setup consumed excessive time for a time-constrained assessment

Alternative sparse convolution library (spconv) was also evaluated but presented similar installation complexity.

### Production Deployment Considerations

For production deployment with more time and resources:
1. **Sparse U-Net with MinkowskiEngine** for highest accuracy
2. Use Docker containers with pre-built environments to avoid compilation issues
3. Target Linux-based deployment platforms
4. Implement PointNet as a fallback for edge deployment or testing

### Expected Performance

Based on literature and similar datasets:
- **PointNet on outdoor LiDAR:** 35-50% mIoU
- **Sparse U-Net (production target):** 60-75% mIoU

The current PointNet implementation provides a working baseline that can be trained and evaluated within assessment constraints while maintaining code quality and reproducibility.

### Dataset Challenges

**PandaSet Size:** The full dataset is approximately 33GB, containing 103 sequences with dense LiDAR and annotation metadata. This size poses practical challenges related to bandwidth, disk space, and processing time—especially on standard hardware.

**Solution Implemented:**
- Built a clean Python-based download script (download_pandaset.py) using the Kaggle API.
- The script downloads the entire dataset programmatically, fulfilling the assessment requirement for Python-controlled acquisition.
- The full dataset is extracted locally, but development and experimentation were done on a small subset (e.g., sequences 001, 002) to keep iterations fast and resource usage manageable.
