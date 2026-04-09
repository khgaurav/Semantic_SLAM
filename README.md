# Semantic SLAM

## EECE 5554: Robotics Sensing and Navigation Final Project Proposal
### Semantic Embedding Maps: Lightweight Maps with VLMs
**Gaurav Kothamachu Harish**  
**Stanley Kim**  
**Hayden Pavlovic**  
February 27, 2026

## 1. Objective
The objective of this project is to design and evaluate a hybrid localization technique that
leverages Lidar-Inertial SLAM for accurate map construction, while enabling lightweight
embedding-based localization via a Vision Language Model. The proposed system
performs embedding-to-embedding nearest-neighbor retrieval to estimate the camera's
pose by selecting the most similar keyframe. This approach significantly reduces the map
representation required for localization to a compact set of poses and embeddings,
therefore eliminating the need to use large point clouds at runtime. We hypothesize that
this hybrid approach can achieve similar localization accuracy compared to a SLAM
baseline while substantially reducing map storage requirements.

## 2. Background
### 2.1 Localization techniques
Metric-based approaches to localization, such as visual or LiDAR-based SLAM, localize a
robot by estimating a robot pose to a geometric map via sensor observations, using
techniques such as ICP or NDT. While accurate under stable conditions, the approach is
sensitive to environmental changes such as lighting and new objects, which can degrade
long-term performance. Topological localization methods focus on previously visited
places rather than using geometric alignment. Image retrieval techniques such as Bag-of-
Words enable place recognition by comparing feature vectors. However, topological
localization can suffer from perceptual aliasing, repetitive environments, and coarse
embeddings which degrade performance.

### 2.2 Visual Language models
Visual-Language models are deep neural networks trained on both images and text. VLMs
can produce more robust embeddings as they capture higher level semantic embeddings.
For example, two images of the same corridor under different lighting conditions will have
different pixel information but still yield similar embeddings through a VLM. While VLMs
have been extensively studied for tasks such as zero-classification, captioning and image
retrieval, integration into robotic mapping and localization remains an emerging field.

## 3. Methodology
### 3.1 System Architecture
The system consists of two decoupled pipelines: mapping and localization. They are
connected only through the on-disk semantic map.

### 3.2 Mapping Pipeline
**LiDAR-Inertial SLAM.** LIO-SAM [1] ingests 3D LiDAR point clouds and 9-axis IMU data from
the dataset. The output is a set of keyframe poses and a globally consistent registered
point cloud.

**VLM Embedding Extraction.** At each keyframe, the nearest-in-time RGB camera image is
retrieved via timestamp matching. The image is passed through SigLIP 2 ViT-B/16 [2] in a
single forward pass to produce a global embedding vector. It produces embeddings that
serve as a visual place recognition descriptor via embedding-to-embedding cosine
similarity.

**Map Storage.** The global embeddings for all keyframes are inserted into a FAISS index
configured for exact inner-product search. Each entry stores the embedding vector, the
associated SE(3) pose, and a keyframe identifier. The resulting map is saved as a compact
set of files: the FAISS index binary, a NumPy array of keyframe poses, and metadata.

### 3.3 Localization Pipeline
Given a query RGB image, first we run SigLIP 2 on the query image to produce a global
descriptor. Then we Query the pre-loaded FAISS index for the top-k most similar keyframes
by cosine similarity. Finally, we can return the SE(3) pose of the best-matching keyframe as
the localization estimate.

### 3.4 Dataset
M2DGR (SJTU) [3]: This dataset provides a synchronized Velodyne 3D LiDAR, six fisheye
cameras, and an Xsens 9-axis IMU with both motion-capture and RTK ground truth across
36 sequences spanning indoor hallways, outdoor gardens, and parking structures.

## 4. Expectations
In this project, we aim to demonstrate that keyframe-level semantic embeddings provide a
lightweight, robust alternative to traditional dense SLAM point clouds. By storing global
VLM embeddings at keyframe poses instead of dense geometry, we expect to achieve
significant map compression while preserving essential structural and semantic
information. We also anticipate improved re-localization performance under
environmental changes, also with increased robustness to moderately dynamic
environments, even without explicit map updates for moved or removed objects.

Performance will be evaluated using Absolute Trajectory Error (ATE), re-localization
success rate, and map storage size, compared against a geometric SLAM baseline. We
expect comparable localization accuracy with substantially reduced memory usage and
improved resilience to environmental variation.

## 5. References
[1] T. Shan, "LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and
Mapping," in 2020 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS), Las Vegas, NV, USA, 2020.

[2] M. Tschannen, "SigLIP 2: Multilingual Vision-Language Encoders with Improved
Semantic Understanding, Localization, and Dense Features," arXiv:2502.14786, 2025.

[3] J. Yin, "M2DGR: A Multi-sensor and Multi-scenario SLAM Dataset for Ground Robots,"
in RA-L & ICRA2022, 2022.
