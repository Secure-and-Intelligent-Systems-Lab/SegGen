# SegGen: An Unreal Engine 5 Pipeline for Generating Multimodal Semantic Segmentation Datasets

SegGen is a fully automated dataset generation pipeline built in Unreal Engine 5 (UE5). It leverages procedural biome generation and a spline-driven drone actor to capture synchronized RGB, depth, and pixel-perfect semantic labels.

The pipeline enables scalable creation of multimodal semantic segmentation datasets for remote sensing, environmental monitoring, and general computer vision tasks.

## Features

- Procedural scene generation using UE5’s biome tools.

- Automated drone flight path and image capture along user-defined splines.

- Multimodal support: RGB, depth, and extensible to LiDAR, surface normals, etc.

- Pixel-perfect semantic label generation via object-to-class mappings.

- Modular, reproducible datasets with minimal human effort.

## Installation
### Requirements

- Unreal Engine 5.4+

- Windows/Linux system capable of running UE5 maps

- (Optional) Gaea for terrain generation


### The repository includes:

- Example biome definitions

- Drone spline actor Blueprints

- Semantic label material database

- Python post-processing tools

## Creating a Dataset with SegGen
### Step 1: Environment Setup

Generate terrain using Gaea or UE5’s Landscape Sculpting Tools.

Define biome regions with a color-coded biome mask image (each color = biome type).

Import into UE5 and assign biome definitions (forest, grassland, rocky, etc.).

### Step 2: Semantic Class Mapping

Edit the SemanticClassDB inside UE5.

Map object name substrings (e.g., "oak", "rock") → semantic label materials.

Example mappings are provided in the repo (coarse 7-class or detailed 50+ classes).

### Step 3: Drone Trajectory

Place the Drone Spline Actor in your scene.

Define the spline path and sampling interval.

Adjust camera settings (aperture, focal length) in the drone blueprint.

### Step 4: Automated Data Collection

Play the UE5 scene.

Press the “Collect Data” keybinding on the drone actor.

For each spline point, the pipeline captures:

RGB image → /data/rgb/

Depth map → /data/depth/

Semantic label → /data/labels/

### Step 5: Post-Processing

Run the provided Python script to convert RGB label images → integer masks.

Optionally apply morphological filtering to smooth labels.

python tools/label_postprocess.py --input ./data/labels --output ./data/labels_int

### Step 6: Training / Usage

The dataset is ready for semantic segmentation tasks.

Example splits: train/ (80%), test/ (20%).

## Example Dataset

Size: 1169 samples

Resolution: 1920×1080

Modalities: RGB, Depth, Semantic Labels

Classes: Ground, European Beech, Rocks, Norway Maple, Dead Wood, Black Alder

## Extending Modalities

SegGen is modular:

- UE5 built-in passes: albedo, surface normals, roughness

- LiDAR simulation: MetaLidar plugin (`https://github.com/metabotics-ai/MetaLidar`)

- Custom modalities: Add via Blueprint material passes





## Citation
If you use our work, please cite:

```bibtex
@Article{s25175569,
AUTHOR = {McMillen, Justin and Yilmaz, Yasin},
TITLE = {SegGen: An Unreal Engine 5 Pipeline for Generating Multimodal Semantic Segmentation Datasets},
JOURNAL = {Sensors},
VOLUME = {25},
YEAR = {2025},
NUMBER = {17},
ARTICLE-NUMBER = {5569},
URL = {https://www.mdpi.com/1424-8220/25/17/5569},
PubMedID = {40942996},
ISSN = {1424-8220},
ABSTRACT = {Synthetic data has become an increasingly important tool for semantic segmentation, where collecting large-scale annotated datasets is often costly and impractical. Prior work has leveraged computer graphics and game engines to generate training data, but many pipelines remain limited to single modalities and constrained environments or require substantial manual setup. To address these limitations, we present a fully automated pipeline built within Unreal Engine 5 (UE5) that procedurally generates diverse, labeled environments and collects multimodal visual data for semantic segmentation tasks. Our system integrates UE5’s biome-based procedural generation framework with a spline-following drone actor capable of capturing both RGB and depth imagery, alongside pixel-perfect semantic segmentation labels. As a proof of concept, we generated a dataset consisting of 1169 samples across two visual modalities and seven semantic classes. The pipeline supports scalable expansion and rapid environment variation, enabling high-throughput synthetic data generation with minimal human intervention. To validate our approach, we trained benchmark computer vision segmentation models on the synthetic dataset and demonstrated their ability to learn meaningful semantic representations. This work highlights the potential of game-engine-based data generation to accelerate research in multimodal perception and provide reproducible, scalable benchmarks for future segmentation models.},
DOI = {10.3390/s25175569}
}
```
