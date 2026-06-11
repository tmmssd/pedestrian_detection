# pedestrian_detection

## TODO

- [x] Create standard model
- [ ] Make the dataset downloadable
- [x] Test with the whole dataset (see "Dataset Structure & File Descriptions: OneDrive Repository")
- [ ] Implement parameter optimization
- [ ] Modify the code to make it work also on Colab

## Dataset Structure & File Descriptions: OneDrive Repository

### 1. What "Raw pedestrian data" Files Represent

The **12 files** with the **`.aedat`** extension (`1.aedat` to `12.aedat`) are raw, asynchronous recordings captured using a **DAVIS346redColor event-based camera** with a resolution of **346x260** (as summarized in the following image).

![alt text](image.png)

* Unlike traditional frame-based cameras, this neuromorphic sensor records only pixel-level changes in brightness as a continuous stream of events defined by timestamp, $(x, y)$ coordinates, and polarity.
* Each recording has an **average length of 30 seconds**, capturing pedestrians walking across real-world environments including corridors, streets, and squares under varying weather conditions (**Sunny and Rainy**).

### 2. How They Relate to the "train" Folder

Because deep learning models cannot directly ingest raw, continuous event streams, the neuromorphic data is pre-processed into a standard dataset format. The **"train"** folder contains the structured outputs of this transformation:

* **"pedestrian frame":** The continuous event streams from the 12 `.aedat` recordings are accumulated over fixed time windows to reconstruct **4,670 2D frames**, highlighting moving subjects while filtering out static backgrounds.
* **"pedestrian label":** This folder contains **4,670 corresponding ground-truth annotations** (bounding boxes), mapping the precise locations of the pedestrians for binary object detection.

### Summary

**"Raw pedestrian data"** contains the 12 original neuromorphic video sequences (30s each, 346x260 resolution) from the DAVIS sensor. **"train"** represents that same source material sliced, integrated, and annotated into a ready-to-use dataset of 4,670 frame-label pairs for model training.