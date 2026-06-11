# pedestrian_detection

## TODO

- [x] Create standard model
- [x] Make the dataset downloadable (see "Dataset Structure & File Descriptions: OneDrive Repository")
- [ ] Test with the whole dataset
- [ ] Implement parameter optimization
- [ ] 

## Dataset Structure & File Descriptions: OneDrive Repository
### 1. What "Raw pedestrian data" Files Represent

The files with the **`.aedat`** extension (e.g., `1.aedat`, `2.aedat`) are recordings from an **event-based camera** (a neuromorphic vision sensor).

* Unlike traditional cameras that capture full frames at a fixed frame rate, event cameras only record pixel-level changes in brightness asynchronously.
* Each `.aedat` file contains a continuous stream of individual spikes/events, where each event records the exact timestamp, coordinates $(x, y)$, and polarity (increase or decrease in light) of the motion.
* In this specific dataset, these files represent raw recordings of pedestrians moving across the camera's field of view.

---

### 2. How They Relate to the "train" Folder

Standard neural networks cannot process a continuous, raw stream of asynchronous micro-events directly. Therefore, the raw data must be pre-processed and framed. The **"train"** folder contains the structured output of this conversion:

* **"pedestrian frame":** The raw event stream from the `.aedat` files is accumulated over fixed time windows (e.g., every 30ms) to reconstruct 2D artificial frames. These images capture the contours of moving pedestrians while stationary backgrounds disappear.
* **"pedestrian label":** This folder contains the corresponding ground-truth annotations (typically `.txt` or `.xml` files). For every generated frame, a label file provides the bounding box coordinates marking exactly where the pedestrians are located.

### Summary

**"Raw pedestrian data"** is the raw, neuromorphic source material. **"train"** is that exact same data sliced into synchronized frames and paired with bounding boxes, formatted specifically for the object detection model to learn from.