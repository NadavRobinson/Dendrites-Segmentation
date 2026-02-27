# Final Project in Image Processing Course: Morphological Segmentation of Dendrites in SEM

## 1. Scientific Background
* Dendrites in lithium batteries are one of the most significant technological barriers in developing the next generation of high-energy batteries.
* These are microscopic metallic structures with a fractal geometry resembling branches, which grow on the anode during charging cycles.

**Principles of the physical mechanism**
To understand the engineering challenge in detection, one must understand the formation process:
1. **Deposition process:** During charging, lithium ions migrate from the cathode to the anode. Under ideal conditions, the ions intercalate in a uniform and layered manner.
2. **Formation of irregularities:** Under extreme conditions (such as high current density or low temperature), a non-uniform accumulation of ions occurs on the surface. The lithium ions accumulate locally and form a needle-like crystal that grows vertically from the surface.
3. **Dielectric failure:** The main danger lies in the sharpness of the structure. An overgrown dendrite might puncture the separating membrane (Separator), creating an internal short circuit, and lead to Thermal Runaway.

**Project Goal:** Developing a system for automatic segmentation of dendrites from Scanning Electron Microscope (SEM) images for the purpose of predicting and preventing failures.

---

## 2. Task Definition: Segmentation in a Noisy Environment
* The main objective is to perform Semantic Segmentation: Pixel-wise classification to separate the dendrite structure from the background.
* **The technical challenge:** SEM images are characterized by a low Signal-to-Noise Ratio (SNR). They include artifacts such as saturated ("burned") areas, blurring caused by the charging effect, and soft gradients.
* One idea (this is just one idea out of many possible proposals) is to treat the image as a topographic map, where intensity values represent height, and not to rely solely on a global threshold.

**Required deliverables:**
1. **Binary Mask:** Separation between the dendrite and the background.
2. **Pre-processing and cleaning:** Removal of foreign elements (technical text, scales) and sensor noise.
3. **Skeletonization:** Extracting the topological structure to a single-pixel wide line (Centerline Extraction) for geometric calculations.

---

## 3. Methodology: Comparative Analysis (Deep Learning vs. Classic CV)
In this project, you will be required to implement and compare two complementary approaches to solving the problem:

| Criterion | Approach A: Deep Learning (YOLO-Seg) | Approach B: Classic Image Processing |
| :--- | :--- | :--- |
| **Architecture** | Using SOTA Models such as YOLOv8/v11. | Developing a deterministic pipeline based on mathematical morphology. |
| **Principle of Operation** | Transfer Learning and identifying non-linear patterns. | Isolating objects based on geometric and statistical rules. |
| **Data Dependency** | High – requires an accurately tagged dataset (Supervised). | Low – can be applied to a small sample without training. |
| **Robustness** | High resilience to complex texture changes. | Sensitive to lighting changes; requires parameter tuning (Finetuning). |

---

## 4. Data Preparation
* The model's quality directly depends on the quality of the data and tagging.
* For Approach A which needs to be implemented, training a YOLO model, you can use one of the following tools:
  * **Roboflow:** Recommended for use due to its Auto-labeling capabilities and native export to YOLO Segmentation format. Suitable for unclassified projects.
  * **CVAT (Computer Vision Annotation Tool):** The industry standard for CV projects requiring data privacy (On-premise).
* **Tagging method:** It is mandatory to use Polygons only.

---

## 5. Implementing the Classic Approach
To deal with noisy images without using neural networks, you must implement the processing chain (Pipeline) - an initial recommendation for implementation (there are others):
1. **Contrast Enhancement:** Using CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve local contrast and highlight details in shaded areas.
2. **Denoising:** Applying a Bilateral Filter. This choice is critical for Edge Preserving while smoothing noise in uniform areas.
3. **Morphological Reconstruction:** An advanced technique based on Geodesic Dilation:
   * Performing aggressive Erosion to create certain "Seeds/Markers".
   * Reconstructing the structure using Conditional Dilation.
   * Where the original mask serves as an upper boundary. This method is especially effective for filtering noise that is not connected to the main structure.
4. **Separation Algorithms:** Using the Distance Transform algorithm combined with Watershed to separate branches that touch each other.

---

## 6. Evaluation Metrics
The quality of the solution will be tested based on engineering criteria:
1. **IoU / Dice Score:** Measuring the overlap between the algorithm's result and the manual Ground Truth.
2. **Robustness:** Testing the algorithm's stability under different images.
3. **Failure Analysis:** In-depth discussion of cases where the system failed, characterizing the reasons (such as: borderline resolution, unique artifacts).

---

## 7. Submission Guidelines
The project submission will be in a single folder (Zip) including the following components:

**A. Summary Report (PDF)**
The report structure will be like a technical paper and will include:
1. **Abstract:** Defining the problem, the chosen solution, and the main results.
2. **Methodology:** Detailed explanation of both Pipelines (Classic and DL), including parameter selection.
3. **Results and Discussion:** Presenting accuracy metrics (IoU/Precision/Recall), comparison tables, and visual analysis of successes and failures.
4. **Conclusions:** Which method to use in which scenario.

**B. Source Code**
* Organized and documented code (Docstrings + Comments).
* A `requirements.txt` file for installation.
* A `README.md` file with clear run instructions (how to run training, how to run Inference).
* Implementation can be done with `cv2` and `scikit-image`.

**C. Results Folder (Artifacts)**
* **Visual examples:** At least 5 images showing: Source Image -> Mask (Classic) -> Mask (YOLO) -> Skeleton.
* **Weights files:** A `.pt` file of the best trained model (Best Model). Providing a shared Drive link is allowed.

**D. Presentation**
* A 5-minute presentation including a live demonstration with a recorded video of the algorithm running on a new dataset.

**Good luck!**

To receive the data for the exercise, register at the following link:
https://forms.gle/zeTSoa78uBNgLrMM7

---

## Appendix - A slightly more expanded proposal for image processing
**This is just a proposal**, any other approach can be implemented.

### Stage A: Pre-processing
Scanning Electron Microscopy (SEM) images often suffer from "grainy" noise (Shot Noise), uneven lighting (Vignetting), and low contrast at the thin edges of the dendrites. The goal of this stage is to normalize the raw data and maximize the separation between the object and the background before the algorithm starts making decisions.

1. **Histogram Normalization:**
   * Before any complex operation, a linear stretch of pixel values to the full range must be performed.
   * **Rationale:** This ensures a "common ground" for all images, so that threshold values set later will be stable even if one image was taken at a different exposure than another.
2. **Local Contrast Enhancement (CLAHE):**
   * Using standard (global) Histogram Equalization is a common mistake in these images, as it tends to "burn" bright areas and amplify noise in dark areas. The solution is CLAHE (Contrast Limited Adaptive Histogram Equalization).
   * **How it works:** The algorithm divides the image into areas (Tiles) and calculates a separate histogram for each area, while limiting the amplification (Clipping) to prevent background noise from becoming an "object".
   * **The result:** A dramatic improvement in identifying thin and delicate branches, even if they are in a shaded area of the electrode.

### Stage B: Segmentation & Denoising
Converting a continuous grayscale image to a binary map (black/white), while constantly struggling between removing noise and preserving details.

1. **Edge-Preserving Filter:**
   * Common filters like Gaussian Blur perform spatial averaging and therefore blur the edges of the dendrites. This blurring causes a loss of critical information about the true branch thickness.
   * **The recommended solution:** Bilateral Filter.
   * **Mechanism of action:** The filter smooths surfaces (Denoising), but it includes an additional component that measures the color intensity difference. If it detects a sharp "jump" (Edge), it stops the smoothing at that point. This preserves the sharpness of the dendrite's walls while internal and external noise is removed.
2. **Thresholding Strategies:**
   * The choice of method depends on image quality:
   * **Adaptive Thresholding (generally recommended):** The preferred method for SEM images. The algorithm calculates a dynamic threshold for each pixel separately based on the average of its neighbors.
   * **Otsu's Binarization:** Suitable only if the lighting is completely uniform (without shading). This method is statistical-global and finds the optimal threshold that minimizes the variance within the two classes (background and object).

### Stage C: Post-Processing (Morphological Processing)
After binarization, the raw result is not perfect. It often contains "holes" inside the body of the branches, segmentations in thin branches, and "islands" of random noise in the background. This stage is intended to correct these defects using geometric logic.

1. **Morphological Reconstruction:**
   * This is an advanced technique for cleaning noise without damaging the dendrite structure (unlike regular Opening operations which might erase thin branches).
   * The method works based on two images:
     * **The Mask:** The original binary image (containing the full dendrite + noise).
     * **The Marker:** An image that underwent aggressive Erosion, leaving only the most certain and thickest "seeds" of the main branches (with no background noise at all).
   * **The Process:** Iterative Dilation of the Marker into the Mask is performed. The seeds grow until they meet the boundaries of the original dendrite in the mask.
2. **Final Cleaning (Cleaning Artifacts):**
   * **Closing:** A morphological operation to close holes to create a full sequence.
   * **Connected Components Analysis:** Any object whose area is smaller than a certain threshold (e.g., under 50 pixels) is classified as noise and removed, under the physical assumption that a dendrite is a continuous and large structure.