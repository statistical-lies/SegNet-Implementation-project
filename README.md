# SegNet: Deep Convolutional Encoder-Decoder for Image Segmentation

This repository contains a PyTorch implementation of the **SegNet** architecture, as described in the paper *"SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation"* (Badrinarayanan et al., 2017).

This project was developed as an implementation study to reproduce the results on the **CamVid** road scene dataset.

## 📄 Paper Summary
**SegNet** is designed to be efficient in terms of memory and computational time. Its core novelty is the **Encoder-Decoder** structure where:
1.  **Encoder:** Uses VGG16 topological layers to extract features.
2.  **Max-Pooling Indices:** Crucially, the encoder stores the *indices* of max-pooling operations.
3.  **Decoder:** Uses these stored indices to perform sparse upsampling (Max Unpooling), eliminating the need for learning upsampling parameters.

## 📂 Project Structure
* `main.py`: Contains the **SegNet Model** definition, the **Dataset Class**, and the **Training Loop**.
* `evaluate.py`: Calculates quantitative metrics (Global Accuracy, Class Average, mIoU) on the test set.
* `visualize_final.py`: Generates a side-by-side grid of Input images vs. Model Predictions.
* `CamVid/`: Directory containing the dataset and split definition files (`train.txt`, `val.txt`, `test.txt`).

## 💿 Dataset: CamVid
This implementation uses the **CamVid (Cambridge-driving Labeled Video Database)** dataset, a standard benchmark for road scene segmentation.

* **Source:** [University of Cambridge - Machine Intelligence Laboratory](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* **Context:** The dataset consists of high-definition video frames captured from a driving automobile.
* **Preprocessing:** Following the methodology in the SegNet paper, we utilize the **11-class version** of the dataset (grouping the original 32 classes into 11 macro-classes) to focus on major road features.
* **Classes:** `Sky`, `Building`, `Pole`, `Road`, `Pavement`, `Tree`, `SignSymbol`, `Fence`, `Car`, `Pedestrian`, `Bicyclist`.
* **Resolution:** Images are resized to **360×480** for training, as specified in the original SegNet benchmarks.
* **Splits:** We utilize the specific training/validation/test splits provided by the SegNet authors in their [original tutorial](https://github.com/alexgkendall/SegNet-Tutorial).

**Citation:**
> G. Brostow, J. Fauqueur, and R. Cipolla, "Semantic object classes in video: A high-definition ground truth database," *Pattern Recognition Letters*, vol. 30, no. 2, pp. 88-97, 2009.

## 🛠️ Implementation Details
This implementation features:
* **Framework:** PyTorch
* **Architecture:** 13-layer VGG-style Encoder and matching Decoder.
* **Upsampling:** implemented via `nn.MaxUnpool2d` to utilize pooling indices.
* **Class Balancing:** Implemented **Median Frequency Balancing** (hardcoded weights based on the original Caffe implementation) to solve the "dominant class" problem where the model initially collapsed to predicting only 'Sky' or 'Road'.
* /SegNet-Implementation


        |-- main.py
  
        |-- CamVid/
  
        |-- train.txt
  
        |-- val.txt
  
        |-- test.txt


## Result after model training

  <img width="1990" height="1180" alt="image" src="https://github.com/user-attachments/assets/0b767130-01c1-4262-863d-0cdc7aab0197" />


### 📉 Current Evaluation Results (Work in Progress)

The following metrics represent the current state of the model during training.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Global Accuracy** | 16.94% | Percentage of total pixels correctly classified. |
| **Class Average Accuracy** | 16.94% | Average accuracy across all 11 classes. |
| **Mean IoU** | 1.54% | Intersection over Union (Standard Segmentation Metric). |

### Per-Class Breakdown
| Class | IoU Score |
| :--- | :--- |
| **Sky** | **16.94%** |
| Building | 0.00% |
| Pole | 0.00% |
| Road | 0.00% |
| Pavement | 0.00% |
| Tree | 0.00% |
| SignSymbol | 0.00% |
| Fence | 0.00% |
| Car | 0.00% |
| Pedestrian | 0.00% |
| Bicyclist | 0.00% |

*The Current results indicate the model is biased towards the dominant class (Sky) and requires further training epochs or hyperparameter tuning to resolve the issue.*


## 📊 Final Quantitative Results (Improved)

After training for 100 epochs with Median Frequency Balancing, the model achieved the following performance on the CamVid test set. These results align closely with the benchmarks reported in the original paper (SegNet-Basic).

| Metric | My Result | Paper Benchmark (Approx) | Status |
| :--- | :--- | :--- | :--- |
| **Global Accuracy (G)** | **83.76%** | 82.8% | ✅ Replicated |
| **Class Average Accuracy (C)** | **59.33%** | 62.0% | ✅ Replicated |
| **Mean IoU (mIoU)** | **47.91%** | 46.3% | ✅ Replicated |

### Per-Class IoU Breakdown
The class weighting successfully resolved the mode collapse issue, allowing the model to detect smaller objects like Poles and Signs.

| Class | IoU Score |
| :--- | :--- |
| **Sky** | 89.45% |
| **Road** | 87.80% |
| **Building** | 68.77% |
| **Car** | 65.36% |
| Pavement | 62.82% |
| Tree | 61.88% |
| Pedestrian | 26.75% |
| Bicyclist | 20.01% |
| Pole | 17.77% |
| Fence | 13.47% |
| SignSymbol | 12.97% |


