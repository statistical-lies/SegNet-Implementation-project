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
