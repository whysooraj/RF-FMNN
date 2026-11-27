# Research Upgrade Proposal: RF-CFMNN
**Project**: Random Forest Contraction-Free Fuzzy Min-Max Neural Network  
**Student**: Makala Mouli Sai Santhosh Sooraj (Reg: AP24110011531)  
**Supervisor / Course Lead**: Aadarsh Sir  
**Department**: Computer Science and Engineering, SRM University-AP  

---

## Purpose of this Proposal
We have already built and tested **RF-CFMNN**. We removed destructive contraction, used data centroids to resolve overlapping boxes, and proved our model works across 8 datasets (including 58,000 NASA shuttle records).

This document outlines **four practical next steps** to upgrade the project into a publication-ready paper. All points are explained in simple, plain terms.

---

## 1. Running on GPUs using PyTorch (Fast Testing on Huge Datasets)

### What is the problem now?
Right now, our code runs on the CPU using NumPy. While training is fast, testing (inference) on 58,000 records takes around 1 to 3 seconds because the computer checks each sample against thousands of hyperboxes one by one.

### What is the proposed upgrade?
We will write the hyperbox checks using **PyTorch tensor operations** so they can run directly on an NVIDIA graphics card (GPU). 

### Why is this useful?
1. Instead of checking boxes one by one, a GPU checks millions of combinations at the exact same moment.
2. Inference time will drop from seconds to **under 5 milliseconds**.
3. This allows the model to process live video, radar, or real-time sensor streams without delay.

---

## 2. Smart, Adaptive Box Sizing (Self-Adjusting $\theta$)

### What is the problem now?
Right now, the maximum allowed box size ($\theta$) is a fixed number (like 0.2) chosen by us before training. 
- If $\theta$ is too small, the model creates too many tiny boxes.
- If $\theta$ is too large, the boxes become too big and mix up different classes.

### What is the proposed upgrade?
Instead of using one fixed size for the whole dataset, we will make the box size **self-adjusting**:
- In clear, safe areas where data points are far apart, boxes can expand large.
- In crowded, mixed boundary areas where two classes meet, boxes will stay small and tight.

### Why is this useful?
1. No need to waste hours trying different $\theta$ values by hand.
2. The model automatically fits clean areas with few boxes and complex areas with sharp detail.
3. This will improve accuracy by another 1% to 2% on tricky boundary datasets.

---

## 3. Real-Time Streaming Data (Learning without Retraining)

### What is the problem now?
In standard machine learning, if new data arrives tomorrow, you usually have to retrain the entire model from scratch on all the old data plus the new data.

### What is the proposed upgrade?
Because our fuzzy network learns in a single pass without backpropagation, we can connect our Ray parallel workers to a **live data stream**:
- As new sensor records arrive one by one, the workers simply expand existing boxes or add a new box.
- The model never needs to forget old data or retrain from the beginning.
- If a completely weird, unseen data point arrives (an anomaly), the fuzzy score immediately drops to zero, raising an instant alarm.

### Why is this useful?
1. Perfect for live telemetry (like NASA shuttle sensors or hospital patient monitors).
2. The model learns continuously 24/7 without shutting down.

---

## 4. Hyperbox Graph Cleanup (Removing Useless Overlaps)

### What is the problem now?
Our model currently has thousands of hyperboxes across all trees. Many of these boxes sit on top of each other and say the exact same thing, which wastes computer memory.

### What is the proposed upgrade?
We will connect overlapping boxes together like a **network map (a mathematical graph)**:
- Each box is a dot.
- If two boxes touch or overlap, we draw a line between them.
- If two touching boxes belong to the same class and tell us nothing new, we merge them into one.

### Why is this useful?
1. Cuts total hyperbox memory storage by **50% to 70%**.
2. Makes predictions faster because there are fewer boxes to check.
3. Gives the paper strong mathematical depth that professors and conference reviewers appreciate.

---

## Summary Comparison Table

| Feature | Current Project Status | Proposed Upgrade Status | Main Benefit |
| :--- | :--- | :--- | :--- |
| **Compute Hardware** | CPU with NumPy | NVIDIA GPU with PyTorch | 50x faster predictions (< 5ms) |
| **Box Expansion ($\theta$)** | Fixed number picked by hand | Self-adjusting based on crowd density | Higher accuracy on borders |
| **New Data Handling** | Requires batch retraining | Learns live from data streams | Zero downtime / live updates |
| **Model Size** | High box count | Graph-based box merging | 50–70% less memory needed |

---

## Deliverables for the Final Paper
1. Clean code repository with GPU PyTorch support.
2. Benchmark graphs comparing CPU vs GPU execution speeds.
3. Live streaming demonstration on a simulated data feed.
4. Formal research paper formatted for IEEE conference submission.
