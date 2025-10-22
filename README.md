# Multilingual-Multimodal-Sarcasm-Detection-Using-Text-and-Emojis

## Model Overview

This project implements a **multimodal sarcasm detection model** for tweets using:

- **[XLM-RoBERTa-base]** as the core multilingual encoder
- A learned **emoji embedding layer** to represent emojis explicitly
- A **fusion mechanism** combining the [CLS] token output with the averaged emoji embeddings
- A **deep classifier (MLP)** with dropout regularization
- **Focal Loss** to address class imbalance and focus on hard examples

We designed this architecture to handle both:
- Explicit sarcasm (emoji, hashtags, exaggeration)
- Implicit and **subtle sarcasm**, including political or context-dependent tweets with no emojis

### Why XLM-RoBERTa?

- It's **multilingual**, enabling future expansion to non-English tweets (e.g. Persian)
- It has shown strong performance on social media tasks in low-resource scenarios

### Why Emoji Embedding?

- Emojis often play a key role in sarcasm
- Instead of relying on raw Unicode or removing them, we **learn a dedicated embedding space**



##  Optimization Journey

The model was initially fine-tuned with **all transformer layers unfrozen**, which led to **overfitting**:  
- **Train Accuracy**: ~0.85  
- **Validation Accuracy**: ~0.66  
- **Trainable Parameters**: ~278M  

To mitigate this, I applied several regularization strategies:

-  **Freezing early transformer layers** (initially only layers 10–11 were trainable)  
-  Introduced **Dropout** (started at 0.5, later tuned) and **Weight Decay** (`0.01`)  
-  Applied **class weights** in the loss function to address class imbalance  

**Result:**  
- `val_acc ≈ 0.67`, `train_acc_epoch ≈ 0.79`  
- Validation performance:
non-sarcastic: f1 = 0.70
sarcastic: f1 = 0.63 (recall = 0.58)

##  Data Augmentation Experiments
To improve recall on the sarcastic class (which was initially just 0.52), I tried targeted data augmentation:

1) Added external sarcastic samples from SemEval dataset

2) Generated sarcastic tweets using prompt-based LLMs (OpenChat with one-shot learning)

However:

These additions helped on training/validation, but test set performance dropped, likely due to domain mismatch and noisy generation.

##  Loss-Level Optimization

Shifting strategy, I focused on improving the loss function:

-  Introduced **Focal Loss** to give more weight to hard-to-classify sarcastic samples  
-  **Unfroze more transformer layers (7–11)**  
-  **Increased Dropout** to strengthen generalization

This final setup yielded:

- **Recall (sarcastic)**: improved from **0.58 → 0.75**  
- **F1-score (sarcastic)**: improved from **0.63 → 0.67**  
- **Test Accuracy**: **0.71**

### Final Test Set Performance

| Class          | Precision | Recall | F1-score | Support |
|----------------|-----------|--------|----------|---------|
| non-sarcastic  | 0.81      | 0.68   | 0.74     | 473     |
| sarcastic      | 0.61      | 0.75   | 0.67     | 311     |
| **Accuracy**   |           |        | **0.71** | **784** |
| **Macro Avg**  | 0.71      | 0.72   | 0.70     | 784     |
| **Weighted Avg** | 0.73    | 0.71   | 0.71     | 784     |


##  What's Next?

-  **Add Persian sarcasm data** to evaluate XLM-R's multilingual capabilities  
-  Explore **Contrastive Learning** to separate literal vs sarcastic intent more effectively

###  Download Trained Model

You can download the trained model checkpoint (`best_model.ckpt`) from [this Google Drive link] (https://drive.google.com/file/d/1c9wIv9qYzPWo9v7V2_Awfe63UF53kDyS/view?usp=drive_link).
  
##  How to Use:
##  Multilingual___Multimodal_Sarcasm_Detection.py
Enter a tweet: Oh great, another Monday!😢

Prediction: Sarcastic 
Probabilities → Non-sarcastic: 0.18, Sarcastic: 0.82
