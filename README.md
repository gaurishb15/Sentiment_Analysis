# **Sentiment Classification Using a Custom Transformer Model**

## **Overview**
This project focuses on building and training a **custom Transformer model** from scratch to classify sentences into sentiment categories: **positive**, **negative**, or **neutral**. Unlike pre-trained models such as BERT or GPT, this implementation starts from the basics, demonstrating the power of a well-designed architecture.

The model was trained on a dataset of **social media sentences**:  
- **Training Set:** 92,000 sentences  
- **Test Set:** 5,000 sentences  

The project includes all steps, from data preprocessing to model evaluation, achieving competitive F1 scores on validation and test datasets.

---

## **Features**
- **Custom Transformer Model**: Built entirely from scratch using PyTorch.  
- **Pre-trained GloVe Embeddings**: Used 100-dimensional GloVe embeddings for word representation.  
- **End-to-End Pipeline**: Includes data preprocessing, training, evaluation, and inference.  
- **Robust Performance**: Achieved F1 scores of 0.658 on the validation set and 0.646 on the test set.  

---

## **Project Workflow**

### **1. Data Preprocessing**
- Converted sentences to lowercase.
- Tokenized text using SpaCy.  
- Applied lemmatization and generated part-of-speech (POS) tags.  
- Used GloVe embeddings to convert lemmatized tokens into 100-dimensional vectors.  
- Applied padding and truncation to standardize sentence lengths.

### **2. Model Architecture**
- **Transformer Encoder:** 4 layers, each with 10 multi-head attention blocks.  
- **Pooling Layers:** Reduced the 3D encoder output using MaxPooling, MinPooling, and AveragePooling.  
- **Feed-Forward Layer:** A fully connected layer produced logits for each sentiment class.  
- **Loss Function:** CrossEntropy Loss.  

### **3. Training**
- Optimized using the Adam optimizer.  
- Trained over 30 epochs with a batch size of 128 and a learning rate of \(7 \times 10^{-5}\).  
- Training conducted on Google Colab.

### **4. Evaluation**
- Achieved F1 scores:  
  - **Validation Set:** 0.658  
  - **Test Set:** 0.646  

---

## **Results**
The model performed robustly, achieving competitive F1 scores on both validation and test sets. However, longer sentences posed challenges in capturing complex dependencies.

| **Phase**     | **F1 Score** |
|----------------|--------------|
| Validation     | 0.658        |
| Test           | 0.646        |

---

## **Future Improvements**
- Use sentence-level attention mechanisms for better handling of long sentences.  
- Experiment with deeper Transformer architectures.  
- Incorporate pre-trained models (e.g., BERT, RoBERTa) for comparison.  

---

## **Dependencies**
- Python 3.7+
- PyTorch
- SpaCy
- NumPy
- Matplotlib
- GloVe Embeddings (downloadable from [GloVe Website](https://nlp.stanford.edu/projects/glove/))

Install all dependencies using `requirements.txt`.

