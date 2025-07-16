# 🚢 Titanic Survival Prediction with TensorFlow

This project uses a neural network model built with TensorFlow/Keras to predict survival on the **Titanic** dataset — a classic problem in binary classification from the [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic).

---

## 🧠 Problem Statement

Given data on passengers (like age, sex, class, etc.), predict whether they survived the sinking of the Titanic. The target is binary:

- `1` = Survived  
- `0` = Did not survive

This project demonstrates how deep learning can be applied to structured tabular data.

---

## 🏗️ Model Architecture

The model is a feedforward neural network (MLP) with the following architecture:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
- Input: Structured features (e.g., Pclass, Sex, Age, Fare, etc.)

- Hidden layers: Capture nonlinear relationships

- Dropout: Prevents overfitting

- Output: Sigmoid activation for binary classification
- 
🛠️ Tools & Technologies
- Python
- TensorFlow / Keras
- Pandas / NumPy
- Scikit-learn for preprocessing and evaluation

🚀 How to Run
Clone the repo:

```bash
Copy code
git clone https://github.com/yourusername/titanic-tf-model.git
cd titanic-tf-model
```
Install dependencies:

```bash
Copy code
pip install -r requirements.txt
```

Run training:
```bash
python train.py
```
📊 Evaluation
Model is evaluated on the validation set using:
- Accuracy

🔮 Possible Improvements
- Feature engineering: Combine SibSp + Parch, extract titles from Name
- Hyperparameter tuning
- Ensemble with classical models (e.g., Random Forest)
- Export to TFLite for deployment

📬 Contact
Open an issue or reach out if you'd like to collaborate or improve the model!
