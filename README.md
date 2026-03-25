<div align="center">

```
███████╗██████╗  █████╗ ███╗   ███╗    ██████╗ ███████╗████████╗███████╗ ██████╗████████╗ ██████╗ ██████╗ 
██╔════╝██╔══██╗██╔══██╗████╗ ████║    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
███████╗██████╔╝███████║██╔████╔██║    ██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝
╚════██║██╔═══╝ ██╔══██║██║╚██╔╝██║    ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
███████║██║     ██║  ██║██║ ╚═╝ ██║    ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝     ╚═╝    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
```

### 📧 Spam Mail Detection

### *Classifying emails as Spam or Ham using TF-IDF + Logistic Regression*

## 📌 Overview

This project builds a **Natural Language Processing (NLP) based classifier** that detects whether an email is **Spam 🚫** or **Ham ✅** (legitimate). Using the classic SMS Spam Collection dataset with 5,572 messages, the model leverages **TF-IDF vectorization** and **Logistic Regression** to achieve over **95% accuracy** on unseen data.

> 🎯 **Goal:** Filter out spam messages accurately to protect users from unwanted and potentially harmful emails.

---

## 📊 Dataset

| Column | Description |
|---|---|
| `Category` | Label — `spam` or `ham` |
| `Message` | The raw email/SMS text content |

- **Total Records:** 5,572  
- **Spam Messages:** labeled as `0`  
- **Ham Messages:** labeled as `1`  
- **Missing Values:** None ✅

---

## 🔧 Project Workflow

```
Raw Text Data  ──►  Label Encoding  ──►  Train-Test Split  ──►  TF-IDF Vectorization  ──►  Logistic Regression  ──►  Prediction
```

### 1️⃣ Label Encoding
Converted text labels to binary integers:
```
spam  →  0
ham   →  1
```

### 2️⃣ Train-Test Split
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
# Train: 4457 messages | Test: 1115 messages
```

### 3️⃣ TF-IDF Feature Extraction
Transformed raw text into numerical feature vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**:

```python
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features  = feature_extraction.transform(x_test)
```

- Removes common English stop words
- Converts all text to lowercase
- Produces a sparse matrix of shape **(4457 × 7458)**

### 4️⃣ Model — Logistic Regression
```python
model = LogisticRegression()
model.fit(x_train_features, y_train)
```

Logistic Regression works well here because TF-IDF features are high-dimensional and linearly separable — spam messages tend to contain very distinct vocabulary patterns.

### 5️⃣ Predictive System
```python
input_mail = ["Congratulations! You've won a FREE prize. Call now!"]
input_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_features)

# Output: SPAM MAIL 🚫 or HAM MAIL ✅
```

---

## 📈 Results

| Dataset | Accuracy |
|---|---|
| 🟢 Training Set | **96.86%** |
| 🔵 Test Set | **95.34%** |

> The model generalizes well with minimal overfitting — only ~1.5% gap between training and test accuracy. A strong result for a baseline Logistic Regression model!

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | TF-IDF, train-test split, Logistic Regression, accuracy score |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas scikit-learn
```

### Run the Notebook
```bash
# Clone the repo
git clone https://github.com/your-username/spam-mail-detection.git
cd spam-mail-detection

# Open in Jupyter or Google Colab
jupyter notebook Spam_Mail_Detection.ipynb
```

> 💡 You can also open directly in **Google Colab** using the badge at the top.

---

## 📁 Project Structure

```
spam-mail-detection/
│
├── 📓 Spam_Mail_Detection.ipynb    # Main notebook
├── 📄 mail_data.csv                # Dataset (SMS Spam Collection)
└── 📘 README.md                    # You are here
```

---

## 💡 How TF-IDF Works

> TF-IDF assigns a weight to each word based on how often it appears in a message (TF) vs. how rare it is across all messages (IDF). Spam-specific words like **"FREE"**, **"WIN"**, **"PRIZE"** get high scores in spam messages — making them easy for the model to flag.

```
TF-IDF(word) = TF(word) × IDF(word)
             = (word count in doc / total words) × log(total docs / docs with word)
```

---

## 🔮 Future Improvements

- [ ] Try advanced models — Naive Bayes, SVM, Random Forest
- [ ] Add confusion matrix and precision/recall/F1 metrics
- [ ] Use word embeddings (Word2Vec, BERT) for richer representation
- [ ] Build a real-time email classifier web app with Streamlit
- [ ] Train on larger, more diverse email datasets

---

## 🙋 Author

<div align="center">

**Built with ❤️ using Python & scikit-learn**

*If this helped you, please ⭐ star the repo!*

</div>
