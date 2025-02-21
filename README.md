# 🛡️ Credit Card Fraud Detection Using Machine Learning 💳  

![Fraud Detection](https://miro.medium.com/v2/resize:fit:1200/1*5jG9imnH38A6Szj8HZb_Uw.png)  

## 🔍 Abstract

With the rise of **online banking and digital transactions**, credit and debit card fraud has become a major concern. Fraudsters exploit loopholes in security systems to carry out unauthorized transactions, leading to **financial losses** for both consumers and businesses. 

This project applies **Machine Learning techniques** to detect fraudulent transactions and **prevent financial fraud**. The **Isolation Forest algorithm** is used to classify transactions as **legitimate or fraudulent**, providing an effective and scalable solution to enhance security.

**Key Challenges:**
- Fraudsters use **advanced techniques** to bypass security.
- **Highly imbalanced dataset** (fraud transactions make up only 0.172% of total transactions).
- **Real-time fraud detection** is critical to minimize losses.

Using **supervised machine learning**, this model detects fraudulent activities with high accuracy, reducing risks and safeguarding users.

---

## 📊 Dataset Overview  

This project uses a dataset of **credit card transactions made by European cardholders** in **September 2013**.

📌 **Dataset Key Facts:**
- **Total Transactions:** 284,807  
- **Fraud Cases:** 492 (~0.172%)  
- **Time Period:** 2 Days  
- **Data Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

### **Feature Breakdown**
| Feature | Description |
|---------|------------|
| **Time** | Seconds elapsed between each transaction and the first transaction |
| **Amount** | Transaction amount in Euros |
| **V1 - V28** | Principal Components obtained via **PCA** |
| **Class** | Target variable (**0 = Genuine**, **1 = Fraudulent**) |

**🔍 Data Visualization**
![Fraud Distribution](https://www.mdpi.com/applsci/applsci-11-06447/article_deploy/html/images/applsci-11-06447-g001.png)

As seen in the **Fraud Distribution graph**, fraudulent transactions are **extremely rare**, making fraud detection a challenging problem.

---

## 🚨 Problem Statement  

The goal of this project is to **predict fraudulent transactions using machine learning algorithms**. 

With millions of transactions occurring daily, **manual fraud detection** is nearly impossible. An AI-powered fraud detection system can:  
✅ **Automatically identify suspicious transactions** in real-time.  
✅ **Prevent financial losses** for individuals and institutions.  
✅ **Detect patterns and anomalies** using historical data.

To tackle the **imbalance in data**, we employ **oversampling (SMOTE), undersampling, and feature engineering** techniques.

---

## 📂 Data Source  

**📌 Dataset Download:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

### **How to Run the Project**  

1️⃣ **Clone the Repository**
```sh
git clone https://github.com/FazilMammadli/FraudShieldAI.git
cd FraudShieldAI
```

2️⃣ **Download the Dataset from Kaggle**  
- Extract `creditcard.csv` inside the project folder.

3️⃣ **Create and Activate a Virtual Environment**
```sh
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

4️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```

5️⃣ **Run Jupyter Notebook**
```sh
jupyter notebook
```
Open and execute `Fraud_Detection.ipynb` to explore the dataset and train models.

---

## 🛠️ Prerequisites  

📌 Ensure you have the following **software & libraries** installed:  
- **Python 3.x**  
- **Anaconda** (Includes Jupyter Notebook)  
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`  

---

## 📉 Machine Learning Approach  

We tested multiple **ML models** to classify transactions into **fraudulent (1) or genuine (0)**.

### 🔬 **Feature Engineering**
✅ **Scaled transaction amounts using MinMaxScaler**  
✅ **Created new features for anomaly detection**  
✅ **Used SMOTE to handle class imbalance**  

### 🏆 **Model Comparison**
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|----------|------------|
| Logistic Regression | 94% | 87% | 82% | 84% |
| Random Forest | 97% | 93% | 89% | 91% |
| XGBoost | 99% | 97% | 95% | 96% |

📌 **Confusion Matrix for XGBoost**
![Confusion Matrix](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*2JcWW1If-wexWT3U2RAJuw.png)

The **XGBoost model** performed best with an **accuracy of 99%**, detecting fraudulent transactions with high precision.

---

## 🔑 Key Findings  

✅ **Fraudulent transactions exhibit distinct patterns compared to genuine ones.**  
✅ **Machine learning can significantly improve fraud detection accuracy.**  
✅ **Data balancing techniques (SMOTE) improve recall for minority classes.**  

---

## 🔥 Business Applications  

💡 **Fraud Prevention for Banks & FinTech**  
- Detect **fraudulent transactions in real-time**.  
- Reduce **financial losses and chargebacks**.  

💡 **Secure Online Payments**  
- Enhance security in **e-commerce and digital wallets**.  
- Identify and block **suspicious transactions** before processing.  

💡 **AI-Powered Security Systems**  
- Use **deep learning and NLP** to detect **fraudulent patterns** in transaction history.  
- Implement **automated fraud monitoring** for large-scale financial systems.  

---

## 🎯 Conclusion  

This project successfully applies **AI-driven fraud detection** to **enhance financial security**.

🚀 **Key Takeaways:**  
✅ **XGBoost achieved the highest accuracy (99%)**.  
✅ **Machine learning models can efficiently detect fraud in large datasets**.  
✅ **Financial institutions can use AI to prevent fraud and protect customers**.  

---

## ⭐ References  

📌 [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
📌 [Fraud Detection Research - Siddhardhan](https://www.youtube.com/watch?v=NCgjcHLFNDg&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6&index=10)  

---

## ⭐ Support & Contributions  

If this project was useful, **drop a ⭐ on GitHub!**  
For feedback or contributions, **submit a pull request!** 🚀  

