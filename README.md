# Iris Flower Classification with Streamlit 🌸  
#### Video Demo: [Watch the demo](<https://youtu.be/KTdDu_Qms-s>)

---

#### Description:
This project is an interactive web application that performs real-time classification of iris flowers using machine learning. It was developed as the final project for CS50x, with the aim of combining data science, machine learning, and web-based UI design into one cohesive application. The project uses the Iris dataset from the UCI Machine Learning Repository, which contains measurements for sepal length, sepal width, petal length, and petal width across three flower species: Setosa, Versicolor, and Virginica. The machine learning model is built using the K-Nearest Neighbors (KNN) algorithm from Scikit-learn. The dataset is split into training and test sets, and the model is trained on 70% of the data. The trained model can then predict the species of a flower based on input measurements provided by the user. Model performance is evaluated using metrics like accuracy and a detailed classification report. The entire application is deployed through Streamlit, a Python library that allows rapid prototyping of web apps. The app features data visualization using Seaborn and Matplotlib, enabling users to explore pairplots and distribution insights. Users can interact with the model directly by entering flower measurements via the UI and instantly seeing the prediction results. This project serves as a practical example of machine learning deployment in an accessible and user-friendly environment. It also demonstrates the power of open-source tools in making AI and ML more approachable for beginners.


---

## 📁 Features:
- 📊 **Data Visualization** using Seaborn (Pairplot)
- ⚙️ **Model Training** using Scikit-learn’s KNN
- 🧪 **Train/Test Split** for evaluation
- 📈 **Performance Metrics**: Accuracy and classification report
- 🖱️ **Interactive UI**: Input flower measurements and predict species in real-time
- 🚀 **Live Prediction** for user-defined input values

---

## 🧠 Tech Stack:
- **Python**
- **Streamlit**
- **Pandas**
- **Seaborn & Matplotlib**
- **Scikit-learn**

---

## 🔢 Dataset:
- Source: [UCI Machine Learning Repository – Iris Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
- Attributes:
  - `sepal_length`
  - `sepal_width`
  - `petal_length`
  - `petal_width`
  - `class`

---

## ▶️ How to Run Locally:
1. **Clone the repository**
```bash
git clone https://github.com/prabhakarvenkat/Iris-Flower-Classification-with-Streamlit.git
cd Iris-Flower-Classification-with-Streamlit
