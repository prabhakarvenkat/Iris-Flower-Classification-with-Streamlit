import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# App title
st.set_page_config(page_title="Iris Flower Classifier", layout="wide")
st.title("🌸 Iris Flower Classification with KNN")

# Load Data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    return pd.read_csv(url, names=col_names)

iris_data = load_data()
st.subheader("🔍 Dataset Preview")
st.dataframe(iris_data.head())

# Show description and stats
with st.expander("📊 Statistical Summary"):
    st.write(iris_data.describe())

# Visualizations
st.subheader("📈 Pairplot Visualization")
fig = sns.pairplot(iris_data, hue="class")
st.pyplot(fig)

# Sidebar for KNN parameters
st.sidebar.header("🔧 KNN Settings")
k = st.sidebar.slider("Select number of neighbors (K)", min_value=1, max_value=15, value=3)

# Split data
X = iris_data.drop("class", axis=1)
y = iris_data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Model Performance
st.subheader("📈 Model Performance")
st.write(f"**Accuracy Score:** {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Custom Prediction
st.subheader("🌼 Try a Custom Prediction")
with st.form("prediction_form"):
    sepal_length = st.number_input("Sepal Length", min_value=0.0, value=5.1)
    sepal_width = st.number_input("Sepal Width", min_value=0.0, value=3.5)
    petal_length = st.number_input("Petal Length", min_value=0.0, value=1.4)
    petal_width = st.number_input("Petal Width", min_value=0.0, value=0.2)
    submit = st.form_submit_button("Predict")

    if submit:
        new_data = pd.DataFrame({
            "sepal_length": [sepal_length],
            "sepal_width": [sepal_width],
            "petal_length": [petal_length],
            "petal_width": [petal_width]
        })
        prediction = knn.predict(new_data)
        st.success(f"🌼 Predicted Iris Class: **{prediction[0]}**")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")
