# Breast Cancer Classifier

This repository contains a simple **Streamlit** web application that uses machine learning
models to classify breast tumors as **malignant** or **benign** based on cell nucleus
measurements. It is intended for educational purposes and demonstrates data
preprocessing, model training, and deployment with an interactive UI.

---

## 🚀 Project Overview

The application:

1. Loads the Breast Cancer Wisconsin (Diagnostic) dataset from `data.csv`.
2. Cleans and preprocesses the data by dropping irrelevant columns and scaling
   features.
3. Trains four classification models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest (selected as the best model)
4. Displays model performance metrics in a dashboard.
5. Accepts user input for ten nucleus feature means and predicts tumor type.

The front‑end allows users to enter measurements, run a classification, and
view the prediction along with confidence.

> **Disclaimer:** This tool is for educational purposes only and is not a
> substitute for professional medical advice.

---

## 📦 Requirements

- Python 3.8 or later
- Packages listed below (install via `pip`):
  - streamlit
  - pandas
  - numpy
  - scikit-learn

You can install the dependencies using:

```bash
pip install streamlit pandas numpy scikit-learn
```

Alternatively, create a `requirements.txt` with the above packages and run
`pip install -r requirements.txt`.

---

## 📁 Data

The application expects a CSV file named `data.csv` located in the root of the
workspace (i.e. alongside `new.py`). The code loads the dataset using a
relative path so the project can be deployed without modifying the source. The file should be the [Breast Cancer Wisconsin (Diagnostic)
dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
obtained from the UCI Machine Learning Repository.

The code automatically drops the `id` column and any unnamed trailing column
(e.g. `Unnamed: 32`). It uses only the ten mean feature columns (those containing
`_mean` in their names) for prediction.

---

## ⚙️ Running the Application

### 🔧 Deployment
You can deploy the app to any host that supports Python web processes. Two
common options are:

1. **Streamlit Community Cloud** (formerly Streamlit Sharing):
   - Push your repo (including `data.csv` and `requirements.txt`) to GitHub.
   - Add an app entry pointing to `new.py`.
   - The service installs the dependencies automatically.

2. **Heroku / other PaaS**:
   - Add a `Procfile` with:
     ```
     web: streamlit run new.py --server.port=$PORT
     ```
   - Commit `requirements.txt` and `Procfile`.
   - Deploy using the platform's git workflow.

In both cases the dataset will be read from the repo root and no further
configuration is required.

## ⚙️ Running the Application

1. Ensure `data.csv` is present in the project root (same folder as
   `new.py`).
2. Install the required Python packages (see **Requirements** section).
3. From the project directory, launch Streamlit:

   ```bash
   streamlit run new.py
   ```

4. A browser window will open showing the Breast Cancer Classifier UI. Enter
   the feature values and click **Run Classification** to see results.

---

## 🧠 How It Works

- The data is split into training and test sets (80/20) with a fixed random
  seed for reproducibility.
- Features are scaled using `StandardScaler`.
- Each model is trained and evaluated; accuracy percentages are shown on the
  dashboard.
- The Random Forest model is chosen as the default for making predictions.

User inputs are scaled with the same `StandardScaler` instance before
prediction.

---

## 📌 Notes

- Default input values in the UI are set to `0.0`. You may replace them with
  typical dataset averages or values for faster testing.
- The application caches model training to avoid re-running heavy computations
  on every interaction.

---

## 📄 License

This project is provided under the MIT License. Feel free to adapt and reuse
it for learning purposes.

---

## 📬 Contact

For questions or suggestions, please reach out to the repository owner.
