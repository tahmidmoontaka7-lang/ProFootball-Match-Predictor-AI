# ⚽ ProFootball Match Predictor AI

An advanced, data-driven **Machine Learning Web Application** that predicts the probabilities of international football match outcomes (Home Win, Away Win, or Draw). The system utilizes historical global football fixture data spanning from 1872 to 2026 to output real-time match analytics.

---

## 🚀 Key Features
- **Advanced Predictive AI:** Driven by Scikit-Learn's Random Forest Classifier optimized for multi-class outcome forecasting.
- **On-the-Fly Cloud Training:** Engineered with Streamlit caching (`@st.cache_resource`) to instantly train the model directly on the server, eliminating the need to upload heavy serialized `.pkl` files.
- **Dynamic Probabilistic Dashboard:** Seamlessly processes historical head-to-head match metrics and generates clean visual probability trackers for users.
- **Big Data Infrastructure:** Trained on a comprehensive Kaggle dataset detailing tens of thousands of official international football matches.

---

## 🛠 Tech Stack
- **Core Architecture:** Python 🐍
- **Machine Learning Engine:** Scikit-Learn (Random Forest)
- **Data Engineering:** NumPy, Pandas
- **Production UI Framework:** Streamlit

---

## 📂 Project Architecture
```text
├── app.py / ui_app.py                # Core Streamlit web application & cached training engine
├── results.csv                       # Historical global football results dataset (1872 - 2026)
├── requirements.txt                  # Deployment environment dependency list
└── README.md                         # Comprehensive project documentation
```

---

## 💻 Local Installation & Setup

1. Clone this repository to your local directory:
```bash
git clone https://github.com
cd ProFootball-Match-Predictor-AI
```

2. Install all necessary framework dependencies:
```bash
pip install -r requirements.txt
```

3. Boot up the local production Streamlit server:
```bash
streamlit run ui_app.py
```
