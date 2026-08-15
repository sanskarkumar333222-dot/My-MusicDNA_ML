# 🎧 Spotify MusicDNA: Machine Learning Matcher & Hit Recommender

An end-to-end Machine Learning project that profiles an artist's career attributes (era, genre, solo independence vs. collaborative streams, language) to match users to their top 3 superstar artist twins and generate personalized top 5 song recommendations.

---

## 📁 Project Architecture & Workflow
1. **Data Cleaning & Preprocessing**: Handled missing values, standardized string categories, and removed duplicates.
2. **Feature Engineering**:
   - `Career_Age`: $2024 - \text{Debut Year}$
   - `Streams_Per_Year`: Average stream generation velocity.
   - `Feature_Reliance_Ratio`: Feature streams share of total stream volume.
   - `Collab_To_Solo_Ratio`: Collaborative stream dominance.
   - `Superstar_Tier`: Categorical binning of global streaming tiers.
3. **Exploratory Data Analysis (EDA)**: Analyzed solo vs. collab dynamics, genre performance, and generated correlation heatmaps.
4. **Machine Learning Model**:
   - Multi-dimensional normalized latent space created via `StandardScaler` + `OneHotEncoder`.
   - **Cosine Similarity & K-Nearest Neighbors (KNN)** distance metrics.
5. **Interactive UI**:
   - Python pipeline (`main.py`)
   - Streamlit Web App (`app.py`) with Spotify Dark Mode aesthetics and direct Spotify track links.

---

## 🚀 How to Run in VS Code

### 1. Install Dependencies
Open the VS Code Terminal (`Ctrl + \``) and run:
```bash
pip install -r requirements.txt
```

### 2. Run the Main Python Pipeline
```bash
python main.py
```
*Outputs:* Prints data cleaning results, answers research questions, outputs the ML simulation, and saves all plots into `output_plots/eda_dashboard.png`.

### 3. Launch the Interactive Web App (Optional / For Demo)
```bash
streamlit run app.py
```
This will automatically launch the interactive Spotify-themed web interface in your browser.
