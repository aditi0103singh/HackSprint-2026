**Data Preprocessing & Visualization Pipeline — HackSprint 2026
📌 Overview**

This repository contains the complete data preprocessing and exploratory data analysis (EDA) pipeline developed for HackSprint 2026.
The project focuses on merging multiple datasets, cleaning data, handling missing values, extracting time-based features, encoding categorical variables, scaling numerical variables, and generating automated visualizations.

**🚀 Key Features**

Merge multi-file datasets into one master dataset

Handle missing and inconsistent values

Convert Date & Time into usable ML features

Encode categorical columns

Scale numeric columns using StandardScaler

Save final cleaned dataset

Auto-generate all major EDA visualizations:

Missing value heatmap with annotations

Distribution plots

Boxplots for outliers

Correlation heatmap

Time series trends

Category value counts

Pairplots

📂 Project Structure
├── merged_master_data.csv        # Combined dataset after merging
├── clean_dataset.csv             # Cleaned dataset after preprocessing
├── preprocessing_visuals.py      # Full EDA + preprocessing code
├── processed_data.csv            # Final dataset after encoding + scaling
└── README.md                     # Documentation

**🛠️ Technologies Used**

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-Learn

**🔧 Setup Instructions**
1️⃣ Install Required Libraries
pip install pandas numpy matplotlib seaborn scikit-learn

2️⃣ Run the Preprocessing & Visualization Script
python preprocessing_visuals.py

3️⃣ Outputs Generated

Cleaned dataset: clean_dataset.csv

Final processed dataset: processed_data.csv

All visualizations displayed automatically

📊 Visualizations Included

**The script automatically generates:**

🔥 1. Missing Values Heatmap (with value labels)

Shows which columns contain missing data and the exact missing counts.

📈 2. Numeric Feature Distributions

Histogram + KDE curve for all numeric features.

📦 3. Outlier Boxplots

Identify extreme values in numeric columns.

🔗 4. Correlation Heatmap

Shows relationships between numeric variables.

🕒 5. Time-Series Trend (if Date exists)

Daily record count visualization.

⏰ 6. Hour, Month, Weekday Trends

Activity patterns extracted from time features.

🏷️ 7. Categorical Countplots

Top frequent values for each categorical feature.

🔍 8. Pairplot

Multivariate analysis for top numerical columns.

**📁 Processed Dataset**

Final dataset saved as:

processed_data.csv


Includes:

Cleaned values

Date/Time features

Label encodings

Scaled numeric features

Perfect for ML modeling, dashboards, or analytics.

**🎯 Purpose (HackSprint 2026)**

This pipeline helps teams quickly:

Standardize and clean messy real-world data

Prepare ML-ready features

Generate insights within minutes

Save time during hackathon development

Designed for fast integration with ML models, dashboards, and RAG systems.

