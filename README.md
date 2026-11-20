# Developer Compensation Analysis: EDA, Clustering & Classification

A machine learning project that analyses developer survey data to (1) explore the dataset, (2) group developers into clusters, and (3) predict whether a developer is on **high or low compensation**.

Built for **COMP40721** coursework. Author: Bhagyesh Chaudhari

All code is in one notebook: `comp40721_coursework.ipynb`.

---

## Objectives

1. **Exploratory data analysis (EDA)** to understand the dataset and its features.
2. **Cluster analysis** to understand the characteristics of developers in each cluster.
3. **Classification** to predict whether a developer is high income from their profile.

Compensation is labelled using annual salary in USD (the survey's `ConvertedCompYearly` column, renamed `Salary`):

| Label | Definition |
| --- | --- |
| Low | Annual compensation < $50,000 |
| High | Annual compensation ≥ $50,000 |

## Dataset

The notebook uses `survey_results_public.csv`, a public developer survey (89,184 rows × 84 columns in the raw file). The data is **not included in this repo**; download it from the survey's official source and place it next to the notebook.

After removing respondents with no salary, **47,989 records** remain. These ten columns were selected:

`Age`, `Employment`, `RemoteWork`, `EdLevel`, `YearsCodePro`, `WorkExp`, `DevType`, `Country`, `ProfessionalTech`, `Salary`

## Workflow

| Stage | What was done |
| --- | --- |
| **Cleaning** | Dropped rows with no salary; removed "Prefer not to say" ages; checked for nulls and empty cells |
| **Feature engineering** | Mapped age ranges to numeric values; converted `YearsCodePro` text ("Less than 1 year", "More than 50 years") to numbers; encoded education level; created the `CompensationSalary` target (Low/High) |
| **Missing values** | Filled with the column mode; also tried `SimpleImputer` (mean) and `KNNImputer` |
| **EDA** | Box plots, histograms, count plots, scatter plots and a pair plot for age, experience, employment, developer type and education |
| **Encoding & scaling** | `LabelEncoder` for categorical columns; `StandardScaler` / `MinMaxScaler` for model inputs |
| **Clustering** | K-Means and Agglomerative (hierarchical) clustering with 2 clusters, compared with silhouette score, visualised with PCA and a dendrogram |
| **Classification** | k-NN (k = 1–25), Decision Tree, Logistic Regression and Random Forest |
| **Evaluation** | Accuracy, error rate, confusion matrix, sensitivity/specificity, classification report, F1 score and ROC curves |
| **Tuning** | Decision tree depth sweep (train vs validation curve) and `GridSearchCV` (10-fold) over depth and criterion |

## Results

**Clustering** (silhouette score, 2 clusters):

| Method | Silhouette |
| --- | --- |
| K-Means | 0.231 |
| Agglomerative | 0.195 |

**Classification** (test-set accuracy):

| Model | Accuracy |
| --- | --- |
| Decision Tree (tuned with GridSearchCV, `max_depth=10`, gini) | ~0.832 |
| Random Forest | ~0.829 |
| Decision Tree (`max_depth=2`) | ~0.778 |
| Logistic Regression | ~0.740 |

The Random Forest classification report gives weighted precision of 0.82, recall of 0.83 and F1 of 0.83.

> Models were evaluated on slightly different train/test splits (70/30 and 80/20), so compare these numbers with care.

## Tech Stack

Python · pandas · NumPy · scikit-learn · SciPy · seaborn · matplotlib

## Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn scipy seaborn matplotlib jupyter
   ```
3. Download `survey_results_public.csv` and place it in the project folder.
4. Open the notebook and **update the data path**. It was written for Google Colab and loads the data from Google Drive:
   ```python
   # Replace the Colab/Drive lines
   # from google.colab import drive
   # drive.mount('/content/drive')
   survey = pd.read_csv("survey_results_public.csv")
   ```
5. Run all cells:
   ```bash
   jupyter notebook comp40721_coursework.ipynb
   ```

## Project Structure

```
.
├── comp40721_coursework.ipynb   # Full analysis: EDA, clustering, classification
├── survey_results_public.csv    # Dataset (download separately, not tracked)
└── README.md
```

## Acknowledgements

Dataset: public developer survey data. Coursework for COMP40721.
