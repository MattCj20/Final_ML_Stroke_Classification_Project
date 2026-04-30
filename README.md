# Final ML Project - Stroke Prediction Classification

**Student:** Mahdi Mohammadkhani  
**Project Type:** Classification  
**Dataset:** `healthcare-dataset-stroke-cleaned.csv`  
**Target Column:** `stroke`

## Project Overview
This project predicts whether a patient has experienced a stroke using a cleaned healthcare stroke dataset. The project follows the final exam instructions for a classification problem.

## Required Models Trained
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors Classifier
- Support Vector Classifier

## Dataset Split
The dataset was split using stratified sampling:
- Training: 70%
- Validation: 15%
- Test: 15%

## Metrics Used
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Ensemble Models
Two ensemble approaches were created:
1. Soft Voting Ensemble using the top 3 validation ROC-AUC models.
2. Bayesian Weighted Ensemble using normalized validation ROC-AUC scores as weights.

Top three models by validation ROC-AUC:
- Logistic Regression
- Gradient Boosting Classifier
- Random Forest Classifier

## Main Result
The best test F1-Score was achieved by **Soft Voting Ensemble (Top 3)** with:
- Accuracy: 0.7924
- Precision: 0.1288
- Recall: 0.6562
- F1-Score: 0.2154
- ROC-AUC: 0.7901

## Repository Structure
```text
Final_ML_Stroke_Classification_Project/
├── data/
│   └── healthcare-dataset-stroke-cleaned.csv
├── notebooks/
│   └── final_ml_project_stroke_classification.ipynb
├── src/
│   └── train_models.py
├── reports/
│   ├── final_project_report.pdf
│   ├── model_comparison_table.pdf
│   └── student_contribution_summary.pdf
├── presentation/
│   ├── final_ml_project_presentation.pptx
│   └── final_ml_project_presentation.pdf
├── results/
│   └── model_metrics.csv
├── assets/
│   └── charts used in the report and presentation
├── requirements.txt
└── README.md
```

## How to Run
1. Clone or download this repository.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebook:
```bash
jupyter notebook notebooks/final_ml_project_stroke_classification.ipynb
```
Or run the script:
```bash
python src/train_models.py
```

## Submission Note
Canvas file uploads are not permitted for this assignment. Upload this full folder to GitHub and submit the GitHub repository URL in Canvas.
