# EasyVisa
End-to-end ML system predicting visa approvals with MLOps best practices
## Overview
Machine learning system that predicts visa approval outcomes with 95% F1-score. Built with MLOps best practices including experiment tracking, model versioning, and [coming soon: API deployment, containerization, and CI/CD].

## Problem Statement
The Office of Foreign Labor Certification (OFLC) processes 700,000+ visa applications annually. This ML system helps identify applications with high approval probability, streamlining the review process.

## Dataset
- **Source**: OFLC historical visa application data
- **Size**: 25,480 applications
- **Features**: Education, job experience, region, prevailing wage, company size, etc.
- **Target**: Certified vs Denied

## Approach
1. **Data Preprocessing**: Handled missing values, encoded categorical variables, addressed class imbalance with SMOTE
2. **Model Training**: Evaluated 6 algorithms (Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Bagging)
3. **Model Selection**: Random Forest achieved best performance (95% F1-score, 94% precision, 96% recall)
4. **MLOps**: Implemented MLflow for experiment tracking and model versioning
## Technical Stack
- **ML**: Python, pandas, scikit-learn, XGBoost
- **MLOps**: MLflow
- [Coming soon: FastAPI, Docker, Kubernetes, CI/CD]

## Results
| Model | F1-Score | Precision | Recall |
|-------|----------|-----------|--------|
| Random Forest | 0.95 | 0.94 | 0.96 |
| XGBoost | 0.94 | 0.93 | 0.95 |
| Gradient Boosting | 0.93 | 0.92 | 0.94 |

## Key Features Influencing Visa Approval
1. Prevailing wage (higher wages → higher approval)
2. Region of employment (Northeast and West have higher approval rates)
3. Education level (Master's/PhD → higher approval)
4. Full-time position (vs part-time)

## How to Run

### Prerequisites
Python 3.10+
pip install -r requirements.txt
### Train Model with MLflow Tracking
```bash
python src/train.py
```

### View MLflow UI
```bash
mlflow ui
```
Then open http://localhost:5000

## Project Status
- [x] Data preprocessing and EDA
- [x] Model training and evaluation
- [x] MLflow experiment tracking
- [ ] FastAPI model serving
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Model monitoring dashboard

## Author
Vision | [LinkedIn](your-linkedin) | [GitHub](your-github)

## License
MIT License

