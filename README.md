# AIE425 Final Project: Dimensionality Reduction & Podcast Recommender System

## Group Information
**Group Number:** 5  
**Team Members:**
- Hala Soliman (222102480)
- Malak Amgad (221100451)
- Menna Salem Elsayed (221101277)
- Arwa Ahmed Mostafa (221100209)

**Course:** AIE425 Intelligent Recommender Systems  
**Instructor:** Dr. Samy Ghoneim  
**Institution:** Galala University  
**Submission Date:** Monday, January 5, 2026

---

## Project Overview

This project implements comprehensive recommender systems using both **dimensionality reduction techniques** (PCA & SVD) on MovieLens data and a **complete hybrid podcast recommendation engine**. The implementation addresses real-world challenges including data sparsity, cold-start problems, and scalability.

### Key Objectives
1. Compare PCA (Mean-Filling & MLE) and SVD for collaborative filtering
2. Build a hybrid podcast recommender combining content-based and collaborative filtering
3. Evaluate performance using multiple metrics and demonstrate practical applications

---

## Repository Structure

```
AIE425_FinalProject_Group5/
│
├── SECTION1_DimensionalityReduction/
│   ├── data/
│   │   ├── ratings.csv                # MovieLens ratings
│   │   └── movies.xlsx                # Movie metadata
│   ├── code/
│   │   ├── pca_mean_filling.py
│   │   ├── pca_mle.py
│   │   └── svd_analysis.py
│   ├── results/
│   │   ├── plots/                     # Scree plots, variance curves
│   │   └── tables/                    # Prediction results
│   └── README_SECTION1.md
│
├── SECTION2_DomainRecommender/
│   ├── data/
│   │   ├── preprocessed_data.csv      # Podcast dataset
│   │   ├── cb_data.csv                # Content-based recommendations
│   │   └── cb_data_rating_pred.csv    # k-NN predictions
│   ├── code/
│   │   ├── data_preprocessing.py
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   └── hybrid.py
│   ├── results/
│   └── README_SECTION2.md
│
├── Final_Report.pdf
├── requirements.txt
└── README.md                          # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Clone repository
git clone https://github.com/[username]/AIE425_FinalProject_Group5.git
cd AIE425_FinalProject_Group5

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for Section 2)
python -c "import nltk; nltk.download('stopwords')"
```

---

## SECTION 1: Dimensionality Reduction (10%)

### Dataset: MovieLens 20M
- **Users:** 138,493 | **Movies:** 27,278 | **Ratings:** 20M+
- **Target Items:** Movie ID 2 (Jumanji), Movie ID 8860 (LOTR)

### Methods Implemented

#### Part 1: PCA with Mean-Filling
- Matrix: 3,000 users × 1,500 items
- Covariance-based peer selection (top 5/10)
- Item-mean imputation for missing values

#### Part 2: PCA with Maximum Likelihood Estimation
- Matrix: 15,000 users × 800 items
- Unbiased covariance using only co-rated entries
- 90% variance retention criterion

**Results:**
| Item | Peers | MAE | RMSE | Variance |
|------|-------|-----|------|----------|
| I1   | 5     | 0.67 | 0.85 | 90.2%    |
| I1   | 10    | 0.67 | 0.85 | 94.8%    |
| I2   | 5     | 0.59 | 0.74 | 91.5%    |
| I2   | 10    | 0.59 | 0.74 | 95.1%    |

#### Part 3: Singular Value Decomposition
- Matrix: 16,000 users × 1,100 items
- Full SVD with truncated approximations (k=5,20,50,100)
- Latent factor interpretation and cold-start analysis

**Results:**
| k   | MAE  | RMSE | Variance | Compression |
|-----|------|------|----------|-------------|
| 5   | 0.009 | 0.077 | 99.94%   | 99.5%       |
| 20  | 0.008 | 0.066 | 99.96%   | 98.2%       |
| 100 | 0.007 | 0.041 | 99.98%   | 90.9%       |

### Key Findings
- **SVD outperforms PCA:** 88% better MAE with k=100
- **Efficiency:** k=20 captures 99.96% variance with minimal error
- **MLE advantage:** 10% more accurate than mean-filling

### Running Section 1
```bash
cd SECTION1_DimensionalityReduction/code
python pca_mean_filling.py
python pca_mle.py
python svd_analysis.py
```

---

## SECTION 2: Podcast Recommender System (20%)

### Domain: Podcast Recommendation with Transcript Analysis

### Dataset Statistics
- **Source:** Kaggle Podcast Reviews (2020-2023)
- **Users:** 100,000+ | **Podcasts:** 10,000+ | **Ratings:** 1M+
- **Sparsity:** 99.5% | **Scale:** 1-5 stars

### System Architecture
```
Data → Preprocessing → TF-IDF Feature Extraction
                           ↓
      Content-Based (Cosine Similarity) + k-NN (k=10,20)
                           ↓
              Hybrid Recommendations (Top-10/20)
```

### Implementation

#### Part 1: Content-Based Filtering
**Features:**
- TF-IDF vectors (5,000 features, bigrams)
- Category encoding (0.6 weight)
- Review aggregation (0.3 weight)

**User Profiles:**
```python
User_Profile = Σ(rating_i × item_vector_i) / Σ(rating_i)
```

**Cold-Start Handling:**
- <3 ratings → Popular items fallback
- 85%+ negative ratings → Item average

#### Part 2: k-NN Rating Prediction
- Item-based CF with cosine similarity
- Weighted prediction: `Σ(sim_i × rating_i) / Σ(sim_i)`
- Tested k=10 and k=20

#### Part 3: Hybrid Strategy
```python
Final_Score = α × CB_Score + (1-α) × k-NN_Rating
```
Tested α ∈ {0.3, 0.5, 0.7}

### Performance Metrics
- **Mean Similarity (Top-10):** 0.78
- **Mean Similarity (Top-20):** 0.72
- **Drop:** 8.5%
- **Cold-Start Coverage:** 89% (1-2 ratings)

### Running Section 2
```bash
cd SECTION2_DomainRecommender/code
python data_preprocessing.py
python content_based.py
python collaborative.py
python hybrid.py
```

---

## Dependencies

```
numpy>=1.24.3
pandas>=2.0.3
scikit-learn>=1.3.0
scipy>=1.11.1
matplotlib>=3.7.2
nltk>=3.8.1
tqdm>=4.65.0
jupyter>=1.0.0
openpyxl>=3.0.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Key Results Summary

### SECTION 1: Dimensionality Reduction
✅ SVD with k=20 achieves 99.96% variance retention  
✅ MLE-PCA reduces error by 10% vs mean-filling  
✅ Just 5 components capture 99.94% of data variance  

### SECTION 2: Podcast Recommender
✅ Content-based achieves 0.78 mean similarity (Top-10)  
✅ k-NN (k=20) successfully predicts 71.8% of ratings  
✅ Cold-start users handled via popularity fallback  

---

## Datasets

### Section 1
- **MovieLens 20M:** https://grouplens.org/datasets/movielens/20m/

### Section 2
- **Podcast Reviews Dataset 1:** https://www.kaggle.com/datasets/thoughtvector/podcastreviews/data
- **Podcast Reviews Dataset 2:** https://www.kaggle.com/datasets/thoughtvector/podcastreviews/versions/28

---


### References
- MovieLens dataset: GroupLens Research
- Kaggle
- scikit-learn: https://scikit-learn.org
- NLTK: https://www.nltk.org
