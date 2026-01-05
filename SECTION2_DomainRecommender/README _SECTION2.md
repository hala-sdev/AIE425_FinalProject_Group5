
# AIE425 Final Project - Section 2: Podcast Recommender System with Transcript Analysis

## Group Information
**Group Number:** 5
**Team Members:**
- Hala Soliman (222102480)
- Malak Amgad (221100451)
- Menna Salem Elsayed (221101277)
- Arwa Ahmed Mostafa (221100209)

**Selected Domain:** Podcast Recommendation Engine with Transcript Analysis  
**Submission Date:** Monday, January 5, 2026

---

## Project Overview

This project implements a **complete hybrid podcast recommendation system** that combines content-based filtering, collaborative filtering, and k-NN approaches to provide personalized podcast recommendations. The system addresses key challenges including:

- **Cold-start problem:** Users with minimal rating history
- **Data sparsity:** Large gaps in user-item interaction matrix
- **Content diversity:** Leveraging podcast descriptions, reviews, and categories
- **Scalability:** Efficient processing of 1M+ interactions

### System Architecture
```
Data Collection → Preprocessing → Feature Extraction
                                       ↓
                    Content-Based (TF-IDF + Cosine Similarity)
                                       ↓
                    Collaborative Filtering (Item-based CF)
                                       ↓
                    k-NN Rating Prediction (k=10, k=20)
                                       ↓
                    Hybrid Recommendations (Top-10, Top-20)
```

---

## Repository Structure

```
SECTION2_DomainRecommender/
├── data/
│   ├── preprocessed_data.csv          # Main preprocessed dataset
│   ├── cb_data.csv                     # Content-based recommendations
│   ├── cb_data_rating_pred.csv         # k-NN rating predictions
│   └── README_DATA.md                  # Data description
│
├── code/
│   ├── data_preprocessing.py           # Data loading & cleaning
│   ├── content_based.py                # TF-IDF, user profiles, similarity
│   ├── collaborative.py                # Item-based CF implementation
│   ├── hybrid.py                       # Hybrid recommendation logic
│   └── main.py                         # Main pipeline execution
│
│
├── results/
│
├── requirements.txt                     # Python dependencies
└── README_SECTION2.md                  # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Visual Studio Code (recommended IDE)

### Step 1: Clone Repository
```bash
git clone https://github.com/[your-username]/AIE425_FinalProject_Group5.git
cd AIE425_FinalProject_Group5/SECTION2_DomainRecommender
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
import nltk
nltk.download('stopwords')
```

---

## Dataset Information

### Source
- dataset 1: https://www.kaggle.com/datasets/thoughtvector/podcastreviews/data?select=podcasts.json
- dataset 2: https://www.kaggle.com/datasets/thoughtvector/podcastreviews/versions/28


### Dataset Statistics
- **Total Users:** 100,000+
- **Total Podcasts:** 10,000+
- **Total Ratings:** 1,000,000+
- **Rating Scale:** 1-5 stars
- **Sparsity Level:** ~99.5%
- **Main Features:**
  - User ID (`author_id`)
  - Podcast ID (`itunes_id`)
  - Rating (`rating`)
  - Review Content (`content`)
  - Podcast Description (`description`)
  - Category (`category`)

### Data Preprocessing Steps
1. Text cleaning (lowercase, URL removal, special character removal)
2. Tokenization and stopword removal
3. Stemming using Porter Stemmer
4. Vocabulary filtering (min frequency = 5)
5. Category encoding
6. Aggregation of podcast-level features

---

## Implementation Components

### Part 1: Content-Based Recommendation 

#### 3.1 Feature Extraction
```python
# TF-IDF Vectorization
- Max Features: 5,000
- Min Document Frequency: 5
- N-gram Range: (1, 2)
- Vocabulary: Filtered tokens with frequency ≥ 5
```

**Key Features:**
- **Description TF-IDF:** Podcast description weighted at 1.0
- **Category One-Hot:** 0.6 weight
- **Review TF-IDF:** Aggregated user reviews weighted at 0.3

#### 3.2 User Profile Construction
```python
User_Profile = Σ(rating_i × item_feature_vector_i) / Σ(rating_i)
```

**Cold-Start Handling:**
- Users with <3 ratings → Use top-20 popular podcasts
- Users with 85%+ negative ratings → Fallback to popular items

#### 3.3 Similarity Computation
```python
Cosine_Similarity(user, item) = (user_profile · item_vector) / (||user_profile|| × ||item_vector||)
```

#### 3.4 k-NN Implementation
**k=10 Configuration:**
- Find 10 most similar podcasts to target item
- Weight by cosine similarity
- Predict rating: `Σ(similarity_i × rating_i) / Σ(similarity_i)`

**k=20 Configuration:**
- Extended neighborhood for more robust predictions

### Part 2: Collaborative Filtering & Hybrid 

#### Hybrid Strategy: Weighted Combination
```python
Final_Score = α × Content_Based_Score + (1-α) × k-NN_Predicted_Rating
```
- Tested α values: 0.3, 0.5, 0.7
- Selected α based on validation performance

---

## Running the Code

Run Individual Components
```bash
# Data preprocessing
python data_preprocessing.py

# Content-based recommendations
python content_based.py

# k-NN rating predictions
python knn_prediction.py

# Hybrid recommendations
python hybrid.py
```


### Expected Runtime
- **Data Preprocessing:** ~2-3 minutes
- **TF-IDF Computation:** ~5-7 minutes
- **k-NN Predictions (k=10):** ~15-20 minutes (1000 users)
- **k-NN Predictions (k=20):** ~20-25 minutes (1000 users)
- **Total Pipeline:** ~45-60 minutes


---

## Key Features

### 1. Robust Text Processing
- Porter Stemmer for normalization
- Custom vocabulary filtering (frequency ≥ 5)
- Bigram support for phrase capture

### 2. Multi-Feature Integration
- Description + Category + Reviews
- Weighted feature combination
- L2 normalization for feature scaling

### 3. Cold-Start Mitigation
- Popularity-based initialization
- Threshold-based user profiling
- Fallback to item average ratings

### 4. Scalable Architecture
- Sparse matrix operations
- Efficient k-NN with pre-computed distances
- Batch processing for large user sets

### 5. Comprehensive Evaluation
- Similarity score analysis
- Prediction accuracy metrics
- Method comparison (k=10 vs k=20)
- Cold-start performance breakdown

---

## Dependencies

### Core Libraries
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
scipy==1.11.1
matplotlib==3.7.2
nltk==3.8.1
tqdm==4.65.0
```

### Installation
All dependencies are listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

---

## License

This project is submitted as coursework for AIE425 - Intelligent Recommender Systems, Fall 2025-2026, Galala University.

---

**Last Updated:** January 5, 2026  
