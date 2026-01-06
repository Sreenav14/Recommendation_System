# Movie Recommendation System

A comprehensive movie recommendation system built with TensorFlow Recommenders (TFRS), LightGBM, and FAISS. This project implements multiple recommendation approaches including collaborative filtering, content-based filtering, and hybrid methods.

## 🎯 Overview

This system provides personalized movie recommendations using various machine learning techniques:
- **Retrieval Models**: Efficient candidate generation using TFRS
- **Ranking Models**: Personalized ranking using LightGBM
- **Efficient Search**: FAISS-based similarity search for real-time recommendations

## 📁 Project Structure

```
├── data/
│   ├── raw/           # Raw MovieLens 20M dataset
│   └── processed/     # Cleaned and processed datasets
├── src/               # Source code (see detailed breakdown below)
├── artifact/          # Trained models and artifacts
│   ├── tfrs_retrival_model*    # TFRS retrieval models
│   └── faiss/         # FAISS indices
└── notebooks/         # Jupyter notebooks for analysis
```

## 📂 Source Code Files (`src/`)

### Data Processing Pipeline

#### **`load_and_validate.py`**
**Purpose**: Initial data loading and validation from raw MovieLens 20M dataset
- **Input**: Raw CSV files (`rating.csv`, `movie.csv`)
- **Processing**:
  - Loads ratings and movies data
  - Removes duplicates and null values
  - Converts data types appropriately
- **Output**:
  - `ratings_clean.parquet`: Cleaned user-item ratings
  - `movies_clean.parquet`: Cleaned movie metadata
- **Statistics**: Prints dataset statistics (users, movies, rating ranges)

#### **`make_positives.py`**
**Purpose**: Filters interactions to keep only positive user feedback
- **Input**: `ratings_clean.parquet`
- **Processing**:
  - Filters ratings ≥ 4.0 (positive threshold)
  - Keeps only essential columns: userId, movieId, timestamp
  - Removes duplicate user-movie pairs (keeps latest)
- **Output**: `positive_interactions.parquet`
- **Why**: Recommendation systems focus on positive interactions

#### **`time_split.py`**
**Purpose**: Creates temporal train/validation/test splits per user
- **Input**: `positive_interactions.parquet`
- **Processing**:
  - Sorts interactions by timestamp per user
  - Last interaction → test set
  - Second-to-last interaction → validation set
  - Remaining → training set
- **Output**: `train.parquet`, `val.parquet`, `test.parquet`
- **Why**: Prevents data leakage by respecting temporal order

#### **`make_user_history.py`**
**Purpose**: Creates user interaction history sequences for sequential models
- **Input**: `train.parquet`, `val.parquet`
- **Processing**:
  - For each user, builds interaction sequence up to current point
  - Creates history sequences of length 20 for training
- **Output**: `train_hist.parquet`, `val_hist.parquet`
- **Why**: Sequential models need user history context

#### **`build_genre_features.py`**
**Purpose**: Creates multi-hot encoded genre features for content-based filtering
- **Input**: `movies_clean.parquet`
- **Processing**:
  - Extracts unique genres from movie metadata
  - Creates multi-hot vectors (0/1) for each movie's genres
- **Output**: `movie_genre_features.json`
- **Why**: Enables content-based recommendations using genre information

### TFRS Retrieval Model Training

#### **`train_tfrs.py`** - Basic Collaborative Filtering
**Purpose**: Trains foundational two-tower neural network for user-item similarity
- **Architecture**: User tower + Movie tower with shared embedding space
- **Features**: User ID, Movie ID embeddings
- **Training**: Retrieval task with in-batch negatives
- **Output**: Saved model in `tfrs_retrival_model/` with user/movie embedders
- **Why**: Establishes baseline collaborative filtering performance

#### **`train_tfrs_genre.py`** - Genre-Enhanced Model
**Purpose**: Incorporates content features (genres) into collaborative filtering
- **Architecture**: Two-tower with genre multi-hot features
- **Features**: User embeddings + Movie embeddings + Genre features
- **Training**: Joint optimization of collaborative + content signals
- **Output**: Enhanced retrieval model with content awareness
- **Why**: Improves cold-start and content-based recommendations

#### **`train_tfrs_hardneg.py`** - Hard Negative Sampling
**Purpose**: Uses hard negatives for better training convergence
- **Architecture**: Two-tower with BPR (Bayesian Personalized Ranking) loss
- **Features**: Samples hard negatives from popular items
- **Training**: Explicit negative sampling strategy
- **Output**: More robust embeddings with better ranking
- **Why**: Hard negatives improve model generalization

#### **`train_tfrs_history.py`** - Sequential History Model
**Purpose**: Considers user's interaction history for context-aware recommendations
- **Architecture**: User tower processes history sequence (last 50 items)
- **Features**: User history sequences + target movie
- **Training**: Sequential context modeling
- **Output**: History-aware retrieval model
- **Why**: Captures user preferences evolution over time

### Dataset Creation for Ranking

#### **`make_ranker_dataset.py`** - Basic Ranker Dataset
**Purpose**: Creates training data for pointwise ranking models
- **Input**: Trained TFRS model + FAISS index
- **Processing**:
  - Uses retrieval model to generate candidates (200 per user)
  - Adds popularity ranking features
  - Creates positive/negative labels
- **Output**: `ranker_dataset.parquet` with features and labels
- **Why**: Prepares data for ranking model training

#### **`ranker_make_dataset_history.py`** - History Features Dataset
**Purpose**: Creates ranking dataset with user history patterns
- **Input**: History-aware retrieval model + user interaction sequences
- **Processing**: Adds temporal and sequential features
- **Output**: Enhanced dataset for advanced ranking
- **Why**: Incorporates user behavior patterns

#### **`ranker_make_dataset_history_features.py`** - Advanced Features
**Purpose**: Comprehensive feature engineering for ranking
- **Features**: Time-based, recency, frequency, diversity features
- **Processing**: Complex feature extraction from user history
- **Output**: Feature-rich dataset for high-performance ranking
- **Why**: Advanced features improve ranking accuracy

### LightGBM Ranker Training

#### **`train_ranker_lgbm.py`** - Basic Pointwise Ranker
**Purpose**: Trains LightGBM model for re-ranking retrieved candidates
- **Input**: `ranker_dataset.parquet`
- **Features**: Retrieval scores, popularity ranks
- **Training**: Pointwise ranking with group structure
- **Output**: Trained ranker model
- **Why**: Improves ranking beyond retrieval scores

#### **`train_ranker_lgbm_history_features.py`** - Advanced Ranker
**Purpose**: Trains ranker with comprehensive user history features
- **Features**: 20+ engineered features from user behavior
- **Training**: Handles complex feature interactions
- **Output**: High-accuracy ranking model
- **Why**: Leverages rich user context for better ranking

#### **`train_ranker_lgbm_history_hitonly.py`** - Hit-Optimized Ranker
**Purpose**: Optimizes specifically for hit-rate metrics
- **Training**: Focuses on binary hit/miss classification
- **Output**: Hit-rate optimized model
- **Why**: Specialized for evaluation metrics

### FAISS Index & Embeddings

#### **`export_item_embeddings.py`** - Export Embeddings
**Purpose**: Extracts movie embeddings from trained TFRS models
- **Input**: Trained movie embedder model
- **Processing**: Batched inference on all movies
- **Output**: `items_embeddings.npy`, `movie_ids.json`
- **Why**: Pre-computed embeddings for fast retrieval

#### **`export_item_embeddings_history.py`** - History Embeddings
**Purpose**: Exports embeddings from history-aware models
- **Processing**: Handles sequential context embeddings
- **Output**: History-aware item embeddings
- **Why**: Different embedding space for sequential models

#### **`build_faiss_index.py`** - Basic FAISS Index
**Purpose**: Builds efficient similarity search index
- **Input**: Item embeddings
- **Processing**: Creates FAISS index with inner product similarity
- **Output**: `index.faiss` for fast retrieval
- **Why**: Enables real-time candidate generation

#### **`build_faiss_index_history.py`** - History FAISS Index
**Purpose**: Builds FAISS index for history-aware embeddings
- **Processing**: Optimized for sequential model embeddings
- **Output**: History-specific search index
- **Why**: Separate index for different embedding spaces

### Offline Evaluation

#### **`offline_eval.py`** - Basic Retrieval Evaluation
**Purpose**: Evaluates retrieval model performance
- **Metrics**: Recall@K, NDCG@K vs popularity baseline
- **Processing**: User-by-user evaluation with negative sampling
- **Output**: Performance metrics JSON
- **Why**: Quantifies retrieval quality

#### **`offline_eval_genre.py`** - Genre Model Evaluation
**Purpose**: Evaluates genre-enhanced retrieval performance
- **Metrics**: Same as basic eval + genre-specific analysis
- **Output**: Genre model performance metrics
- **Why**: Assesses content feature contribution

#### **`offline_eval_history_retrieval.py`** - Sequential Retrieval Eval
**Purpose**: Evaluates history-aware retrieval models
- **Processing**: Handles sequential user context
- **Output**: History model evaluation results
- **Why**: Tests sequential recommendation quality

#### **`offline_eval_history_hit.py`** - Hit Rate Evaluation
**Purpose**: Evaluates hit-rate performance for history models
- **Focus**: Binary hit/miss metrics
- **Output**: Hit-rate specific evaluation
- **Why**: Alternative evaluation perspective

#### **`offline_eval_ranker.py`** - Basic Ranker Evaluation
**Purpose**: Evaluates ranking model performance
- **Metrics**: Ranking accuracy, NDCG improvements
- **Output**: Ranker performance metrics
- **Why**: Measures ranking quality gains

#### **`offline_eval_ranker_history_features.py`** - Advanced Ranker Eval
**Purpose**: Evaluates feature-rich ranking models
- **Analysis**: Feature importance and contribution
- **Output**: Comprehensive ranker evaluation
- **Why**: Assesses advanced feature engineering impact

#### **`offline_evaluation_score_hitonly.py`** - Hit-Only Evaluation
**Purpose**: Specialized evaluation for hit-optimized rankers
- **Focus**: Hit-rate optimization results
- **Output**: Hit-specific performance metrics
- **Why**: Measures specialized ranking objectives

### Testing & Utilities

#### **`test_faiss_retrival.py`** - FAISS Testing
**Purpose**: Tests end-to-end retrieval functionality
- **Processing**: Loads models, performs retrieval, shows results
- **Output**: Example recommendations with titles
- **Why**: Validates complete retrieval pipeline

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- TensorFlow 2.8+
- TensorFlow Recommenders
- LightGBM
- FAISS
- pandas, numpy

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow tensorflow-recommenders lightgbm faiss-cpu pandas numpy
```

### Data Preparation
The system uses the MovieLens 20M dataset. Processed data is already available in `data/processed/`.

## 🏗️ Models & Approaches

### 1. TFRS Retrieval Models

#### Basic Collaborative Filtering (`train_tfrs.py`)
- User-item interaction matrix factorization
- Learns embeddings for users and movies
- Simple but effective baseline

#### Genre-Based Model (`train_tfrs_genre.py`)
- Incorporates movie genre features
- Multi-hot encoded genre vectors
- Content-based filtering enhancement

#### Hard Negative Sampling (`train_tfrs_hardneg.py`)
- Uses hard negatives for better training
- Improved ranking accuracy
- Better generalization

#### History-Aware Model (`train_tfrs_history.py`)
- Considers user's viewing history (last 50 movies)
- Sequential recommendation approach
- Most sophisticated retrieval model

### 2. LightGBM Ranking Models

#### Pointwise Ranking (`train_ranker_lgbm.py`)
- Predicts relevance scores for user-item pairs
- Feature engineering from user/movie metadata

#### History Features (`train_ranker_lgbm_history_features.py`)
- Incorporates user history patterns
- Time-based features and recency
- Advanced feature engineering

#### Hit-Only Ranking (`train_ranker_lgbm_history_hitonly.py`)
- Optimized for hit rate metrics
- Simplified training objective

### 3. Efficient Retrieval

#### FAISS Index Building
- `build_faiss_index.py`: Basic item embeddings
- `build_faiss_index_history.py`: History-aware embeddings
- L2 distance similarity search
- Real-time candidate retrieval

## 📊 Evaluation

### Metrics
- **Retrieval**: Recall@K, NDCG@K
- **Ranking**: Hit Rate, Precision@K
- **Coverage**: Item coverage analysis

### Offline Evaluation Scripts
- `offline_eval.py`: Basic retrieval evaluation
- `offline_eval_history_retrieval.py`: History-aware retrieval
- `offline_eval_ranker*.py`: Ranker model evaluation

## 🔧 Usage Examples

### Training a TFRS Model
```bash
cd src
python train_tfrs_history.py
```

### Building FAISS Index
```bash
python build_faiss_index_history.py
```

### Offline Evaluation
```bash
python offline_eval_history_retrieval.py
```

### Training Ranker
```bash
python train_ranker_lgbm_history_features.py
```

## 📈 Results

The system achieves:
- **Recall@200**: ~0.85 for history-aware model
- **NDCG@20**: ~0.12 for ranking models
- **Hit Rate**: ~0.75 for top recommendations

## 🔍 Key Features

- **Multi-Stage Architecture**: Retrieval + Ranking
- **Scalable**: FAISS enables real-time recommendations
- **Feature-Rich**: Multiple data sources and feature engineering
- **Evaluated**: Comprehensive offline evaluation suite
- **Extensible**: Easy to add new models and features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run evaluations to ensure quality
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.