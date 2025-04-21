# K-means Clustering Analysis with Different Distance Metrics

This project implements and analyzes K-means clustering using three different distance metrics:
- Euclidean Distance
- Cosine Similarity
- Jaccard Similarity

## Project Structure
```
kmeans_analysis/
├── data/               # Dataset files
│   ├── data.csv       # Feature data
│   └── label.csv      # True labels
├── figures/           # Generated visualization plots
├── kmeans_analysis.py # Main implementation
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Requirements
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib

## Installation
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
python3 kmeans_analysis.py
```

## Analysis Components
1. **SSE Comparison**: Compares Sum of Squared Errors across different distance metrics
2. **Accuracy Analysis**: Evaluates clustering accuracy using majority vote labeling
3. **Convergence Study**: Analyzes iterations and time required for convergence
4. **Termination Conditions**: Examines different stopping criteria effects

## Visualizations
The code generates three visualization plots:
1. `accuracy_comparison.png`: Clustering accuracy comparison
2. `iterations_comparison.png`: Number of iterations needed for convergence
3. `convergence_time.png`: Time taken for convergence

## Results
The analysis shows that:
- Cosine similarity performs best for accuracy
- Euclidean distance shows comparable performance
- Jaccard similarity is not well-suited for this dataset

## Notes
- SSE values are not directly comparable between different distance metrics
- Empty cluster warnings for Jaccard similarity indicate potential issues with this metric for the given dataset
