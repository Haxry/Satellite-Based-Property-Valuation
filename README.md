# Satellite Imagery Based Property Valuation

A multimodal machine learning project that explores how satellite imagery complements traditional housing data for property price prediction.

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Key Results](#key-results)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)

## Overview

This project develops an end-to-end real estate valuation system that combines structured property data with satellite imagery to improve price prediction accuracy.

**Data Sources:**
- **Structured Features**: Living area, lot size, bedrooms, bathrooms, construction quality, geographic coordinates
- **Visual Context**: Satellite imagery providing neighborhood characteristics like green coverage, water proximity, road patterns, and urban layout

**Research Goal**: Determine whether visual context from satellite imagery can enhance traditional property valuation models.

## Problem Statement

Traditional real estate valuation models rely on structured property attributes:
- Living area and lot size
- Number of bedrooms and bathrooms  
- Construction quality ratings
- Geographic coordinates

However, these models often miss important neighborhood-level context such as:
- Proximity to water bodies
- Green space density vs urban concrete
- Road connectivity and accessibility
- Overall urban layout patterns

**Research Question**: Can satellite imagery capture this missing spatial context to improve property valuation accuracy?

## Methodology

Our approach follows a multimodal machine learning pipeline:

1. **Data Collection**: Property coordinates are used to fetch high-resolution satellite imagery via Mapbox API
2. **Feature Extraction**: 
   - Tabular features: Direct use of structured property data
   - Image features: ResNet18 CNN extracts 512-dimensional embeddings from satellite images
3. **Model Development**:
   - Baseline tabular model using traditional regression
   - Image-only model using CNN embeddings
   - Fusion model combining both data modalities
4. **Evaluation**: Performance comparison using RMSE and R² metrics
5. **Interpretability**: Grad-CAM visualization to understand CNN focus areas

## Repository Structure

```
satellite-property-valuation/
├── data/
│   ├── raw/                    # Original train & test datasets
│   ├── processed/              # Cleaned CSVs and aligned subsets
│   └── images/                 # Satellite images (not committed)
├── notebooks/
│   ├── 01_preprocessing.ipynb        # Data cleaning and EDA
│   ├── 02_tabular_model.ipynb        # Baseline regression model
│   ├── 03_image_model.ipynb          # Image-only CNN model
│   ├── 04_fusion_model.ipynb         # Multimodal fusion approach
│   ├── 05_grad_cam.ipynb             # Model interpretability
│   └── 06_evaluation.ipynb           # Performance comparison
├── src/
│   └── data_fetcher.py         # Satellite image acquisition script
├── requirements.txt
├── README.md
└── .gitignore
```

## Models

### 1. Tabular-Only Model
Uses structured housing features with traditional regression algorithms.
- **Strengths**: Strong performance, interpretable, fast training
- **Result**: Best overall performance

### 2. Image-Only Model  
Processes satellite images through ResNet18 to extract spatial features.
- **Approach**: CNN embeddings for property valuation
- **Result**: Captures some signal but produces noisy predictions

### 3. Multimodal Fusion Model
Combines tabular features with image embeddings using early fusion.
- **Approach**: Concatenates structured and visual features
- **Result**: Did not improve upon tabular-only baseline

## Key Results

| Model Type | RMSE Performance | R² Score | Overall Ranking |
|------------|------------------|----------|-----------------|
| Tabular Only | Best | Highest | Winner |
| Image Only | Weak | Negative | Poor |
| Multimodal Fusion | Lower than baseline | Lower than baseline | No improvement |

### Main Findings

**Structured tabular features provide the strongest predictive signal for property valuation.** While satellite imagery captures meaningful neighborhood context (vegetation, water bodies, road patterns), naive fusion with high-dimensional image embeddings introduces noise rather than improving performance.

**Implications**: This result highlights the importance of selective feature fusion strategies in multimodal systems, rather than simple concatenation approaches.

## Model Interpretability

We applied Grad-CAM (Gradient-weighted Class Activation Mapping) to understand what features the CNN focuses on in satellite images.

**Key Observations:**
- **High-value properties**: Model attention on water bodies, green spaces, open layouts, good road access
- **Low-value properties**: Focus on dense rooftops, concrete-heavy regions, industrial textures, poor connectivity

**Validation**: Even though satellite imagery didn't improve prediction accuracy, it does capture semantically meaningful spatial patterns that correlate with property values.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Mapbox API key (for satellite image download)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/satellite-property-valuation.git
cd satellite-property-valuation
```

2. **Create virtual environment**
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API access**
Create a `.env` file in the project root:
```env
MAPBOX_TOKEN=your_mapbox_api_key_here
```

5. **Download satellite images (optional)**
```bash
python src/data_fetcher.py
```
Note: Images are not included in the repository due to size constraints.

## Usage

### Recommended Execution Order

Execute the notebooks in the following sequence:

1. `01_preprocessing.ipynb` - Data cleaning and exploratory analysis
2. `02_tabular_model.ipynb` - Baseline model using structured features
3. `03_image_model.ipynb` - CNN model for satellite image analysis
4. `04_fusion_model.ipynb` - Multimodal approach combining both data types
5. `05_grad_cam.ipynb` - Model interpretability and visualization
6. `06_evaluation.ipynb` - Comprehensive performance comparison

### Output

Final predictions are generated using the best-performing model and saved as:
```
outputs/predictions.csv
```

Format:
```csv
id,predicted_price
1,285000
2,342000
...
```

## Dependencies

Key libraries used in this project:
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost  
- **Deep Learning**: torch, torchvision
- **Computer Vision**: opencv-python
- **API Integration**: requests, python-dotenv
- **Visualization**: matplotlib, seaborn


