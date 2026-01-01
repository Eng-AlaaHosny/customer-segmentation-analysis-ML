# Customer Segmentation Analysis Using Machine Learning

**Course:** SE 3007 - Introduction to Machine Learning  
**Student:** Alaa Hsony Saber  
**Section:** ŞB: 1 ÖRGÜN  
**Instructor:** Doktor Öğretim Üyesi Selim Yılmaz  


---

## Table of Contents
- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model Architecture and Methodology](#model-architecture-and-methodology)
- [Installation and Setup](#installation-and-setup)
- [Execution Instructions](#execution-instructions)
- [Results and Evaluation](#results-and-evaluation)
- [Project Structure](#project-structure)
- [References](#references)

---

## Problem Description

### Business Challenge
Modern retail businesses face a critical challenge: **one-size-fits-all marketing is ineffective and costly**. Companies need to:
- Identify high-value customers for retention strategies
- Personalize offers based on customer purchasing behavior
- Optimize marketing spend across different customer segments
- Identify at-risk customers before they churn

### Research Question
**How can we automatically group customers based on their purchasing behavior to enable targeted marketing strategies?**

### Expected Business Impact
- **Increase customer lifetime value by 20-30%**
- **Reduce marketing costs** through targeted campaigns
- **Improve customer retention** in high-value segments
- **Identify at-risk customers** proactively

---

## Dataset

### UCI Online Retail Dataset

**Source:** UCI Machine Learning Repository  
**Description:** Real transactional data from a UK-based online retailer

#### Dataset Characteristics
- **Time Period:** December 2010 - December 2011
- **Total Transactions:** 541,909 records
- **Total Customers:** 4,372 unique customers
- **Geographic Scope:** Primarily UK-based transactions

#### Features (8 columns)
| Feature | Description |
|---------|-------------|
| `InvoiceNo` | Unique invoice number for each transaction |
| `StockCode` | Product/item code |
| `Description` | Product name/description |
| `Quantity` | Number of items purchased |
| `InvoiceDate` | Date and time of transaction |
| `UnitPrice` | Price per unit in GBP (£) |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer's country |

#### Why This Dataset is Ideal
✅ **Real-world e-commerce data** (not synthetic)  
✅ **Perfect for RFM analysis** (Recency, Frequency, Monetary)  
✅ **Sufficient size** for robust clustering (4K+ customers)  
✅ **Complete transaction details** for comprehensive feature engineering  
✅ **Industry-standard benchmark** for segmentation research  
⚠️ **Realistic challenges:** Missing values, cancellations, outliers

---

## Preprocessing Pipeline

### Phase 1: Data Cleaning

#### Step 1: Remove Invalid Records
```python
# Remove records with missing CustomerID
data = data.dropna(subset=['CustomerID'])

# Remove cancelled transactions (InvoiceNo starting with 'C')
data = data[~data['InvoiceNo'].str.startswith('C')]

# Remove invalid quantities and prices
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
```

**Impact:** Removed **156,366 rows** (invalid transactions)

#### Step 2: Data Type Corrections
- Convert `InvoiceDate` to datetime format
- Convert `CustomerID` to integer
- Ensure numeric types for `Quantity` and `UnitPrice`

#### Step 3: Create Transaction Value
```python
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
```

### Phase 2: Outlier Detection and Removal

**Method:** Isolation Forest Algorithm

**Features Analyzed for Outliers:**
- Recency (153 outliers)
- Frequency (266 outliers)
- Monetary (409 outliers)
- Avg_Transaction_Value (260 outliers)
- Total_Items (389 outliers)
- Unique_Products (303 outliers)

**Total Outliers Removed:** 215 customers (5.0% of dataset)  
**Final Dataset:** 4,075 customers

**Visualization:** See `06_outlier_removal_impact.png`

---

## Feature Engineering

### Reference Date
**December 10, 2011** - Used as the reference point for recency calculations

### Feature Categories

#### 1. Base RFM Features (3 features)
| Feature | Description | Formula |
|---------|-------------|---------|
| `Recency` | Days since last purchase | Reference Date - Last Invoice Date |
| `Frequency` | Number of transactions | Count of unique invoices |
| `Monetary` | Total spending ($) | Sum of all transaction values |

#### 2. Behavioral Features (7 features)
| Feature | Description | Formula |
|---------|-------------|---------|
| `Avg_Transaction_Value` | Average spend per order | Monetary / Frequency |
| `Tenure` | Days since first purchase | Reference Date - First Invoice Date |
| `Total_Items` | Total quantity purchased | Sum of all quantities |
| `Unique_Products` | Distinct product variety | Count of unique StockCodes |
| `Frequency_Rate` | Transaction frequency | Frequency / Tenure (transactions per day) |
| `Monetary_Rate` | Spending rate | Monetary / Tenure (spending per day) |
| `Items_Per_Transaction` | Basket size | Total_Items / Frequency |

#### 3. Log-Transformed Features (4 features)
Applied to reduce skewness in highly right-skewed distributions:

| Feature | Original Skew | After Log Transform |
|---------|---------------|---------------------|
| `Log_Monetary` | 18.05 | 1.66 |
| `Log_Avg_Transaction_Value` | 6.11 | 0.34 |
| `Log_Total_Items` | 20.89 | 1.12 |
| `Log_Max_Transaction` | 15.23 | 0.89 |

**Total Features Created:** 14

### Feature Scaling
**Method:** RobustScaler (resistant to outliers)
- Uses median and interquartile range
- More robust than StandardScaler for skewed data

---

## Model Architecture and Methodology

### Clustering Approach: Unsupervised Learning

Since customer segmentation has **no predefined labels**, we use unsupervised learning (clustering) to discover natural groupings in the data.

### Optimal Number of Clusters

#### Evaluation Metrics Used (4 methods)

| Metric | Best k | Score | Interpretation |
|--------|--------|-------|----------------|
| **Silhouette Score** | k=2 | 0.400 | Measures cluster cohesion |
| **Calinski-Harabasz** | k=2 | 2065 | Variance ratio criterion |
| **Davies-Bouldin** | k=5 | 1.216 | Cluster separation (lower is better) |
| **Gap Statistic** | k=8 | 4.281 | Compares to random data |

**Consensus Recommendation:** k=2 (3 out of 4 metrics favor k=2)

#### Decision Override: k=5 Selected

**Rationale:**
- k=2 creates only "High vs. Low value" segments → **Not actionable for marketing**
- k=5 provides **interpretable business segments:**
  - Champions (VIP customers)
  - High-Potential (growing customers)
  - Standard (average customers)
  - New Customers (recent joiners)
  - At-Risk (inactive but valuable)
- Davies-Bouldin metric (best for separation quality) favors k=5
- Business requirement: 3-7 segments typical in retail
- **Trade-off accepted:** Lower silhouette score (0.27) but better business utility

### Algorithms Compared (5 algorithms)

| Algorithm | Clusters | Silhouette | Calinski-Harabasz | Davies-Bouldin | Noise Points |
|-----------|----------|------------|-------------------|----------------|--------------|
| K-Means | 5 | 0.2684 | 1433.17 | 1.2164 | 0 |
| Agglomerative | 5 | 0.1734 | 1203.44 | 1.4636 | 0 |
| Gaussian Mixture | 5 | 0.0568 | 770.53 | 1.9395 | 0 |
| DBSCAN | 19 | -0.1797 | 31.22 | 1.3937 | 2,549 |
| **Consensus** | **5** | **0.2695** | **1433.16** | **1.2157** | **0** |

### Winner: Consensus Clustering (Ensemble Method)

**Why Consensus Clustering?**
- **Highest Silhouette Score:** 0.2695
- **Method:** Combines 10 K-Means runs with different random seeds
- **More stable** than single-run algorithms
- **Better Calinski-Harabasz** than Agglomerative
- **Avoids DBSCAN issues** (2,549 noise points = 62.5% of data)

**Why Not DBSCAN?**
- Created 19 micro-clusters (too fragmented)
- 62.5% of customers labeled as "noise"
- Negative silhouette score = poor cluster quality

**Final Model:** Consensus K-Means (k=5)

### Model Validation Strategy

Since clustering is **unsupervised** (no train/test split), validation uses:

#### 1. Consensus Clustering (Stability Check)
- Ran K-Means 10 times with different random seeds
- Built co-occurrence matrix (how often customers cluster together)
- Applied hierarchical clustering on consensus matrix
- **Result:** ✅ Stable assignments across runs

#### 2. Silhouette Analysis (Quality Check)
- Measures how well each customer fits their cluster
- Range: -1 (misassigned) to +1 (perfect fit)
- **Overall Score:** 0.269 (moderate, expected for customer data)
- **Best Cluster:** Cluster 3 with 0.335 (fair quality)

#### 3. Statistical Validation (ANOVA)
- **Test:** Do clusters actually differ statistically?
- **Result:** All 7 features p < 0.0001 (highly significant)
- **Effect Sizes:** Large (η² > 0.14 for all features)
- **Conclusion:** ✅ Clusters are statistically distinct

#### 4. Business Validation (Interpretability)
- ✅ Segments make intuitive business sense
- ✅ Marketing team can create actionable strategies
- ✅ Clear differentiation between customer groups

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Environment Setup

#### Option 1: Using pip
```bash
# Clone the repository
git clone https://github.com/Eng-AlaaHosny/customer-segmentation-analysis-ML
cd customer-segmentation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using conda
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate customer-segmentation
```

### Required Libraries
See `requirements.txt` for complete list. Key dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0

---

## Execution Instructions

### Step 1: Data Preparation
Ensure `Online Retail.csv` is in the project root directory.

### Step 2: Run the Complete Pipeline
```bash
python customer_segmentation.py
```

This will execute all 10 phases:
1. Data loading and cleaning
2. Feature engineering
3. Exploratory data analysis
4. Data preprocessing
5. Optimal cluster determination
6. Clustering algorithms comparison
7. Statistical validation
8. Cluster visualization
9. Business interpretation
10. Results export

### Step 3: View Results
All outputs are saved in the project directory:
- **Visualizations:** 20 PNG files (01-19)
- **Log file:** `customer_segmentation.log`
- **Data exports:** CSV files with segment assignments



---

## Results and Evaluation

### Quantitative Metrics

#### Clustering Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | 0.2695 | Poor (but expected for customer data) |
| **Calinski-Harabasz** | 1433.16 | Good separation |
| **Davies-Bouldin** | 1.2157 | Good (lower is better) |
| **Number of Clusters** | 5 | Optimal for business use |

#### Statistical Validation (ANOVA Results)

| Feature | F-statistic | P-value | η² (Effect Size) | Significance |
|---------|-------------|---------|------------------|--------------|
| Recency | 366.01 | < 0.0001 | 0.265 | Large ✅ |
| Frequency | 1988.66 | < 0.0001 | 0.662 | Large ✅ |
| Monetary | 2851.74 | < 0.0001 | 0.737 | Large ✅ |
| Avg_Transaction_Value | 1310.85 | < 0.0001 | 0.563 | Large ✅ |
| Tenure | 370.40 | < 0.0001 | 0.267 | Large ✅ |
| Total_Items | 2508.43 | < 0.0001 | 0.711 | Large ✅ |
| Unique_Products | 828.26 | < 0.0001 | 0.449 | Large ✅ |

**Conclusion:** All features show **large effect sizes** (η² > 0.14), confirming strong cluster separation.

### Why Silhouette Score is Low (0.269)

#### Real-World Context
Customer behavior exists on a **CONTINUUM**, not discrete boxes:
- Customers gradually transition between segments
- "High-Potential" customers naturally evolve into "Champions"
- No sharp boundaries in real customer data

#### Industry Benchmark
- Customer segmentation typically achieves **0.2-0.4 silhouette scores**
- This is **normal and expected** for behavioral data
- Image segmentation can achieve 0.7+ (clear boundaries)

#### What We Tried to Improve Score
✅ Tested k=2 to k=8 (k=2 gave 0.40, but not actionable)  
✅ Compared 5 algorithms (Consensus performed best)  
✅ Applied log transformations to reduce skew  
✅ Removed 215 outliers (5%)  
✅ Used 10-iteration consensus for stability  

**Result:** Score improved from 0.18 → 0.27 (+50%)  
Further improvement requires sacrificing business utility.

### Cluster Profiles

#### Detailed Segment Characteristics

| Cluster | Segment Name | Customers | Recency (days) | Frequency | Monetary ($) | Avg Transaction | Total Items | Unique Products |
|---------|--------------|-----------|----------------|-----------|--------------|-----------------|-------------|-----------------|
| **1** | 🟠 High-Potential | 1,106 (27.1%) | 44.6 ± 48.3 | 5.0 ± 2.4 | $1,432 ± $657 | $311.5 ± $105.1 | 854.7 ± 440.2 | 78.4 ± 50.1 |
| **2** | 🔴 Standard | 427 (10.5%) | 99.8 ± 85.7 | 2.0 ± 1.2 | $1,502 ± $993 | $751.0 ± $267.8 | 904.0 ± 556.0 | 64.1 ± 46.1 |
| **3** | ⚪ Standard | 1,992 (48.9%) | 142.7 ± 109.9 | 1.7 ± 1.0 | $325 ± $188 | $209.7 ± $106.6 | 182.0 ± 122.7 | 22.4 ± 18.9 |
| **4** | 🔵 New Customers | 205 (5.0%) | 12.6 ± 9.4 | 2.6 ± 1.9 | $879 ± $752 | $338.3 ± $169.0 | 555.8 ± 506.8 | 57.9 ± 52.0 |
| **5** | 🟡 Champions | 345 (8.5%) | 18.6 ± 22.1 | 12.0 ± 4.9 | $4,450 ± $1,521 | $415.5 ± $184.8 | 2776.3 ± 1092.7 | 163.0 ± 99.3 |

#### Business Interpretation

**Cluster 5: 🟡 Champions (VIP Customers)**
- **Value Score:** 0.980 (Highest)
- **Priority:** 1 (Immediate attention)
- **Characteristics:** High frequency + High spending + Recent activity
- **Strategy:** VIP treatment, premium offers, loyalty programs

**Cluster 1: 🟠 High-Potential**
- **Value Score:** 0.571
- **Priority:** 2
- **Characteristics:** Recent customers with high spending
- **Strategy:** Upsell, cross-sell, personalized recommendations

**Cluster 4: 🔵 New Customers**
- **Value Score:** 0.511
- **Priority:** 4
- **Characteristics:** Very recent, low engagement
- **Strategy:** Onboarding, education, welcome offers

**Cluster 2: 🔴 Standard**
- **Value Score:** 0.443
- **Priority:** 6
- **Characteristics:** Average customers with balanced metrics
- **Strategy:** General marketing, value-based offers

**Cluster 3: ⚪ Standard (Largest Segment)**
- **Value Score:** 0.308
- **Priority:** 6
- **Characteristics:** Lowest engagement, largest group (49%)
- **Strategy:** General marketing, value-based offers

### Cluster Distribution

**Imbalance is REALISTIC:**
- **80/20 Rule:** 20% of customers generate 80% of revenue
- Most customers are casual, occasional buyers
- High-value customers (VIPs) are naturally rare
- This reflects **real retail customer distribution**

### Visualizations

#### 1. t-SNE Visualization (`11_tsne_visualization.png`)
- Non-linear dimensionality reduction
- Perplexity: 30 | Iterations: 1,000
- Shows natural cluster separation and overlap

#### 2. PCA Visualization (`10_pca_visualization.png`)
- PC1: 51.0% variance
- PC2: 17.4% variance
- Total: 68.4% variance explained

#### 3. Feature Profiles Heatmap (`12_feature_profiles_heatmap.png`)
- Shows normalized feature values by cluster
- Clearly differentiates cluster characteristics

#### 4. 3D RFM Plot (`14_3d_rfm_plot.png`)
- Interactive 3D visualization of Recency, Frequency, Monetary
- Color-coded by cluster assignment

#### 5. Silhouette Analysis (`15a_silhouette_analysis.png`)
- Per-cluster silhouette coefficients
- Overall average: 0.269
- Best cluster: Cluster 3 (0.335)

#### 6. Statistical Validation (`09_statistical_validation.png`)
- ANOVA results with effect sizes
- All features show large effect sizes (η² > 0.14)

---

## Project Structure

```
customer-segmentation/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── customer_segmentation.py
├── customer_segmentation.log
├── Online Retail.csv
│
├── 01_summary_statistics.png
├── 02_feature_distributions.png
├── 03_correlation_heatmap.png
├── ... (all 20 PNG files here)
└──  presentation .pdf
```

---

## Key Findings and Recommendations

### Key Findings

1. **Five distinct customer segments identified** with statistically significant differences
2. **Champions (8.5%)** generate the highest value despite being the smallest segment
3. **Standard customers (48.9%)** form the largest segment with low engagement
4. **Strong statistical validation:** All features show large effect sizes (η² > 0.14)
5. **Cluster quality acceptable** for real-world customer data (silhouette = 0.269)

### Business Recommendations

#### Priority 1: Champions (345 customers)
- **Action:** VIP treatment, premium offers, loyalty programs
- **Expected Impact:** Retain highest-value customers

#### Priority 2: High-Potential (1,106 customers)
- **Action:** Upsell, cross-sell, personalized recommendations
- **Expected Impact:** Convert to Champions segment

#### Priority 4: New Customers (205 customers)
- **Action:** Onboarding, education, welcome offers
- **Expected Impact:** Increase engagement and retention

#### Priority 6: Standard Customers (2,419 customers)
- **Action:** General marketing, value-based offers
- **Expected Impact:** Maintain engagement, prevent churn

### Future Work

1. **Feature Engineering:**
   - Add product category features (Electronics vs. Clothing)
   - Include temporal patterns (Seasonal vs. Year-round buyers)
   - Calculate Customer Lifetime Value (CLV) metric
   - **Expected gain:** +0.05-0.08 silhouette score

2. **Advanced Models:**
   - Deep learning approaches (Autoencoders)
   - Time-series clustering for behavioral trends
   - Hierarchical segmentation (sub-segments within segments)

3. **Deployment:**
   - Real-time scoring API
   - Automated segment migration tracking
   - A/B testing framework for marketing strategies

---

## References

1. Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197-208.

2. Daqing, C., Sai Liang, S., & Kun, G. (2012). *UCI Online Retail Data Set*. UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/online+retail](https://archive.ics.uci.edu/ml/datasets/online+retail)

3. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.

4. Kaufman, L., & Rousseeuw, P. J. (2009). *Finding groups in data: an introduction to cluster analysis* (Vol. 344). John Wiley & Sons.

---

## Contact

**Student:** Alaa Hsony Saber  
**Course:** SE 3007 - Introduction to Machine Learning  
**Instructor:** Doktor Öğretim Üyesi Selim Yılmaz  


---

## License

This project is submitted as part of academic coursework for SE 3007 - Introduction to Machine Learning.
