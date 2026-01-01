

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union
import logging

# Scikit-learn imports
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('customer_segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class CustomerSegmentation:
    """
    Enhanced Customer Segmentation Analysis Tool
    
    Features:
    - Robust data loading and cleaning
    - Advanced feature engineering with log transformation
    - Multiple clustering algorithms with consensus clustering
    - Statistical validation using ANOVA and effect sizes
    - t-SNE and PCA visualization
    - Business interpretation with priority-based recommendations
    - Comprehensive visual export capabilities (no CSV)
    - Silhouette analysis for cluster quality evaluation
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the segmentation analyzer.
        
        Args:
            filepath (str): Path to the dataset file (CSV or Excel)
        """
        self.filepath = filepath
        self.data_stats = {}
        self.results = {}
        self.customer_df = None
        self.scaled_data = None
        self.column_mapping = {}
        self.scaler = None
        self.feature_columns = []
        self.raw_df = None
        self.consensus_labels = None
        self.best_algorithm = None
        
        # Configuration
        self.config = {
            'max_clusters': 8,
            'outlier_contamination': 0.05,
            'random_state': 42,
            'consensus_iterations': 10,
            'tsne_perplexity': 30,
            'tsne_iterations': 1000
        }
        
        logger.info(f"CustomerSegmentation initialized with file: {filepath}")
    
    # ========================================================================
    # PHASE 1: DATA LOADING & CLEANING
    # ========================================================================
    def load_and_clean_data(self) -> Optional[pd.DataFrame]:
        """
        Load and preprocess the dataset with robust error handling.
        
        Returns:
            Optional[pd.DataFrame]: Cleaned dataframe or None if loading fails
        """
        logger.info("="*70)
        logger.info("PHASE 1: LOADING AND CLEANING DATA")
        logger.info("="*70)
        
        try:
            logger.info(f"Loading data from: {self.filepath}")
            
            # Determine file type and load
            if self.filepath.lower().endswith('.csv'):
                df = self._load_csv_file()
            elif self.filepath.lower().endswith(('.xls', '.xlsx')):
                df = self._load_excel_file()
            else:
                logger.error(f"Unsupported file format: {self.filepath}")
                return None
                
            if df is None or df.empty:
                logger.error("Failed to load data or empty dataset")
                return None
            
            logger.info(f"✓ Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Clean column names
            df = self._clean_column_names(df)
            
            # Store raw statistics
            self._store_raw_statistics(df)
            
            # Identify key columns
            key_columns = self._identify_key_columns(df)
            if not key_columns.get('customer_id'):
                logger.error("✗ Could not find CustomerID column")
                return None
            
            # Convert date column
            if key_columns.get('invoice_date'):
                df = self._convert_date_column(df, key_columns['invoice_date'])
            
            # Apply cleaning pipeline
            df = self._apply_cleaning_pipeline(df, key_columns)
            
            # Store cleaned statistics
            self._store_cleaned_statistics(df, key_columns)
            
            # Standardize column names
            df = self._standardize_column_names(df, key_columns)
            
            self.raw_df = df
            logger.info("✓ Data cleaning complete")
            return df
            
        except Exception as e:
            logger.error(f"✗ Error in load_and_clean_data: {str(e)}")
            return None
    
    def _load_csv_file(self) -> pd.DataFrame:
        """Load CSV file with multiple encoding attempts."""
        encodings = ['utf-8-sig', 'ISO-8859-1', 'latin1', 'cp1252']
        for encoding in encodings:
            try:
                return pd.read_csv(self.filepath, encoding=encoding, low_memory=False)
            except (UnicodeDecodeError, Exception):
                continue
        # fallback without encoding specification
        return pd.read_csv(self.filepath, low_memory=False)
    
    def _load_excel_file(self) -> pd.DataFrame:
        """Load Excel file."""
        try:
            return pd.read_excel(self.filepath)
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            return None
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names."""
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        return df
    
    def _identify_key_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Identify key columns in the dataset."""
        key_columns = {}
        
        # Map possible column names
        column_mapping = {
            'customer_id': ['customerid', 'customer_id', 'customer', 'clientid'],
            'invoice_no': ['invoiceno', 'invoice_no', 'invoice_number', 'orderid'],
            'invoice_date': ['invoicedate', 'invoice_date', 'orderdate', 'date'],
            'quantity': ['quantity', 'qty', 'amount', 'units'],
            'price': ['unitprice', 'price', 'unit_price', 'cost'],
            'stock_code': ['stockcode', 'stock_code', 'productcode', 'sku']
        }
        
        for key, possible_names in column_mapping.items():
            for col in df.columns:
                if col.lower() in [name.lower() for name in possible_names]:
                    key_columns[key] = col
                    break
        
        return key_columns
    
    def _convert_date_column(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Convert date column to datetime."""
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            valid_dates = df[date_col].notna().sum()
            logger.info(f"✓ Converted {date_col} to datetime. Valid dates: {valid_dates}/{len(df)}")
        except Exception as e:
            logger.warning(f"⚠ Could not convert {date_col} to datetime: {e}")
        return df
    
    def _apply_cleaning_pipeline(self, df: pd.DataFrame, key_columns: Dict) -> pd.DataFrame:
        """Apply comprehensive cleaning pipeline."""
        original_rows = len(df)
        
        # 1. Remove rows without customer ID
        if 'customer_id' in key_columns:
            before = len(df)
            df = df.dropna(subset=[key_columns['customer_id']])
            removed = before - len(df)
            if removed > 0:
                logger.info(f"✓ Removed {removed:,} rows with missing CustomerID")
        
        # 2. Convert customer IDs to string
        df[key_columns['customer_id']] = df[key_columns['customer_id']].astype(str).str.strip()
        
        # 3. Remove cancelled orders
        if 'invoice_no' in key_columns:
            before = len(df)
            cancellation_patterns = ['c', 'cancel', 'void']
            mask = ~df[key_columns['invoice_no']].astype(str).str.lower().str.startswith(tuple(cancellation_patterns))
            df = df[mask]
            removed = before - len(df)
            if removed > 0:
                logger.info(f"✓ Removed {removed:,} cancelled orders")
        
        # 4. Remove negative quantities and prices
        if 'quantity' in key_columns:
            before = len(df)
            df = df[df[key_columns['quantity']] > 0]
            removed = before - len(df)
            if removed > 0:
                logger.info(f"✓ Removed {removed:,} rows with non-positive quantities")
        
        if 'price' in key_columns:
            before = len(df)
            df = df[df[key_columns['price']] > 0]
            removed = before - len(df)
            if removed > 0:
                logger.info(f"✓ Removed {removed:,} rows with non-positive prices")
        
        # 5. Calculate total price
        if 'quantity' in key_columns and 'price' in key_columns:
            df['TotalPrice'] = df[key_columns['quantity']] * df[key_columns['price']]
            logger.info("✓ Created TotalPrice column")
        
        # 6. Remove duplicates
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            logger.info(f"✓ Removed {removed:,} duplicate rows")
        
        # 7. remove extreme outliers in Monetary values
        if 'TotalPrice' in df.columns:
            before = len(df)
            q1 = df['TotalPrice'].quantile(0.01)
            q99 = df['TotalPrice'].quantile(0.99)
            df = df[(df['TotalPrice'] >= q1) & (df['TotalPrice'] <= q99)]
            removed = before - len(df)
            if removed > 0:
                logger.info(f"✓ Removed {removed:,} extreme monetary outliers")
        
        logger.info(f"Cleaning pipeline removed {original_rows - len(df):,} rows total")
        return df
    
    def _store_raw_statistics(self, df: pd.DataFrame):
        """Store raw data statistics."""
        self.data_stats['raw_rows'] = len(df)
        self.data_stats['raw_columns'] = len(df.columns)
        
        for col in df.columns:
            if 'customer' in col.lower() or 'client' in col.lower():
                if not df[col].isna().all():
                    self.data_stats['raw_customers'] = df[col].nunique()
                    break
        
        logger.info(f"📊 Raw Data: {self.data_stats['raw_rows']:,} rows, "
                   f"{self.data_stats['raw_columns']} columns")
    
    def _store_cleaned_statistics(self, df: pd.DataFrame, key_columns: Dict):
        """Store cleaned data statistics."""
        if 'customer_id' in key_columns:
            customer_col = key_columns['customer_id']
            self.data_stats['clean_rows'] = len(df)
            self.data_stats['clean_customers'] = df[customer_col].nunique()
            self.data_stats['rows_removed'] = self.data_stats['raw_rows'] - self.data_stats['clean_rows']
            
            # Enhanced customer retention metrics
            if 'raw_customers' in self.data_stats:
                self.data_stats['customers_removed'] = (
                    self.data_stats['raw_customers'] - self.data_stats['clean_customers']
                )
                self.data_stats['customer_retention_rate'] = (
                    self.data_stats['clean_customers'] / self.data_stats['raw_customers'] * 100
                )
            
            # Transaction retention metrics
            if self.data_stats['raw_rows'] > 0:
                self.data_stats['transaction_retention_rate'] = (
                    self.data_stats['clean_rows'] / self.data_stats['raw_rows'] * 100
                )
            
            logger.info(f"📊 Clean Data: {self.data_stats['clean_rows']:,} rows, "
                       f"{self.data_stats['clean_customers']:,} customers")
            logger.info(f"📊 Transaction Retention: "
                       f"{self.data_stats.get('transaction_retention_rate', 0):.1f}%")
            logger.info(f"📊 Customer Retention: "
                       f"{self.data_stats.get('customer_retention_rate', 0):.1f}%")
    
    def _standardize_column_names(self, df: pd.DataFrame, key_columns: Dict) -> pd.DataFrame:
        """Standardize column names for consistency."""
        rename_map = {}
        
        standard_names = {
            'customer_id': 'CustomerID',
            'invoice_no': 'InvoiceNo',
            'invoice_date': 'InvoiceDate',
            'quantity': 'Quantity',
            'price': 'UnitPrice',
            'stock_code': 'StockCode'
        }
        
        for key, standard_name in standard_names.items():
            if key in key_columns and key_columns[key] != standard_name:
                rename_map[key_columns[key]] = standard_name
                self.column_mapping[key] = key_columns[key]
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"✓ Renamed columns: {rename_map}")
        
        return df
    
    # ========================================================================
    # PHASE 2: FEATURE ENGINEERING
    # ========================================================================
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive RFM and behavioral features.
        
        Args:
            df: Cleaned transaction dataframe
            
        Returns:
            DataFrame with customer-level features
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: FEATURE ENGINEERING")
        logger.info("="*70)
        
        # Create invoice-level aggregation
        logger.info("Creating invoice-level features...")
        invoice_data = df.groupby(['CustomerID', 'InvoiceNo']).agg({
            'TotalPrice': 'sum',
            'InvoiceDate': 'first',
            'Quantity': 'sum'
        }).reset_index()
        
        # Set reference date
        latest_date = invoice_data['InvoiceDate'].max() + pd.Timedelta(days=1)
        logger.info(f"Reference Date: {latest_date.date()}")
        
        # Calculate RFM features
        logger.info("Calculating RFM features...")
        rfm_features = invoice_data.groupby('CustomerID').agg({
            'InvoiceDate': [
                ('Recency', lambda x: (latest_date - x.max()).days),
                ('Frequency', 'count'),
                ('Tenure', lambda x: (latest_date - x.min()).days)
            ],
            'TotalPrice': [
                ('Monetary', 'sum'),
                ('Avg_Transaction_Value', 'mean'),
                ('Max_Transaction', 'max'),
                ('Std_Transaction', 'std')
            ],
            'Quantity': [
                ('Total_Items', 'sum'),
                ('Avg_Items_Per_Transaction', 'mean')
            ]
        }).reset_index()
        
        # Flatten multi-index columns
        rfm_features.columns = ['CustomerID', 'Recency', 'Frequency', 'Tenure',
                               'Monetary', 'Avg_Transaction_Value', 'Max_Transaction',
                               'Std_Transaction', 'Total_Items', 'Avg_Items_Per_Transaction']
        
        # Calculate behavioral features
        logger.info("Calculating behavioral features...")
        behavioral_features = df.groupby('CustomerID').agg({
            'StockCode': [
                ('Unique_Products', 'nunique'),
                ('Transaction_Variety', lambda x: x.nunique() / len(x) if len(x) > 0 else 0)
            ]
        }).reset_index()
        
        behavioral_features.columns = ['CustomerID', 'Unique_Products', 'Transaction_Variety']
        
        # Merge features
        customer_features = pd.merge(rfm_features, behavioral_features, on='CustomerID', how='left')
        
        # Create derived rate-based features
        logger.info("Creating derived features...")
        customer_features['Tenure'] = customer_features['Tenure'].replace(0, 1)
        customer_features['Frequency_Rate'] = customer_features['Frequency'] / customer_features['Tenure']
        customer_features['Monetary_Rate'] = customer_features['Monetary'] / customer_features['Tenure']
        customer_features['Items_Per_Transaction'] = customer_features['Total_Items'] / customer_features['Frequency']
        
        # Handle infinite values
        customer_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Apply log transformation for skewed features
        logger.info("Applying log transformation...")
        skewed_features = ['Monetary', 'Avg_Transaction_Value', 'Total_Items', 'Max_Transaction']
        for col in skewed_features:
            if col in customer_features.columns:
                original_skew = customer_features[col].skew()
                if abs(original_skew) > 1:
                    customer_features[f'Log_{col}'] = np.log1p(customer_features[col])
                    log_skew = customer_features[f'Log_{col}'].skew()
                    logger.info(f"  • {col}: skew {original_skew:.2f} → {log_skew:.2f}")
        
        # Fill missing values
        customer_features.fillna(0, inplace=True)
        
        logger.info(f"✓ Feature engineering complete")
        logger.info(f"  • Features created: {len(customer_features.columns) - 1}")
        logger.info(f"  • Customers: {len(customer_features):,}")
        
        return customer_features
    
    # ========================================================================
    # PHASE 3: EXPLORATORY DATA ANALYSIS
    # ========================================================================
    def perform_eda(self, customer_features: pd.DataFrame):
        """
        Perform comprehensive exploratory data analysis.
        
        Args:
            customer_features: DataFrame with customer features
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: EXPLORATORY DATA ANALYSIS")
        logger.info("="*70)
        
        # Define features for analysis
        analysis_features = [
            'Recency', 'Frequency', 'Monetary', 'Avg_Transaction_Value',
            'Tenure', 'Total_Items', 'Unique_Products'
        ]
        
        # Create summary statistics visualization
        logger.info("Creating summary statistics table...")
        self._create_summary_statistics_table(customer_features, analysis_features)
        
        # Create distribution plots
        logger.info("Creating distribution plots...")
        self._create_distribution_plots(customer_features, analysis_features)
        
        # Create correlation matrix
        logger.info("Creating correlation heatmap...")
        self._create_correlation_heatmap(customer_features, analysis_features)
        
        # Create boxplots for outlier detection
        logger.info("Creating outlier detection plots...")
        self._create_outlier_plots(customer_features, analysis_features)
        
        # Create RFM pairplot
        logger.info("Creating RFM pairplot...")
        self._create_rfm_pairplot(customer_features)
        
        logger.info("✓ EDA complete")
    
    def _create_summary_statistics_table(self, df: pd.DataFrame, features: List[str]):
        """Create visual summary statistics table."""
        summary_stats = df[features].describe()
        summary_stats.loc['skewness'] = df[features].skew()
        summary_stats.loc['kurtosis'] = df[features].kurtosis()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, len(features) * 0.5 + 3))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table_data = []
        for col in features:
            stats = summary_stats[col]
            table_data.append([
                col,
                f"{stats['count']:,.0f}",
                f"{stats['mean']:,.2f}",
                f"{stats['std']:,.2f}",
                f"{stats['min']:,.2f}",
                f"{stats['25%']:,.2f}",
                f"{stats['50%']:,.2f}",
                f"{stats['75%']:,.2f}",
                f"{stats['max']:,.2f}",
                f"{stats['skewness']:.2f}",
                f"{stats['kurtosis']:.2f}"
            ])
        
        columns = ['Feature', 'Count', 'Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max', 'Skew', 'Kurtosis']
        
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(len(columns)):
                    table[(i, j)].set_facecolor('#f5f5f5')
        
        plt.title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('01_summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 01_summary_statistics.png")
    
    def _create_distribution_plots(self, df: pd.DataFrame, features: List[str]):
        """Create distribution plots for features."""
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # Histogram with KDE
            sns.histplot(df[feature], bins=50, ax=ax, kde=True,
                        color='steelblue', edgecolor='black', alpha=0.7)
            
            # Add statistics
            mean_val = df[feature].mean()
            median_val = df[feature].median()
            skewness = df[feature].skew()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                      label=f'Median: {median_val:.1f}')
            
            ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Skew: {skewness:.2f}\nMean: {mean_val:.1f}\nMedian: {median_val:.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig('02_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 02_feature_distributions.png")
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, features: List[str]):
        """Create correlation heatmap."""
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdYlGn', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 03_correlation_heatmap.png")
    
    def _create_outlier_plots(self, df: pd.DataFrame, features: List[str]):
        """Create boxplots for outlier detection."""
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # Create boxplot
            box = ax.boxplot(df[feature], patch_artist=True)
            
            # Customize boxplot
            box['boxes'][0].set_facecolor('lightblue')
            box['boxes'][0].set_edgecolor('black')
            box['boxes'][0].set_linewidth(1.5)
            
            # Add statistics
            q1 = df[feature].quantile(0.25)
            q3 = df[feature].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[feature] < q1 - 1.5*iqr) | (df[feature] > q3 + 1.5*iqr)]
            
            ax.set_title(f'{feature} - Outlier Detection', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add outlier count
            ax.text(0.05, 0.95, f'Outliers: {len(outliers):,}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig('04_outlier_detection.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 04_outlier_detection.png")
    
    def _create_rfm_pairplot(self, df: pd.DataFrame):
        """Create RFM pairplot."""
        # Sample data if too large
        if len(df) > 5000:
            sample_df = df[['Recency', 'Frequency', 'Monetary']].sample(5000, random_state=42)
            logger.info("  Using sample of 5,000 customers for pairplot")
        else:
            sample_df = df[['Recency', 'Frequency', 'Monetary']]
        
        # Create pairplot
        pairplot = sns.pairplot(sample_df, diag_kind='kde', plot_kws={'alpha': 0.6})
        pairplot.fig.suptitle('RFM Features Pairplot', y=1.02, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('05_rfm_pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 05_rfm_pairplot.png")
    
    # ========================================================================
    # PHASE 4: DATA PREPROCESSING
    # ========================================================================
    def preprocess_data(self, customer_features: pd.DataFrame) -> List[str]:
        """
        Preprocess data for clustering.
        
        Args:
            customer_features: DataFrame with customer features
            
        Returns:
            List of feature column names used for clustering
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 4: DATA PREPROCESSING")
        logger.info("="*70)
        
        # Handle missing values
        logger.info("Handling missing values...")
        customer_features.fillna(0, inplace=True)
        
        # Select features for clustering
        logger.info("Selecting features for clustering...")
        base_features = [
            'Recency', 'Frequency', 'Monetary', 'Avg_Transaction_Value',
            'Tenure', 'Unique_Products', 'Total_Items',
            'Frequency_Rate', 'Monetary_Rate', 'Items_Per_Transaction'
        ]
        
        # Include log-transformed features
        log_features = [col for col in customer_features.columns if col.startswith('Log_')]
        
        # Combine all features
        all_features = base_features + log_features
        
        # Filter to existing columns
        feature_columns = [col for col in all_features if col in customer_features.columns]
        self.feature_columns = feature_columns
        
        logger.info(f"Selected {len(feature_columns)} features:")
        for col in feature_columns:
            logger.info(f"  • {col}")
        
        # Detect and remove outliers using Isolation Forest
        logger.info("Detecting and removing outliers...")
        iso_forest = IsolationForest(
            contamination=self.config['outlier_contamination'],
            random_state=self.config['random_state'],
            n_estimators=100
        )
        
        outlier_labels = iso_forest.fit_predict(customer_features[feature_columns])
        n_outliers = sum(outlier_labels == -1)
        
        customer_features_clean = customer_features[outlier_labels == 1].copy()
        
        logger.info(f"  • Removed {n_outliers:,} outliers ({n_outliers/len(customer_features)*100:.1f}%)")
        
        # Store outlier statistics
        self.data_stats['before_outlier_removal'] = len(customer_features)
        self.data_stats['after_outlier_removal'] = len(customer_features_clean)
        self.data_stats['outliers_removed'] = n_outliers
        
        # Create outlier removal visualization
        self._create_outlier_removal_visualization(customer_features, customer_features_clean)
        
        # Apply robust scaling
        logger.info("Applying RobustScaler...")
        self.scaler = RobustScaler()
        self.scaled_data = self.scaler.fit_transform(customer_features_clean[feature_columns])
        
        # Store processed data
        self.customer_df = customer_features_clean.reset_index(drop=True)
        self.customer_df['CustomerID'] = customer_features_clean['CustomerID'].values
        
        logger.info("✓ Preprocessing complete")
        logger.info(f"  • Final dataset: {len(self.customer_df):,} customers")
        logger.info(f"  • Scaled data shape: {self.scaled_data.shape}")
        
        return feature_columns
    
    def _create_outlier_removal_visualization(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame):
        """Create visualization showing outlier removal impact."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        key_features = ['Recency', 'Frequency', 'Monetary', 'Total_Items']
        
        for idx, feature in enumerate(key_features):
            if feature not in original_df.columns:
                continue
            
            ax = axes[idx // 2, idx % 2]
            
            # Plot original vs cleaned
            ax.scatter(range(len(original_df)), original_df[feature], 
                      alpha=0.3, s=10, label='Original', color='blue')
            ax.scatter(range(len(cleaned_df)), cleaned_df[feature], 
                      alpha=0.5, s=10, label='Cleaned', color='red')
            
            ax.set_title(f'{feature} - Before vs After Outlier Removal', fontsize=11)
            ax.set_xlabel('Customer Index')
            ax.set_ylabel(feature)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Outlier Removal Impact', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('06_outlier_removal_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 06_outlier_removal_impact.png")
    
    # ========================================================================
    # PHASE 5: OPTIMAL CLUSTER DETERMINATION
    # ========================================================================
    def find_optimal_clusters(self, max_k: int = None) -> int:
        """
        Determine optimal number of clusters using multiple metrics.
        
        Args:
            max_k: Maximum number of clusters to evaluate
            
        Returns:
            Optimal number of clusters
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 5: OPTIMAL CLUSTER DETERMINATION")
        logger.info("="*70)
        
        if max_k is None:
            max_k = self.config['max_clusters']
        
        k_range = range(2, max_k + 1)
        logger.info(f"Testing k from {min(k_range)} to {max(k_range)}...")
        
        # Initialize metrics storage
        metrics = {
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'wcss': [],
            'gap_statistic': []
        }
        
        # Sample data if large
        if len(self.scaled_data) > 10000:
            sample_indices = np.random.choice(
                len(self.scaled_data),
                10000,
                replace=False,
                random_state=self.config['random_state']
            )
            data_sample = self.scaled_data[sample_indices]
            logger.info(f"Using sample of 10,000 customers for evaluation")
        else:
            data_sample = self.scaled_data
        
        # Evaluate each k
        for k in k_range:
            logger.info(f"  Testing k={k}...")
            
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.config['random_state'],
                n_init=10
            )
            labels = kmeans.fit_predict(data_sample)
            
            # Calculate metrics
            if len(np.unique(labels)) > 1:
                metrics['silhouette'].append(silhouette_score(data_sample, labels))
            else:
                metrics['silhouette'].append(-1)
            
            metrics['calinski_harabasz'].append(calinski_harabasz_score(data_sample, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(data_sample, labels))
            metrics['wcss'].append(kmeans.inertia_)
            
            # Calculate gap statistic
            gap_stat = self._calculate_gap_statistic(data_sample, k)
            metrics['gap_statistic'].append(gap_stat)
        
        # Find optimal k for each metric
        optimal_k_silhouette = k_range[np.argmax(metrics['silhouette'])]
        optimal_k_calinski = k_range[np.argmax(metrics['calinski_harabasz'])]
        optimal_k_davies = k_range[np.argmin(metrics['davies_bouldin'])]
        optimal_k_gap = k_range[np.argmax(metrics['gap_statistic'])]
        
        # Calculate consensus score
        optimal_k_consensus = self._calculate_consensus_k(k_range, metrics)
        
        # Log results
        logger.info("\n📊 OPTIMAL K ANALYSIS:")
        logger.info(f"  • Silhouette Score: k={optimal_k_silhouette} "
                   f"(score: {metrics['silhouette'][optimal_k_silhouette-2]:.3f})")
        logger.info(f"  • Calinski-Harabasz: k={optimal_k_calinski} "
                   f"(score: {metrics['calinski_harabasz'][optimal_k_calinski-2]:.0f})")
        logger.info(f"  • Davies-Bouldin: k={optimal_k_davies} "
                   f"(score: {metrics['davies_bouldin'][optimal_k_davies-2]:.3f})")
        logger.info(f"  • Gap Statistic: k={optimal_k_gap} "
                   f"(score: {metrics['gap_statistic'][optimal_k_gap-2]:.3f})")
        logger.info(f"  • Consensus: k={optimal_k_consensus}")
        
        # Plot metrics and create optimal k table
        self._plot_cluster_metrics(k_range, metrics, optimal_k_consensus)
        
        # Final selection
        final_k = optimal_k_consensus
        if final_k < 3 or final_k > 7:
            logger.info(f"⚠  Consensus k={final_k} outside typical range (3-7)")
            logger.info(f"   Using k=5 for better interpretability")
            final_k = 5
        
        logger.info(f"\n🎯 FINAL SELECTED k: {final_k}")
        return final_k
    
    def _calculate_gap_statistic(self, data: np.ndarray, k: int, n_refs: int = 3) -> float:
        """Calculate gap statistic for given k."""
        ref_inertias = []
        
        for i in range(n_refs):
            random_data = np.random.uniform(
                low=data.min(),
                high=data.max(),
                size=data.shape
            )
            
            kmeans_random = KMeans(
                n_clusters=k,
                random_state=self.config['random_state'] + i,
                n_init=10
            )
            kmeans_random.fit(random_data)
            ref_inertias.append(kmeans_random.inertia_)
        
        kmeans_actual = KMeans(
            n_clusters=k,
            random_state=self.config['random_state'],
            n_init=10
        )
        kmeans_actual.fit(data)
        
        gap_stat = np.log(np.mean(ref_inertias)) - np.log(kmeans_actual.inertia_)
        return gap_stat
    
    def _calculate_consensus_k(self, k_range: range, metrics: Dict) -> int:
        """Calculate consensus optimal k using weighted scores."""
        k_scores = []
        
        for i, k in enumerate(k_range):
            sil_norm = (metrics['silhouette'][i] - min(metrics['silhouette'])) / \
                      (max(metrics['silhouette']) - min(metrics['silhouette']) + 1e-10)
            
            cal_norm = (metrics['calinski_harabasz'][i] - min(metrics['calinski_harabasz'])) / \
                      (max(metrics['calinski_harabasz']) - min(metrics['calinski_harabasz']) + 1e-10)
            
            dav_norm = 1 - ((metrics['davies_bouldin'][i] - min(metrics['davies_bouldin'])) / \
                           (max(metrics['davies_bouldin']) - min(metrics['davies_bouldin']) + 1e-10))
            
            gap_norm = (metrics['gap_statistic'][i] - min(metrics['gap_statistic'])) / \
                      (max(metrics['gap_statistic']) - min(metrics['gap_statistic']) + 1e-10)
            
            score = (0.3 * sil_norm + 0.2 * cal_norm + 
                    0.2 * dav_norm + 0.3 * gap_norm)
            k_scores.append(score)
        
        return k_range[np.argmax(k_scores)]
    
    def _plot_cluster_metrics(self, k_range: range, metrics: Dict, optimal_k: int):
        """Plot cluster evaluation metrics and create optimal k table."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot each metric
        plots = [
            ('silhouette', 'Silhouette Score', 'bo-'),
            ('calinski_harabasz', 'Calinski-Harabasz', 'go-'),
            ('davies_bouldin', 'Davies-Bouldin', 'ro-'),
            ('gap_statistic', 'Gap Statistic', 'mo-'),
            ('wcss', 'Within-Cluster Sum of Squares', 'co-')
        ]
        
        for idx, (metric_name, title, style) in enumerate(plots):
            ax = axes[idx // 3, idx % 3]
            
            ax.plot(k_range, metrics[metric_name], style, linewidth=2, markersize=8)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Highlight optimal k
            if metric_name == 'davies_bouldin':
                best_k = k_range[np.argmin(metrics[metric_name])]
            else:
                best_k = k_range[np.argmax(metrics[metric_name])]
            
            ax.axvline(best_k, color='red', linestyle='--', alpha=0.7)
            ax.text(best_k, ax.get_ylim()[1] * 0.9, f'k={best_k}', 
                   ha='center', va='bottom', fontweight='bold')
        
        # Create optimal k comparison table
        ax_table = axes[1, 2]
        ax_table.axis('tight')
        ax_table.axis('off')
        
        table_data = [
            ['Silhouette', k_range[np.argmax(metrics['silhouette'])], f"{max(metrics['silhouette']):.3f}"],
            ['Calinski-Harabasz', k_range[np.argmax(metrics['calinski_harabasz'])], f"{max(metrics['calinski_harabasz']):.1f}"],
            ['Davies-Bouldin', k_range[np.argmin(metrics['davies_bouldin'])], f"{min(metrics['davies_bouldin']):.3f}"],
            ['Gap Statistic', k_range[np.argmax(metrics['gap_statistic'])], f"{max(metrics['gap_statistic']):.3f}"],
            ['Consensus', optimal_k, '']
        ]
        
        columns = ['Metric', 'Optimal k', 'Score']
        
        table = ax_table.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#2E8B57')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style consensus row
        table[(4, 0)].set_facecolor('#FFD700')
        table[(4, 1)].set_facecolor('#FFD700')
        table[(4, 2)].set_facecolor('#FFD700')
        
        ax_table.set_title('Optimal Cluster Selection', fontsize=14, fontweight='bold')
        
        plt.suptitle('Cluster Evaluation Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('07_cluster_evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 07_cluster_evaluation_metrics.png")
    
    # ========================================================================
    # PHASE 6:CLUSTERING ALGORITHMS.
    # ========================================================================
    def apply_clustering_algorithms(self, optimal_k: int):
        """
        Apply multiple clustering algorithms and consensus clustering.
        
        Args:
            optimal_k: Optimal number of clusters
        """
        logger.info("\n" + "="*70)
        logger.info(f"PHASE 6: CLUSTERING ALGORITHMS (k={optimal_k})")
        logger.info("="*70)
        
        # define algorithms to try
        algorithms = {
            'KMeans': KMeans(
                n_clusters=optimal_k,
                random_state=self.config['random_state'],
                n_init=10
            ),
            'Agglomerative': AgglomerativeClustering(n_clusters=optimal_k),
            'GaussianMixture': GaussianMixture(
                n_components=optimal_k,
                random_state=self.config['random_state']
            )
        }
        
        #Try DBSCAN if data is not too large
        if len(self.scaled_data) < 10000:
            algorithms['DBSCAN'] = DBSCAN(eps=0.5, min_samples=5)
        
        logger.info(f"Training {len(algorithms)} clustering algorithms...")
        
        algorithm_results = []
        
        for name, algorithm in algorithms.items():
            logger.info(f"\n🔍 {name}:")
            
            try:
                if name == 'DBSCAN':
                    labels = algorithm.fit_predict(self.scaled_data)
                    n_clusters = len(np.unique(labels[labels != -1]))
                    n_noise = sum(labels == -1)
                    
                    labels = labels.astype(int)
                    valid_mask = labels != -1
                    
                    if n_clusters > 1 and sum(valid_mask) > 0:
                        silhouette = silhouette_score(
                            self.scaled_data[valid_mask],
                            labels[valid_mask]
                        )
                        calinski = calinski_harabasz_score(
                            self.scaled_data[valid_mask],
                            labels[valid_mask]
                        )
                        davies = davies_bouldin_score(
                            self.scaled_data[valid_mask],
                            labels[valid_mask]
                        )
                    else:
                        silhouette = -1
                        calinski = 0
                        davies = 999
                        
                else:
                    labels = algorithm.fit_predict(self.scaled_data)
                    n_clusters = optimal_k
                    n_noise = 0
                    
                    labels = labels.astype(int) + 1
                    
                    silhouette = silhouette_score(self.scaled_data, labels)
                    calinski = calinski_harabasz_score(self.scaled_data, labels)
                    davies = davies_bouldin_score(self.scaled_data, labels)
                
                # Store results
                self.customer_df[f'{name}_Cluster'] = labels
                self.results[name] = {
                    'labels': labels,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise
                }
                
                algorithm_results.append({
                    'Algorithm': name,
                    'Clusters': n_clusters,
                    'Silhouette': f"{silhouette:.4f}",
                    'Calinski-Harabasz': f"{calinski:.0f}",
                    'Davies-Bouldin': f"{davies:.4f}",
                    'Noise Points': f"{n_noise:,}"
                })
                
                logger.info(f"  ✓ Success")
                logger.info(f"    • Clusters: {n_clusters}")
                if n_noise > 0:
                    logger.info(f"    • Noise points: {n_noise:,}")
                logger.info(f"    • Silhouette: {silhouette:.4f}")
                logger.info(f"    • Calinski-Harabasz: {calinski:.2f}")
                logger.info(f"    • Davies-Bouldin: {davies:.4f}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {str(e)}")
                continue
        
        # apply consensus clustering
        self._apply_consensus_clustering(optimal_k)
        
        # add consensus to results
        if 'Consensus' in self.results:
            consensus_result = self.results['Consensus']
            algorithm_results.append({
                'Algorithm': 'Consensus',
                'Clusters': consensus_result['n_clusters'],
                'Silhouette': f"{consensus_result['silhouette']:.4f}",
                'Calinski-Harabasz': f"{consensus_result['calinski_harabasz']:.0f}",
                'Davies-Bouldin': f"{consensus_result['davies_bouldin']:.4f}",
                'Noise Points': '0'
            })
        
        #Create algorithm comparison table
        self._create_algorithm_comparison_table(algorithm_results)
        
        # Determine best algorithm
        self._determine_best_algorithm()
    
    def _apply_consensus_clustering(self, n_clusters: int, n_iterations: int = None):
        """Apply consensus clustering for more stable results."""
        if n_iterations is None:
            n_iterations = self.config['consensus_iterations']
        
        logger.info(f"\n🔍 Consensus Clustering ({n_iterations} iterations)...")
        
        n_samples = len(self.scaled_data)
        consensus_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_iterations):
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.config['random_state'] + i,
                n_init=10
            )
            labels = kmeans.fit_predict(self.scaled_data)
            
            for cluster_id in range(n_clusters):
                cluster_mask = labels == cluster_id
                consensus_matrix[np.ix_(cluster_mask, cluster_mask)] += 1
        
        consensus_matrix /= n_iterations
        
        final_clusters = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        consensus_labels = final_clusters.fit_predict(1 - consensus_matrix) + 1
        
        #Store results
        self.customer_df['Consensus_Cluster'] = consensus_labels
        self.consensus_labels = consensus_labels
        
        # ccalculate metrics
        silhouette = silhouette_score(self.scaled_data, consensus_labels)
        calinski = calinski_harabasz_score(self.scaled_data, consensus_labels)
        davies = davies_bouldin_score(self.scaled_data, consensus_labels)
        
        self.results['Consensus'] = {
            'labels': consensus_labels,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'n_clusters': n_clusters,
            'n_noise': 0
        }
        
        logger.info(f"  ✓ Consensus clustering complete")
        logger.info(f"    • Silhouette: {silhouette:.4f}")
        logger.info(f"    • Calinski-Harabasz: {calinski:.2f}")
        logger.info(f"    • Davies-Bouldin: {davies:.4f}")
    
    def _create_algorithm_comparison_table(self, results: List[Dict]):
        """Create algorithm comparison table visualization."""
        fig, ax = plt.subplots(figsize=(14, len(results) * 0.6 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for result in results:
            table_data.append([
                result['Algorithm'],
                result['Clusters'],
                result['Silhouette'],
                result['Calinski-Harabasz'],
                result['Davies-Bouldin'],
                result['Noise Points']
            ])
        
        columns = ['Algorithm', 'Clusters', 'Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin', 'Noise Points']
        
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best algorithm
        if results:
            best_algo = max(results, key=lambda x: float(x['Silhouette']))
            best_idx = results.index(best_algo) + 1
            for i in range(len(columns)):
                table[(best_idx, i)].set_facecolor('#FFD700')
                table[(best_idx, i)].set_text_props(weight='bold')
        
        plt.title('Algorithm Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('08_algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 08_algorithm_comparison.png")
    
    def _determine_best_algorithm(self):
        """Determine the best performing algorithm."""
        if not self.results:
            return
        
        best_algo = max(self.results.items(), key=lambda x: x[1]['silhouette'])[0]
        self.best_algorithm = best_algo
        
        logger.info("\n" + "="*70)
        logger.info("ALGORITHM COMPARISON")
        logger.info("="*70)
        
        header = f"{'Algorithm':<20} {'Silhouette':<12} {'Calinski-H':<12} {'Davies-B':<12} {'Clusters':<10} {'Noise':<10}"
        logger.info(header)
        logger.info("-" * len(header))
        
        for name, result in self.results.items():
            row = f"{name:<20} {result['silhouette']:>11.4f} " \
                  f"{result['calinski_harabasz']:>11.2f} " \
                  f"{result['davies_bouldin']:>11.4f} " \
                  f"{result['n_clusters']:>9} " \
                  f"{result['n_noise']:>9}"
            logger.info(row)
        
        logger.info(f"\n🏆 BEST ALGORITHM: {best_algo}")
    
    # ========================================================================
    # PHASE 7: STATISTICAL VALIDATION..
    # ========================================================================
    def validate_clusters_statistically(self):
        """
        Perform statistical validation of clusters.
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 7: STATISTICAL VALIDATION")
        logger.info("="*70)
        
        if not self.results or self.best_algorithm is None:
            logger.error("No clustering results to validate")
            return
        
        cluster_col = f'{self.best_algorithm}_Cluster'
        logger.info(f"Validating clusters from {self.best_algorithm}...")
        
        # defin e features for validation
        validation_features = [
            'Recency', 'Frequency', 'Monetary', 'Avg_Transaction_Value',
            'Tenure', 'Total_Items', 'Unique_Products'
        ]
        
        #Filter out noise if present
        validation_df = self.customer_df.copy()
        if -1 in validation_df[cluster_col].values:
            validation_df = validation_df[validation_df[cluster_col] != -1]
        
        # Perform ANOVA for features
        anova_results = []
        
        for feature in validation_features:
            if feature not in validation_df.columns:
                continue
            
            #group data by Cluster
            clusters_data = []
            cluster_labels = sorted(validation_df[cluster_col].unique())
            
            for cluster in cluster_labels:
                cluster_data = validation_df[validation_df[cluster_col] == cluster][feature]
                clusters_data.append(cluster_data)
            
            #perform one-way ANOVA
            try:
                f_stat, p_value = stats.f_oneway(*clusters_data)
                
                #calculate effect size (eta-squared)
                grand_mean = validation_df[feature].mean()
                ss_between = sum(len(data) * (data.mean() - grand_mean) ** 2 for data in clusters_data)
                ss_total = sum((validation_df[feature] - grand_mean) ** 2)
                
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                #Interpret effect size
                if eta_squared >= 0.14:
                    effect_size = 'Large'
                elif eta_squared >= 0.06:
                    effect_size = 'Medium'
                elif eta_squared >= 0.01:
                    effect_size = 'Small'
                else:
                    effect_size = 'Negligible'
                
                anova_results.append({
                    'Feature': feature,
                    'F_statistic': f"{f_stat:.2f}",
                    'P_value': f"{p_value:.4f}",
                    'Significant': p_value < 0.05,
                    'Eta_squared': f"{eta_squared:.3f}",
                    'Effect_Size': effect_size
                })
                
            except Exception as e:
                logger.warning(f"Could not perform ANOVA for {feature}: {e}")
                continue
        
        #create statistical validation visualization
        self._create_statistical_validation_visualization(anova_results)
        
        # Log summary
        n_significant = sum(1 for result in anova_results if result['Significant'])
        avg_eta = sum(float(result['Eta_squared']) for result in anova_results) / len(anova_results)
        
        logger.info(f"\n📊 STATISTICAL VALIDATION RESULTS:")
        logger.info(f"  • Significant features: {n_significant}/{len(anova_results)}")
        logger.info(f"  • Average effect size (η²): {avg_eta:.3f}")
        
        if avg_eta >= 0.14:
            logger.info("  • Interpretation: Strong cluster separation")
        elif avg_eta >= 0.06:
            logger.info("  • Interpretation: Moderate cluster separation")
        else:
            logger.info("  • Interpretation: Weak cluster separation")
    
    def _create_statistical_validation_visualization(self, anova_results: List[Dict]):
        """Create visualization of statistical validation results."""
        if not anova_results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        #pplot 1: Statistical significance table
        ax_table = axes[0]
        ax_table.axis('tight')
        ax_table.axis('off')
        
        table_data = []
        for result in anova_results:
            sig_symbol = '✓' if result['Significant'] else '✗'
            table_data.append([
                result['Feature'],
                result['F_statistic'],
                result['P_value'],
                result['Eta_squared'],
                result['Effect_Size'],
                sig_symbol
            ])
        
        columns = ['Feature', 'F-statistic', 'P-value', 'η²', 'Effect Size', 'Significant']
        
        table = ax_table.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style significant rows
        for i, result in enumerate(anova_results, 1):
            if result['Significant']:
                for j in range(len(columns)):
                    table[(i, j)].set_facecolor('#90EE90')  # Light green
        
        # Style effect size cells
        for i, result in enumerate(anova_results, 1):
            effect_size = result['Effect_Size']
            effect_col_idx = 4
            if effect_size == 'Large':
                table[(i, effect_col_idx)].set_facecolor('#FF9999')
            elif effect_size == 'Medium':
                table[(i, effect_col_idx)].set_facecolor('#FFCC99')
            elif effect_size == 'Small':
                table[(i, effect_col_idx)].set_facecolor('#FFFF99')
        
        ax_table.set_title('ANOVA Results - Statistical Significance', fontsize=12, fontweight='bold')
        
        #plot 2:Effect size visualization
        ax_bar = axes[1]
        features = [r['Feature'] for r in anova_results]
        eta_squared = [float(r['Eta_squared']) for r in anova_results]
        colors = ['#FF9999' if float(r['Eta_squared']) >= 0.14 else 
                 '#FFCC99' if float(r['Eta_squared']) >= 0.06 else
                 '#FFFF99' if float(r['Eta_squared']) >= 0.01 else '#CCCCCC' 
                 for r in anova_results]
        
        bars = ax_bar.barh(features, eta_squared, color=colors, edgecolor='black')
        ax_bar.axvline(0.14, color='red', linestyle='--', linewidth=2, label='Large (η²≥0.14)')
        ax_bar.axvline(0.06, color='orange', linestyle='--', linewidth=2, label='Medium (η²≥0.06)')
        ax_bar.axvline(0.01, color='yellow', linestyle='--', linewidth=2, label='Small (η²≥0.01)')
        
        #sdd values on bars
        for bar, value in zip(bars, eta_squared):
            width = bar.get_width()
            ax_bar.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', va='center', fontsize=9)
        
        ax_bar.set_xlabel('Effect Size (η²)', fontsize=11)
        ax_bar.set_title('Effect Size by Feature', fontsize=12, fontweight='bold')
        ax_bar.legend()
        ax_bar.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f'Statistical Validation - {self.best_algorithm}', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('09_statistical_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 09_statistical_validation.png")
    
    # ========================================================================
    # PHASE 8:VISUALIZATION
    # ========================================================================
    def visualize_clusters(self):
        """Create comprehensive cluster visualizations."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 8: CLUSTER VISUALIZATION")
        logger.info("="*70)
        
        if self.best_algorithm is None:
            logger.error("No clustering results to visualize")
            return
        
        cluster_col = f'{self.best_algorithm}_Cluster'
        
        #1. PCA visualization
        logger.info("Creating PCA visualization...")
        self._create_pca_visualization(cluster_col)
        
        #2. t-SNE Visualization
        logger.info("Creating t-SNE visualization...")
        self._create_tsne_visualization(cluster_col)
        
        # 3.Feature Profile s
        logger.info("Creating feature profiles...")
        self._create_feature_profiles(cluster_col)
        
        #4. cluster distribution
        logger.info("Creating cluster distribution plots...")
        self._create_cluster_distribution(cluster_col)
        
        #5.3D RFM Plot
        logger.info("Creating 3D RFM plot...")
        self._create_3d_rfm_plot(cluster_col)
        
        # 6.sluster Profiles table
        logger.info("Creating cluster profiles table...")
        self._create_cluster_profiles_table(cluster_col)
        
        # 7. silhouette Analysis 
        logger.info("Creating silhouette analysis...")
        self._create_silhouette_analysis(cluster_col)
        
        logger.info("✓ Visualization complete")
    
    def _create_pca_visualization(self, cluster_col: str):
        """Create PCA visualization."""
        pca = PCA(n_components=2, random_state=self.config['random_state'])
        pca_components = pca.fit_transform(self.scaled_data)
        
        explained_var = pca.explained_variance_ratio_
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_components[:, 0], pca_components[:, 1],
                            c=self.customer_df[cluster_col], cmap='tab20',
                            alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'PCA Visualization - {self.best_algorithm}\n'
                 f'Variance Explained: {explained_var.sum():.1%}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=12)
        plt.ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('10_pca_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 10_pca_visualization.png")
    
    def _create_tsne_visualization(self, cluster_col: str):
        """Create t-SNE visualization."""
        if len(self.scaled_data) > 10000:
            sample_indices = np.random.choice(
                len(self.scaled_data),
                5000,
                replace=False,
                random_state=self.config['random_state']
            )
            data_sample = self.scaled_data[sample_indices]
            labels_sample = self.customer_df[cluster_col].iloc[sample_indices].values
            logger.info("  Using sample of 5,000 for t-SNE")
        else:
            data_sample = self.scaled_data
            labels_sample = self.customer_df[cluster_col].values
        
        try:
            tsne = TSNE(
                n_components=2,
                random_state=self.config['random_state'],
                perplexity=self.config['tsne_perplexity'],
                n_iter=self.config['tsne_iterations']
            )
            tsne_components = tsne.fit_transform(data_sample)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(tsne_components[:, 0], tsne_components[:, 1],
                                c=labels_sample, cmap='tab20',
                                alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
            
            plt.colorbar(scatter, label='Cluster')
            plt.title(f't-SNE Visualization - {self.best_algorithm}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE Component 1', fontsize=12)
            plt.ylabel('t-SNE Component 2', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('11_tsne_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✓ Saved: 11_tsne_visualization.png")
            
        except Exception as e:
            logger.warning(f"  t-SNE failed: {e}")
    
    def _create_feature_profiles(self, cluster_col: str):
        """Create feature profiles by cluster."""
        key_features = [
            'Recency', 'Frequency', 'Monetary', 'Avg_Transaction_Value',
            'Tenure', 'Total_Items', 'Unique_Products'
        ]
        
        # filtr out noise
        profile_df = self.customer_df[self.customer_df[cluster_col] != -1].copy()
        
        # Normalize features for comparison
        normalized_profiles = pd.DataFrame()
        for col in key_features:
            if col in profile_df.columns:
                normalized_profiles[col] = (
                    (profile_df[col] - profile_df[col].min()) / 
                    (profile_df[col].max() - profile_df[col].min())
                )
        
        normalized_profiles['Cluster'] = profile_df[cluster_col].values
        
        # Calculate Cluster mean s
        cluster_means = normalized_profiles.groupby('Cluster')[key_features].mean()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='RdYlGn',
                    cbar_kws={'label': 'Normalized Value (0-1)'})
        plt.title(f'Feature Profiles by Cluster - {self.best_algorithm}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig('12_feature_profiles_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 12_feature_profiles_heatmap.png")
    
    def _create_cluster_distribution(self, cluster_col: str):
        """Create cluster distribution plots."""
        cluster_counts = self.customer_df[self.customer_df[cluster_col] != -1][cluster_col].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
        bars = ax1.bar(range(len(cluster_counts)), cluster_counts.values, color=colors)
        ax1.set_title(f'Customer Distribution - {self.best_algorithm}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cluster', fontsize=12)
        ax1.set_ylabel('Number of Customers', fontsize=12)
        ax1.set_xticks(range(len(cluster_counts)))
        ax1.set_xticklabels([f'Cluster {int(i)}' for i in cluster_counts.index])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add counts on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height):,}', ha='center', va='bottom')
        
        #pie chart
        ax2.pie(cluster_counts.values, labels=[f'Cluster {int(i)}' for i in cluster_counts.index],
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Cluster Proportion', fontsize=14, fontweight='bold')
        
        plt.suptitle('Cluster Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('13_cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 13_cluster_distribution.png")
    
    def _create_3d_rfm_plot(self, cluster_col: str):
        """Create 3D RFM scatter plot."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # sample if large
        if len(self.customer_df) > 5000:
            plot_df = self.customer_df.sample(5000, random_state=self.config['random_state'])
        else:
            plot_df = self.customer_df
        
        # Filter ou t noise
        plot_df = plot_df[plot_df[cluster_col] != -1]
        
        #Create Scatter plot
        scatter = ax.scatter(
            plot_df['Recency'],
            plot_df['Frequency'],
            plot_df['Monetary'],
            c=plot_df[cluster_col],
            cmap='tab20',
            alpha=0.7,
            s=30
        )
        
        ax.set_xlabel('Recency (days)', fontsize=11, labelpad=10)
        ax.set_ylabel('Frequency', fontsize=11, labelpad=10)
        ax.set_zlabel('Monetary ($)', fontsize=11, labelpad=10)
        ax.set_title(f'3D RFM Plot - {self.best_algorithm}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Cluster', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('14_3d_rfm_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 14_3d_rfm_plot.png")
    
    def _create_cluster_profiles_table(self, cluster_col: str):
        """Create detailed cluster profiles table."""
        # Filter out noise
        profile_df = self.customer_df[self.customer_df[cluster_col] != -1].copy()
        
        # calculate statistics for each cluster
        cluster_stats = profile_df.groupby(cluster_col).agg({
            'Recency': ['mean', 'std'],
            'Frequency': ['mean', 'std'],
            'Monetary': ['mean', 'std'],
            'Avg_Transaction_Value': ['mean', 'std'],
            'Total_Items': ['mean', 'std'],
            'Unique_Products': ['mean', 'std'],
            'CustomerID': 'count'
        })
        
        # flatten the Multi-level columns
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        
        # Create table Data
        table_data = []
        for cluster in sorted(cluster_stats.index):
            stats = cluster_stats.loc[cluster]
            table_data.append([
                f"Cluster {int(cluster)}",
                f"{int(stats['CustomerID_count']):,}",
                f"{stats['Recency_mean']:.1f} ± {stats['Recency_std']:.1f}",
                f"{stats['Frequency_mean']:.1f} ± {stats['Frequency_std']:.1f}",
                f"${stats['Monetary_mean']:,.0f} ± ${stats['Monetary_std']:,.0f}",
                f"{stats['Avg_Transaction_Value_mean']:.1f} ± {stats['Avg_Transaction_Value_std']:.1f}",
                f"{stats['Total_Items_mean']:.1f} ± {stats['Total_Items_std']:.1f}",
                f"{stats['Unique_Products_mean']:.1f} ± {stats['Unique_Products_std']:.1f}"
            ])
        
        #create visualization
        fig, ax = plt.subplots(figsize=(16, len(table_data) * 0.8 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        columns = ['Cluster', 'Customers', 'Recency (days)', 'Frequency', 
                  'Monetary ($)', 'Avg Transaction', 'Total Items', 'Unique Products']
        
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 2)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#2E8B57')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        #style alternating Rows
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(len(columns)):
                    table[(i, j)].set_facecolor('#f5f5f5')
        
        plt.title(f'Detailed Cluster Profiles - {self.best_algorithm}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('15_cluster_profiles_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 15_cluster_profiles_table.png")
    
    def _create_silhouette_analysis(self, cluster_col: str):
        """
        Create detailed silhouette analysis visualization showing per-cluster performance.
        This addresses the critical question: "Which clusters are well-separated?"
        """
        logger.info("Creating silhouette analysis visualization...")
        
        # Get Cluster label s
        cluster_labels = self.customer_df[cluster_col].values
        
        #filter out Noise if Present (DBSCAN)
        if -1 in cluster_labels:
            valid_mask = cluster_labels != -1
            data_for_silhouette = self.scaled_data[valid_mask]
            labels_for_silhouette = cluster_labels[valid_mask]
        else:
            data_for_silhouette = self.scaled_data
            labels_for_silhouette = cluster_labels
        
        #Calculate silhouette Values for each sample
        silhouette_vals = silhouette_samples(data_for_silhouette, labels_for_silhouette)
        
        #Get unique clusters
        unique_clusters = sorted(np.unique(labels_for_silhouette))
        n_clusters = len(unique_clusters)
        
        # Calculate overall Silhouette score
        silhouette_avg = silhouette_score(data_for_silhouette, labels_for_silhouette)
        
        # create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # ==LEFT PLOT: silhouette plot ===
        y_lower = 10
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for i, cluster in enumerate(unique_clusters):
            # Get Silhouette Values for this Cluster
            cluster_silhouette_vals = silhouette_vals[labels_for_silhouette == cluster]
            cluster_silhouette_vals.sort()
            
            size_cluster = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster
            
            # fill silhouette Plot
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_vals,
                facecolor=colors[i],
                edgecolor=colors[i],
                alpha=0.7,
                label=f'Cluster {int(cluster)}'
            )
            
            #Add cluster label
            ax1.text(-0.05, y_lower + 0.5 * size_cluster, f'{int(cluster)}', 
                    fontsize=10, fontweight='bold')
            
            y_lower = y_upper + 10
        
        # Add aVerage Silhouette line
        ax1.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
                   label=f'Average: {silhouette_avg:.3f}')
        
        # formatting
        ax1.set_xlabel('Silhouette Coefficient', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cluster', fontsize=12, fontweight='bold')
        ax1.set_title(f'Silhouette Plot by Cluster\n{self.best_algorithm}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlim([-0.3, 1.0])
        ax1.set_ylim([0, len(data_for_silhouette) + (n_clusters + 1) * 10])
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.legend(loc='lower right')
        
        # ===RIGHT PLOT: Summary table ==
        ax2.axis('tight')
        ax2.axis('off')
        
        # calculate per cluster statistics
        table_data = []
        for cluster in unique_clusters:
            cluster_vals = silhouette_vals[labels_for_silhouette == cluster]
            cluster_size = len(cluster_vals)
            cluster_avg = cluster_vals.mean()
            cluster_min = cluster_vals.min()
            cluster_max = cluster_vals.max()
            
            # Determine quality
            if cluster_avg >= 0.5:
                quality = "✓ Good"
                color = '#90EE90'  
            elif cluster_avg >= 0.3:
                quality = "⚠ Fair"
                color = '#FFFF99'  
            else:
                quality = "✗ Poor"
                color = '#FFB6C1'  
            
            table_data.append([
                f'Cluster {int(cluster)}',
                f'{cluster_size:,}',
                f'{cluster_avg:.3f}',
                f'{cluster_min:.3f}',
                f'{cluster_max:.3f}',
                quality,
                color
            ])
        
        # Add overall row
        table_data.append([
            'Overall',
            f'{len(data_for_silhouette):,}',
            f'{silhouette_avg:.3f}',
            f'{silhouette_vals.min():.3f}',
            f'{silhouette_vals.max():.3f}',
            'Average' if silhouette_avg >= 0.3 else 'Poor',
            '#D3D3D3'  # Light gray
        ])
        
        # Create table
        table_display = [[row[0], row[1], row[2], row[3], row[4], row[5]] 
                         for row in table_data]
        
        columns = ['Cluster', 'Size', 'Avg Silhouette', 'Min', 'Max', 'Quality']
        
        table = ax2.table(cellText=table_display, colLabels=columns, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style Header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # color code rows by quality
        for i, row in enumerate(table_data, 1):
            for j in range(len(columns)):
                table[(i, j)].set_facecolor(row[6])
        
        ax2.set_title('Cluster Quality Summary', fontsize=14, fontweight='bold')
        
        # add interpretation text
        interpretation = (
            f"Interpretation:\n"
            f"• Average Silhouette: {silhouette_avg:.3f} "
            f"({'Good' if silhouette_avg >= 0.5 else 'Moderate' if silhouette_avg >= 0.3 else 'Poor'})\n"
            f"• Clusters > 0.5: Well-separated\n"
            f"• Clusters 0.3-0.5: Overlapping but acceptable\n"
            f"• Clusters < 0.3: Poorly separated\n"
            f"• Negative values: Possibly misclassified"
        )
        
        fig.text(0.5, 0.02, interpretation, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Silhouette Analysis - Per-Cluster Performance Evaluation', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        plt.savefig('15a_silhouette_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 15a_silhouette_analysis.png")
    
    # ========================================================================
    # PHASE 9: BUSINESS INTERPRETATON
    # ========================================================================
    def interpret_clusters(self) -> Dict:
        """
        Provide business interpretation of clusters.
        
        Returns:
            Dictionary with cluster interpretations
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 9: BUSINESS INTERPRETATION")
        logger.info("="*70)
        
        if self.best_algorithm is None:
            logger.error("No clustering results to interpret")
            return {}
        
        cluster_col = f'{self.best_algorithm}_Cluster'
        interpretation_df = self.customer_df[self.customer_df[cluster_col] != -1].copy()
        
        # calculate cluster Statistics
        cluster_stats = interpretation_df.groupby(cluster_col).agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Avg_Transaction_Value': 'mean',
            'Tenure': 'mean',
            'Unique_Products': 'mean',
            'CustomerID': 'count'
        }).rename(columns={'CustomerID': 'Count'})
        
        overall_stats = interpretation_df.agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        })
        
        #interpret each cluster
        segment_descriptions = {}
        
        for cluster in cluster_stats.index:
            stats = cluster_stats.loc[cluster]
            count = stats['Count']
            percentage = (count / len(interpretation_df)) * 100
            
            #calculate scores
            recency_score = 1 - min(stats['Recency'] / 365, 1)
            frequency_score = min(stats['Frequency'] / cluster_stats['Frequency'].max(), 1)
            monetary_score = min(stats['Monetary'] / cluster_stats['Monetary'].max(), 1)
            
            value_score = 0.4 * recency_score + 0.3 * frequency_score + 0.3 * monetary_score
            
            # Determine segment type
            if value_score > 0.7:
                segment_name = "🏆 Champions"
                description = "High-value, frequent, recent customers"
                strategy = "VIP treatment, premium offers, loyalty programs"
                priority = 1
            elif stats['Recency'] < overall_stats['Recency'] and stats['Monetary'] > overall_stats['Monetary']:
                segment_name = "💰 High-Potential"
                description = "Recent customers with high spending"
                strategy = "Upsell, cross-sell, personalized recommendations"
                priority = 2
            elif stats['Frequency'] > overall_stats['Frequency']:
                segment_name = "🤝 Loyal Customers"
                description = "Frequent but moderate spending"
                strategy = "Loyalty rewards, subscription models"
                priority = 3
            elif stats['Recency'] < overall_stats['Recency']:
                segment_name = "🌱 New Customers"
                description = "Recent but low engagement"
                strategy = "Onboarding, education, welcome offers"
                priority = 4
            elif stats['Recency'] > 365:
                segment_name = "💔 At Risk"
                description = "Inactive for over a year"
                strategy = "Win-back campaigns, special offers"
                priority = 5
            else:
                segment_name = f"📊 Standard"
                description = "Average customers with balanced metrics"
                strategy = "General marketing, value-based offers"
                priority = 6
            
            segment_descriptions[cluster] = {
                'segment_name': segment_name,
                'description': description,
                'strategy': strategy,
                'priority': priority,
                'value_score': value_score,
                'count': int(count),
                'percentage': percentage,
                'avg_recency': float(stats['Recency']),
                'avg_frequency': float(stats['Frequency']),
                'avg_monetary': float(stats['Monetary'])
            }
            
            # add segment Information to dataframe
            self.customer_df.loc[
                self.customer_df[cluster_col] == cluster,
                f'{self.best_algorithm}_Segment'
            ] = segment_name
        
        # create business interpretation Visualization
        self._create_business_interpretation_visualization(segment_descriptions)
        
        # create action Plan visualization
        self._create_action_plan_visualization(segment_descriptions)
        
        return segment_descriptions
    
    def _create_business_interpretation_visualization(self, segment_descriptions: Dict):
        """Create visualization of business interpretations."""
        if not segment_descriptions:
            return
        
        # prepare data f0r visualization
        clusters = []
        segment_names = []
        descriptions = []
        strategies = []
        priorities = []
        value_scores = []
        customer_counts = []
        percentages = []
        
        for cluster, info in sorted(segment_descriptions.items()):
            clusters.append(f"Cluster {int(cluster)}")
            segment_names.append(info['segment_name'])
            descriptions.append(info['description'])
            strategies.append(info['strategy'])
            priorities.append(info['priority'])
            value_scores.append(info['value_score'])
            customer_counts.append(f"{info['count']:,}")
            percentages.append(f"{info['percentage']:.1f}%")
        
        # create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        
        # 1.Segment summary Table
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax1.axis('tight')
        ax1.axis('off')
        
        table_data = list(zip(clusters, segment_names, descriptions, strategies, 
                            [str(p) for p in priorities], 
                            [f"{v:.3f}" for v in value_scores],
                            customer_counts, percentages))
        
        columns = ['Cluster', 'Segment Name', 'Description', 'Strategy', 
                  'Priority', 'Value Score', 'Customers', 'Percentage']
        
        table = ax1.table(cellText=table_data, colLabels=columns, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.1, 1.8)
        
        #Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style By priority
        priority_colors = {
            1: '#FFD700',  # Gold for highest priority
            2: '#FFA500',  # Orange
            3: '#90EE90',  # Light green
            4: '#ADD8E6',  # Light blue
            5: '#FFB6C1',  # Light pink
            6: '#D3D3D3'   # Light gray
        }
        
        for i, priority in enumerate(priorities, 1):
            color = priority_colors.get(priority, '#FFFFFF')
            for j in range(len(columns)):
                table[(i, j)].set_facecolor(color)
        
        ax1.set_title('Business Segment Interpretation', fontsize=12, fontweight='bold')
        
        #2. Value score comparison
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        bars = ax2.barh(clusters, value_scores, color=[priority_colors[p] for p in priorities])
        ax2.set_xlabel('Value Score (0-1)')
        ax2.set_title('Customer Value by Segment', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, value in zip(bars, value_scores):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center')
        
        # 3.Customer Distribution
        ax3 = plt.subplot2grid((2, 2), (1, 1))
        sizes = [info['count'] for info in segment_descriptions.values()]
        labels = [info['segment_name'] for info in segment_descriptions.values()]
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%',
                                          startangle=90, colors=[priority_colors[p] for p in priorities])
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        ax3.set_title('Customer Distribution by Segment', fontsize=11, fontweight='bold')
        
        plt.suptitle('Business Interpretation & Action Plan', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('16_business_interpretation.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 16_business_interpretation.png")
    
    def _create_action_plan_visualization(self, segment_descriptions: Dict):
        """Create action plan visualization."""
        if not segment_descriptions:
            return
        
        # Group segments by Priority
        priority_groups = {}
        for cluster, info in segment_descriptions.items():
            priority = info['priority']
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(info)
        
        # create Action plan table
        fig, ax = plt.subplots(figsize=(14, len(segment_descriptions) * 0.8 + 3))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for priority in sorted(priority_groups.keys()):
            for segment in priority_groups[priority]:
                table_data.append([
                    f"Priority {priority}",
                    segment['segment_name'],
                    f"{segment['count']:,} customers",
                    segment['strategy']
                ])
        
        columns = ['Priority', 'Segment', 'Size', 'Recommended Actions']
        
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#2E8B57')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style by priority
        priority_colors = {
            1: '#FFD700',  # Gold
            2: '#FFA500',  # Orange
            3: '#90EE90',  # Green
            4: '#ADD8E6',  # Blue
            5: '#FFB6C1',  # Pink
            6: '#D3D3D3'   # Gray
        }
        
        current_priority = None
        row_idx = 1
        for priority in sorted(priority_groups.keys()):
            for _ in priority_groups[priority]:
                color = priority_colors.get(priority, '#FFFFFF')
                for j in range(len(columns)):
                    table[(row_idx, j)].set_facecolor(color)
                row_idx += 1
        
        plt.title('Action Plan by Priority Level', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('17_action_plan.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 17_action_plan.png")
    
    # ========================================================================
    # PHASE 10: EXPORT RESULTS
    # ========================================================================
    def export_results(self):
        """Export all analysis results as visualizations."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 10: EXPORTING RESULTS")
        logger.info("="*70)
        
        if self.customer_df is None:
            logger.error("No data to export")
            return
        
        # 1. create customer segment assignment table (sample)
        logger.info("Creating customer segment assignment sample...")
        self._create_customer_segment_sample()
        
        # 2. creete executive summary visualization
        logger.info("Creating executive summary...")
        self._create_executive_summary_visualization()
        
        logger.info("\n" + "="*70)
        logger.info("✓ ALL RESULTS EXPORTED SUCCESSFULLY")
        logger.info("="*70)
        
        logger.info("\n📋 Generated Visual Files:")
        files = [
            "01_summary_statistics.png",
            "02_feature_distributions.png",
            "03_correlation_heatmap.png",
            "04_outlier_detection.png",
            "05_rfm_pairplot.png",
            "06_outlier_removal_impact.png",
            "07_cluster_evaluation_metrics.png",
            "08_algorithm_comparison.png",
            "09_statistical_validation.png",
            "10_pca_visualization.png",
            "11_tsne_visualization.png",
            "12_feature_profiles_heatmap.png",
            "13_cluster_distribution.png",
            "14_3d_rfm_plot.png",
            "15_cluster_profiles_table.png",
            "15a_silhouette_analysis.png",  # 🔥 ADDED: Silhouette analysis
            "16_business_interpretation.png",
            "17_action_plan.png",
            "18_customer_segments_sample.png",
            "19_executive_summary.png"
        ]
        
        for file in files:
            logger.info(f"  • {file}")
    
    def _create_customer_segment_sample(self):
        """Create sample of customer segment assignments."""
        if self.best_algorithm is None:
            return
        
        # Take sample of customers
        sample_size = min(20, len(self.customer_df))
        sample_df = self.customer_df.sample(sample_size, random_state=self.config['random_state']).copy()
        
        # Select columns to display
        display_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        if f'{self.best_algorithm}_Cluster' in sample_df.columns:
            display_columns.append(f'{self.best_algorithm}_Cluster')
        if f'{self.best_algorithm}_Segment' in sample_df.columns:
            display_columns.append(f'{self.best_algorithm}_Segment')
        
        # Format monetary values
        if 'Monetary' in sample_df.columns:
            sample_df['Monetary'] = sample_df['Monetary'].apply(lambda x: f"${x:,.2f}")
        
        # Create table
        fig, ax = plt.subplots(figsize=(12, sample_size * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for _, row in sample_df.iterrows():
            table_row = []
            for col in display_columns:
                value = row[col]
                if isinstance(value, float):
                    if col == 'Recency':
                        table_row.append(f"{value:.0f}")
                    elif col == 'Frequency':
                        table_row.append(f"{value:.1f}")
                    else:
                        table_row.append(f"{value:.2f}")
                else:
                    table_row.append(str(value))
            table_data.append(table_row)
        
        table = ax.table(cellText=table_data, colLabels=display_columns, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style header
        for i in range(len(display_columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style alternating rows
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(len(display_columns)):
                    table[(i, j)].set_facecolor('#f5f5f5')
        
        plt.title(f'Customer Segment Assignments (Sample of {sample_size})', 
                 fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('18_customer_segments_sample.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 18_customer_segments_sample.png")
    
    def _create_executive_summary_visualization(self):
        """Create comprehensive executive summary visualization."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid for the summary
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. project Overview
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('tight')
        ax1.axis('off')
        
        overview_data = [
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Dataset', self.filepath.split('/')[-1]],
            ['Total Customers', f"{self.data_stats.get('clean_customers', 0):,}"],
            ['Total Transactions', f"{self.data_stats.get('clean_rows', 0):,}"],
            ['Transaction Retention', f"{self.data_stats.get('transaction_retention_rate', 0):.1f}%"],
            ['Customer Retention', f"{self.data_stats.get('customer_retention_rate', 0):.1f}%"],
            ['Best Algorithm', self.best_algorithm if self.best_algorithm else 'N/A']
        ]
        
        if self.best_algorithm and self.best_algorithm in self.results:
            best_result = self.results[self.best_algorithm]
            overview_data.extend([
                ['Optimal Clusters', f"{best_result['n_clusters']}"],
                ['Silhouette Score', f"{best_result['silhouette']:.4f}"],
                ['Cluster Quality', 'Good' if best_result['silhouette'] > 0.5 else 
                 'Moderate' if best_result['silhouette'] > 0.3 else 'Poor']
            ])
        
        table1 = ax1.table(cellText=overview_data, cellLoc='left', loc='center')
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1, 2)
        
        for i in range(len(overview_data)):
            for j in range(2):
                if i % 2 == 0:
                    table1[(i, j)].set_facecolor('#f5f5f5')
        
        ax1.set_title('Project Overview', fontsize=12, fontweight='bold')
        
        # 2. Key findings
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('tight')
        ax2.axis('off')
        
        findings_data = []
        if self.best_algorithm and f'{self.best_algorithm}_Segment' in self.customer_df.columns:
            segments = self.customer_df[f'{self.best_algorithm}_Segment'].value_counts()
            for segment, count in segments.items():
                percentage = (count / len(self.customer_df)) * 100
                findings_data.append([segment, f"{count:,} ({percentage:.1f}%)"])
        
        if findings_data:
            table2 = ax2.table(cellText=findings_data, 
                              colLabels=['Segment', 'Customers'], 
                              cellLoc='left', loc='center')
            table2.auto_set_font_size(False)
            table2.set_fontsize(10)
            table2.scale(1, 2)
            
            for i in range(len(findings_data) + 1):
                for j in range(2):
                    if i == 0:
                        table2[(i, j)].set_facecolor('#2E8B57')
                        table2[(i, j)].set_text_props(weight='bold', color='white')
                    elif i % 2 == 1:
                        table2[(i, j)].set_facecolor('#f5f5f5')
        
        ax2.set_title('Segment Summary', fontsize=12, fontweight='bold')
        
        # 3. Recommendations
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('tight')
        ax3.axis('off')
        
        recommendations = [
            "1. Focus on high-value segments (Champions, High-Potential) for retention and upsell",
            "2. Develop targeted campaigns for each segment based on their characteristics",
            "3. Monitor segment migration over time (quarterly reviews recommended)",
            "4. Implement A/B testing for different marketing strategies",
            "5. Use personalized recommendations for loyal customers",
            "6. Create win-back campaigns for at-risk customers",
            "7. Track ROI by segment to optimize marketing spend",
            "8. Re-run analysis quarterly to track customer behavior changes"
        ]
        
        rec_data = [[rec] for rec in recommendations]
        table3 = ax3.table(cellText=rec_data, cellLoc='left', loc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1, 1.5)
        
        for i in range(len(recommendations)):
            table3[(i, 0)].set_facecolor('#e6f3ff')
        
        ax3.set_title('Key Recommendations', fontsize=12, fontweight='bold')
        
        # 4. Technical Summary
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        
        tech_data = [
            ['Features Used', f"{len(self.feature_columns)}"],
            ['Algorithms Tested', f"{len(self.results)}"],
            ['Outliers Removed', f"{self.data_stats.get('outliers_removed', 0):,}"],
            ['Customer Retention', f"{self.data_stats.get('customer_retention_rate', 0):.1f}%"],
            ['Transaction Retention', f"{self.data_stats.get('transaction_retention_rate', 0):.1f}%"],
            ['Analysis Duration', 'See log file for details'],
            ['Generated Files', f"{20} visualizations and tables"]  # Changed from 19 to 20
        ]
        
        table4 = ax4.table(cellText=tech_data, cellLoc='left', loc='center')
        table4.auto_set_font_size(False)
        table4.set_fontsize(10)
        table4.scale(1, 2)
        
        for i in range(len(tech_data)):
            for j in range(2):
                if i % 2 == 0:
                    table4[(i, j)].set_facecolor('#f5f5f5')
        
        ax4.set_title('Technical Summary', fontsize=12, fontweight='bold')
        
        plt.suptitle('CUSTOMER SEGMENTATION ANALYSIS - EXECUTIVE SUMMARY', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('19_executive_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: 19_executive_summary.png")
    
    # ========================================================================
    # MAIN EXECUTION METHOD.
    # ========================================================================
    def run_analysis(self):
        """Execute the complete customer segmentation pipeline."""
        logger.info("\n" + "="*80)
        logger.info("           CUSTOMER SEGMENTATION ANALYSIS")
        logger.info("           COMPLETE PIPELINE - VISUAL OUTPUTS ONLY")
        logger.info("="*80)
        
        start_time = datetime.now()
        logger.info(f"⏰ Analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Phase 1: data Loading & Cleaning
            logger.info("\n[PHASE 1] Loading and cleaning data...")
            df = self.load_and_clean_data()
            if df is None:
                logger.error("✗ Data loading failed. Analysis aborted.")
                return False
            
            # Phase 2: Feature engineering
            logger.info("\n[PHASE 2] Engineering features...")
            customer_features = self.engineer_features(df)
            
            # Phase 3: exploratory Data Analysis
            logger.info("\n[PHASE 3] Performing EDA...")
            self.perform_eda(customer_features)
            
            # Phase 4: Data Pre processing
            logger.info("\n[PHASE 4] Preprocessing data...")
            self.preprocess_data(customer_features)
            
            # Phase 5: Optiml Cluster determination
            logger.info("\n[PHASE 5] Finding optimal clusters...")
            optimal_k = self.find_optimal_clusters()
            
            # Phase 6: clustering algorithms
            logger.info("\n[PHASE 6] Applying clustering algorithms...")
            self.apply_clustering_algorithms(optimal_k)
            
            # Phase 7: Statistical validation
            logger.info("\n[PHASE 7] Validating clusters statistically...")
            self.validate_clusters_statistically()
            
            # Phase 8: visualization
            logger.info("\n[PHASE 8] Creating visualizations...")
            self.visualize_clusters()
            
            # Phase 9: Business interpretation
            logger.info("\n[PHASE 9] Interpreting clusters for business...")
            self.interpret_clusters()
            
            # Phase 10: Export Results
            logger.info("\n[PHASE 10] Exporting results...")
            self.export_results()
            
            # calculate durationn
            end_time = datetime.now()
            duration = end_time - start_time
            
            #  the final summary
            logger.info("\n" + "="*80)
            logger.info("           ANALYSIS COMPLETE - FINAL SUMMARY")
            logger.info("="*80)
            
            logger.info(f"\n⏱️  Duration: {duration.total_seconds():.1f} seconds")
            logger.info(f"📊 Best Algorithm: {self.best_algorithm}")
            
            if self.best_algorithm and self.best_algorithm in self.results:
                best_result = self.results[self.best_algorithm]
                logger.info(f"🎯 Silhouette Score: {best_result['silhouette']:.4f}")
                logger.info(f"🔢 Number of Clusters: {best_result['n_clusters']}")
            
            logger.info(f"👥 Customers Analyzed: {len(self.customer_df):,}")
            logger.info(f"📈 Features Used: {len(self.feature_columns)}")
            logger.info(f"📊 Customer Retention: {self.data_stats.get('customer_retention_rate', 0):.1f}%")
            logger.info(f"📊 Transaction Retention: {self.data_stats.get('transaction_retention_rate', 0):.1f}%")
            logger.info(f"🖼️  Visualizations Generated: 20 files")
            
            logger.info("\n✓ Analysis completed successfully!")
            logger.info("📁 Check the generated PNG files for complete visual results.")
            
            return True
            
        except Exception as e:
            logger.error(f"\n✗ Analysis failed with error: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return False


# ========================================================================
# MAIN EXECUTION
# ========================================================================
if __name__ == "__main__":
    import sys
    import argparse
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Customer Segmentation Analysis')
    parser.add_argument('--filepath', type=str, required=True,
                       help='Path to dataset file (CSV or Excel)')
    parser.add_argument('--max_clusters', type=int, default=8,
                       help='Maximum number of clusters to evaluate')
    
    args = parser.parse_args()
    
    # create & run analyzer
    analyzer = CustomerSegmentation(args.filepath)
    analyzer.config['max_clusters'] = args.max_clusters
    
    success = analyzer.run_analysis()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)