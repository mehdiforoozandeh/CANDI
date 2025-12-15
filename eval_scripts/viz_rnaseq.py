#!/usr/bin/env python3
"""
RNA-seq Evaluation Script

This script evaluates RNA-seq prediction performance using direct model predictions
(no NPZ loading) with consistent feature extraction strategies and nested cross-validation.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from xgboost import XGBRegressor

# Suppress all warnings (convergence, constant input, etc.)
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pred import CANDIPredictor
from data import CANDIDataHandler
from _utils import load_gene_coords, signal_feature_extraction


class RNASeqEvaluator:
    """Evaluator for RNA-seq prediction performance using CANDI model predictions."""
    
    def __init__(self, model_dir: str, data_path: str, resolution: int = 25):
        """
        Initialize RNA-seq evaluator.
        
        Args:
            model_dir: Path to model directory
            data_path: Path to dataset directory
            resolution: Genomic resolution in bp
        """
        self.model_dir = Path(model_dir)
        self.data_path = Path(data_path)
        self.resolution = resolution
        
        # Initialize predictor and data handler
        self.predictor = CANDIPredictor(str(self.model_dir))
        self.predictor.setup_data_handler(str(self.data_path), dataset_type="merged", split="test")
        self.dataset = self.predictor.data_handler
        
        # Reload chromosome sizes with test mode to include chr21
        self.dataset._load_genomic_coords(mode="test")
        
        # Load gene coordinates
        gene_coords_file = Path(__file__).parent.parent / "data" / "parsed_genecode_data_hg38_release42.csv"
        if not gene_coords_file.exists():
            raise FileNotFoundError(f"Gene coordinates file not found: {gene_coords_file}")
        self.gene_coords = load_gene_coords(str(gene_coords_file))
        # Filter to chr21 (same as old_eval.py)
        self.gene_coords = self.gene_coords[self.gene_coords["chr"] == "chr21"].reset_index(drop=True)
        
        # Get experiment names
        self.expnames = list(self.dataset.aliases["experiment_aliases"].keys())
        self.n_assays = len(self.expnames)
        
        print(f"Initialized RNA-seq evaluator with {self.n_assays} assays")
    
    def extract_pval_features(self, pval_data: np.ndarray, genes: pd.DataFrame, 
                             assay_indices: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Extract features from p-value data using signal_feature_extraction.
        
        Args:
            pval_data: P-value array [L, F] in arcsinh space (kept in arcsinh space)
            genes: DataFrame with gene coordinates (geneID, start, end, strand)
            assay_indices: Optional list of assay indices to use (default: all)
            
        Returns:
            DataFrame with features (rows=genes, columns=features)
        """
        features_list = []
        
        if assay_indices is None:
            assay_indices = list(range(pval_data.shape[1]))
        
        for _, row in genes.iterrows():
            gene, start, end, strand = row['geneID'], row['start'], row['end'], row['strand']
            
            # Compute adaptive margins: 10% of gene length (5% upstream + 5% downstream)
            gene_length = end - start
            adaptive_margin = int(gene_length * 0.1)  # 10% of gene length
            
            for idx, a in enumerate(assay_indices):
                if a >= len(self.expnames):
                    continue
                assay = self.expnames[a]
                signal = pval_data[:, idx]  # Keep in arcsinh space
                
                # Extract features with adaptive margins (in arcsinh space)
                feats = signal_feature_extraction(start, end, strand, signal, 
                                                 bin_size=self.resolution, 
                                                 margin_tss=adaptive_margin,
                                                 margin_tes=adaptive_margin)
                
                for suffix, val in feats.items():
                    features_list.append({
                        'geneID': gene,
                        'feature': f"{assay}_{suffix}",
                        'signal': val
                    })
        
        df_long = pd.DataFrame(features_list)
        if len(df_long) == 0:
            return pd.DataFrame()
        
        df_wide = df_long.pivot_table(
            index='geneID', columns='feature', values='signal', aggfunc='mean'
        ).fillna(0)
        
        return df_wide
    
    def extract_latent_features(self, z_data: np.ndarray, genes: pd.DataFrame, 
                               signal_length: int) -> pd.DataFrame:
        """
        Extract features from latent Z representations.
        
        Args:
            z_data: Latent array [L_z, D] where L_z < signal_length
            genes: DataFrame with gene coordinates
            signal_length: Length of original signal (for resolution calculation)
            
        Returns:
            DataFrame with features (rows=genes, columns=features)
        """
        def stats(x):
            """
            Compute summary statistics for each latent dimension in a region.
            Returns median, IQR, mean, std, min, max arrays of shape [D].
            """
            D = z_data.shape[1]
            if x.size == 0:
                zeros = np.zeros(D)
                return zeros, zeros, zeros, zeros, zeros, zeros
            med = np.nanmedian(x, axis=0)
            q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
            iqr = q75 - q25
            mean = np.nanmean(x, axis=0)
            std = np.nanstd(x, axis=0)
            mn = np.nanmin(x, axis=0)
            mx = np.nanmax(x, axis=0)
            return med, iqr, mean, std, mn, mx
        
        # Calculate resolution ratio
        y2z_resolution_ratio = signal_length / z_data.shape[0]
        bp2z_ratio = self.resolution * y2z_resolution_ratio
        
        features_list = []
        
        for _, row in genes.iterrows():
            gene, start, end, strand = row['geneID'], row['start'], row['end'], row['strand']
            
            # Fixed margin: 2kb (2000 bp)
            gene_length = end - start
            adaptive_margin = int(gene_length * 0.1)
            
            # Map genomic coordinates to Z indices
            z_start = int(start // bp2z_ratio)
            z_end = int(end // bp2z_ratio)
            
            # Extract regions
            tss = start if strand == '+' else end
            tes = end if strand == '+' else start
            
            tss_z = int(tss // bp2z_ratio)
            tes_z = int(tes // bp2z_ratio)
            
            # Convert margin from bp to Z bins
            margin_z = int(adaptive_margin // bp2z_ratio)
            
            TSS_z = z_data[max(0, tss_z - margin_z):
                           min(z_data.shape[0], tss_z + margin_z)]
            gene_z = z_data[max(0, z_start):min(z_data.shape[0], z_end)]
            TTS_z = z_data[max(0, tes_z - margin_z):
                           min(z_data.shape[0], tes_z + margin_z)]
            
            # Compute stats for each region
            (
                gene_med, gene_iqr, gene_mean, gene_std, gene_min, gene_max
            ) = stats(gene_z)
            (
                tss_med, tss_iqr, tss_mean, tss_std, tss_min, tss_max
            ) = stats(TSS_z)
            (
                tts_med, tts_iqr, tts_mean, tts_std, tts_min, tts_max
            ) = stats(TTS_z)
            
            # Extract features per latent dimension:
            # median, IQR, mean, std, min, max for TSS / gene body / TES
            for j in range(z_data.shape[1]):
                # Gene body
                features_list.append({'geneID': gene, 'feature': f"Z_gene_med_f{j}",  'signal': gene_med[j]})
                features_list.append({'geneID': gene, 'feature': f"Z_gene_iqr_f{j}",  'signal': gene_iqr[j]})
                # features_list.append({'geneID': gene, 'feature': f"Z_gene_mean_f{j}", 'signal': gene_mean[j]})
                # features_list.append({'geneID': gene, 'feature': f"Z_gene_std_f{j}",  'signal': gene_std[j]})
                features_list.append({'geneID': gene, 'feature': f"Z_gene_min_f{j}",  'signal': gene_min[j]})
                features_list.append({'geneID': gene, 'feature': f"Z_gene_max_f{j}",  'signal': gene_max[j]})
                # TSS
                features_list.append({'geneID': gene, 'feature': f"Z_tss_med_f{j}",   'signal': tss_med[j]})
                features_list.append({'geneID': gene, 'feature': f"Z_tss_iqr_f{j}",   'signal': tss_iqr[j]})
                # features_list.append({'geneID': gene, 'feature': f"Z_tss_mean_f{j}",  'signal': tss_mean[j]})
                # features_list.append({'geneID': gene, 'feature': f"Z_tss_std_f{j}",   'signal': tss_std[j]})
                features_list.append({'geneID': gene, 'feature': f"Z_tss_min_f{j}",   'signal': tss_min[j]})
                features_list.append({'geneID': gene, 'feature': f"Z_tss_max_f{j}",   'signal': tss_max[j]})
                # TES
                features_list.append({'geneID': gene, 'feature': f"Z_tts_med_f{j}",   'signal': tts_med[j]})
                features_list.append({'geneID': gene, 'feature': f"Z_tts_iqr_f{j}",   'signal': tts_iqr[j]})
                # features_list.append({'geneID': gene, 'feature': f"Z_tts_mean_f{j}",  'signal': tts_mean[j]})
                # features_list.append({'geneID': gene, 'feature': f"Z_tts_std_f{j}",   'signal': tts_std[j]})
                features_list.append({'geneID': gene, 'feature': f"Z_tts_min_f{j}",   'signal': tts_min[j]})
                features_list.append({'geneID': gene, 'feature': f"Z_tts_max_f{j}",   'signal': tts_max[j]})
        
        df_long = pd.DataFrame(features_list)
        if len(df_long) == 0:
            return pd.DataFrame()
        
        df_wide = df_long.pivot_table(
            index='geneID', columns='feature', values='signal', aggfunc='mean'
        ).fillna(0)
        
        return df_wide
    
    def run_nested_cv(self, X: np.ndarray, y: np.ndarray, n_jobs: int = -1,
                      model: str = "linear", feature_selection: str = "pca") -> Dict[str, float]:
        """
        Run nested cross-validation with StandardScaler + feature selection + regression model.
        
        Args:
            X: Feature matrix [n_genes, n_features]
            y: Target values [n_genes] (arcsinh-transformed TPM)
            n_jobs: Number of parallel jobs
            model: Regression model type
            feature_selection: Feature selection method ('pca' or 'ftest')
            
        Returns:
            Dictionary with metrics (pearson, spearman, r2, mse)
        """
        # Outer CV: 5-fold split
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Inner CV: GridSearchCV for hyperparameters
        inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)
        
        # Setup feature selection and hyperparameter grid
        if feature_selection == "ftest":
            # F-test based feature selection
            n_features = X.shape[1]
            # Try different k values: 10%, 25%, 50%, 75%, 90% of features, or 'all'
            k_values = [
                max(1, int(n_features * 0.1)),
                max(1, int(n_features * 0.25)),
                max(1, int(n_features * 0.5)),
                max(1, int(n_features * 0.75)),
                max(1, int(n_features * 0.9)),
                'all'
            ]
            # Remove duplicates and sort
            k_values = sorted(list(set([k for k in k_values if k != 'all']))) + ['all']
            
            feature_selector = SelectKBest(score_func=f_regression)
            param_grid = {
                'feature_select__k': k_values
            }
        else:  # Default to PCA
            # PCA with explained variance ratio
            feature_selector = PCA()
            param_grid = {
                'feature_select__n_components': [0.5, 0.7, 0.8, 0.9]
            }
        
        # Choose regression model
        if model == "linear":
            reg = LinearRegression()
        elif model == "ridge":
            reg = Ridge(alpha=1.0)
        elif model == "lasso":
            reg = Lasso(alpha=1.0, max_iter=10000)
        elif model == "svr":
            reg = SVR(kernel='rbf', C=1.0, epsilon=0.2)
        elif model == "xgb":
            reg = XGBRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=n_jobs,
                verbosity=0
            )
        elif model in ("rf", "random_forest", "randomforest"):
            reg = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=n_jobs
            )
        else:
            raise ValueError(f"Unknown model type: {model}")
        
        # Build pipeline: StandardScaler + feature_select + chosen regressor
        pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('feature_select', feature_selector),
            ('reg', reg)
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=inner_cv, 
            scoring='neg_mean_squared_error', n_jobs=n_jobs, verbose=0
        )
        
        all_metrics = {'pearson': [], 'spearman': [], 'r2': [], 'mse': []}
        
        # Outer CV loop
        for train_idx, test_idx in outer_cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit and predict
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.predict(X_test)
            
            # Compute metrics
            all_metrics['pearson'].append(pearsonr(y_test, y_pred)[0])
            all_metrics['spearman'].append(spearmanr(y_test, y_pred)[0])
            all_metrics['r2'].append(r2_score(y_test, y_pred))
            all_metrics['mse'].append(mean_squared_error(y_test, y_pred))
        
        # Return mean metrics across all folds
        return {
            'pearson': np.mean(all_metrics['pearson']),
            'spearman': np.mean(all_metrics['spearman']),
            'r2': np.mean(all_metrics['r2']),
            'mse': np.mean(all_metrics['mse'])
        }
    
    def evaluate_per_assay(self, bios_name: str, P_obs: np.ndarray, P_den: np.ndarray,
                          P_impden: np.ndarray, rna_seq_data: pd.DataFrame,
                          available_indices: List[int], n_jobs: int = -1) -> pd.DataFrame:
        """
        Evaluate RNA-seq prediction performance for each assay individually.
        
        Args:
            bios_name: Biosample name
            P_obs: Observed P-val data [L, F_avail] - available assays only
            P_den: Denoised P-val data [L, F_avail] - available assays only
            P_impden: Imputed+Denoised P-val data [L, 35] - all assays
            rna_seq_data: RNA-seq data with geneID and TPM
            available_indices: List of available assay indices
            n_jobs: Number of parallel jobs
            
        Returns:
            DataFrame with per-assay results
        """
        # Prepare target (log1p of TPM)
        gene_info = (
            rna_seq_data[['geneID', 'TPM']]
            .drop_duplicates(subset='geneID')
            .set_index('geneID')
        )
        y_series = np.log1p(gene_info['TPM'])
        
        # Setup CV
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Define scoring functions
        scoring = {
            'pearson': make_scorer(lambda y_true, y_pred: pearsonr(y_true, y_pred)[0]),
            'spearman': make_scorer(lambda y_true, y_pred: spearmanr(y_true, y_pred)[0]),
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error'
        }
        
        results_list = []
        
        # Process each assay (0 to 34)
        for assay_idx in range(self.n_assays):
            assay_name = self.expnames[assay_idx]
            
            # Determine which categories to evaluate for this assay
            is_available = assay_idx in available_indices
            
            if is_available:
                # Get the position in the available assays array
                avail_pos = available_indices.index(assay_idx)
                
                # Evaluate observed - extract the specific column for this assay
                P_obs_single = P_obs[:, avail_pos:avail_pos+1]  # Shape: [L, 1]
                X_obs = self.extract_pval_features(P_obs_single, rna_seq_data, assay_indices=[assay_idx])
                if not X_obs.empty:
                    common_genes = X_obs.index.intersection(y_series.index)
                    if len(common_genes) > 10:  # Need enough genes for CV
                        X_aligned = X_obs.loc[common_genes].values
                        y_aligned = y_series.loc[common_genes].values
                        
                        pipeline = Pipeline([
                            ('scale', StandardScaler()),
                            ('reg', LinearRegression())
                        ])
                        
                        try:
                            scores = cross_validate(pipeline, X_aligned, y_aligned, 
                                                   cv=cv, scoring=scoring, n_jobs=n_jobs)
                            results_list.append({
                                'biosample': bios_name,
                                'assay_idx': assay_idx,
                                'assay_name': assay_name,
                                'category': 'observed',
                                'pearson': np.mean(scores['test_pearson']),
                                'spearman': np.mean(scores['test_spearman']),
                                'r2': np.mean(scores['test_r2']),
                                'mse': -np.mean(scores['test_neg_mse'])
                            })
                        except Exception as e:
                            print(f"  Warning: Failed to evaluate observed {assay_name}: {e}")
                
                # Evaluate denoised - extract the specific column for this assay
                P_den_single = P_den[:, avail_pos:avail_pos+1]  # Shape: [L, 1]
                X_den = self.extract_pval_features(P_den_single, rna_seq_data, assay_indices=[assay_idx])
                if not X_den.empty:
                    common_genes = X_den.index.intersection(y_series.index)
                    if len(common_genes) > 10:
                        X_aligned = X_den.loc[common_genes].values
                        y_aligned = y_series.loc[common_genes].values
                        
                        pipeline = Pipeline([
                            ('scale', StandardScaler()),
                            ('reg', LinearRegression())
                        ])
                        
                        try:
                            scores = cross_validate(pipeline, X_aligned, y_aligned,
                                                   cv=cv, scoring=scoring, n_jobs=n_jobs)
                            results_list.append({
                                'biosample': bios_name,
                                'assay_idx': assay_idx,
                                'assay_name': assay_name,
                                'category': 'denoised',
                                'pearson': np.mean(scores['test_pearson']),
                                'spearman': np.mean(scores['test_spearman']),
                                'r2': np.mean(scores['test_r2']),
                                'mse': -np.mean(scores['test_neg_mse'])
                            })
                        except Exception as e:
                            print(f"  Warning: Failed to evaluate denoised {assay_name}: {e}")
            else:
                # Evaluate imputed (only for assays NOT in available set) - extract the specific column
                P_imp_single = P_impden[:, assay_idx:assay_idx+1]  # Shape: [L, 1]
                X_imp = self.extract_pval_features(P_imp_single, rna_seq_data, assay_indices=[assay_idx])
                if not X_imp.empty:
                    common_genes = X_imp.index.intersection(y_series.index)
                    if len(common_genes) > 10:
                        X_aligned = X_imp.loc[common_genes].values
                        y_aligned = y_series.loc[common_genes].values
                        
                        pipeline = Pipeline([
                            ('scale', StandardScaler()),
                            ('reg', LinearRegression())
                        ])
                        
                        try:
                            scores = cross_validate(pipeline, X_aligned, y_aligned,
                                                   cv=cv, scoring=scoring, n_jobs=n_jobs)
                            results_list.append({
                                'biosample': bios_name,
                                'assay_idx': assay_idx,
                                'assay_name': assay_name,
                                'category': 'imputed',
                                'pearson': np.mean(scores['test_pearson']),
                                'spearman': np.mean(scores['test_spearman']),
                                'r2': np.mean(scores['test_r2']),
                                'mse': -np.mean(scores['test_neg_mse'])
                            })
                        except Exception as e:
                            print(f"  Warning: Failed to evaluate imputed {assay_name}: {e}")
        
        return pd.DataFrame(results_list)
    
    def process_biosample(self, bios_name: str, n_jobs: int = -1, 
                         feature_selection: str = "pca", run_per_assay: bool = False,
                         run_multi_assay: bool = True) -> Dict[str, Any]:
        """
        Process a single biosample: run predictions and evaluate.
        
        Args:
            bios_name: Name of the biosample
            n_jobs: Number of parallel jobs for CV
            feature_selection: Feature selection method ('pca' or 'ftest')
            run_per_assay: Whether to run per-assay evaluation
            run_multi_assay: Whether to run multi-assay evaluation
            
        Returns:
            Dictionary with results for all settings
        """
        print(f"\n=== Processing {bios_name} ===")
        
        # Load RNA-seq data
        rna_seq_data = self.dataset.load_rna_seq_data(bios_name, self.gene_coords)
        if len(rna_seq_data) == 0:
            print(f"No RNA-seq data found for {bios_name}")
            return None

        # Build per-gene TPM table (one TPM per geneID), matching old_eval.quick_eval_rnaseq
        gene_info = (
            rna_seq_data[['geneID', 'TPM']]
            .drop_duplicates(subset='geneID')
            .set_index('geneID')
        )
        
        # Load genomic data
        locus = ["chr21", 0, self.dataset.chr_sizes["chr21"]]
        if self.predictor.DNA:
            X, Y, P, seq, mX, mY, avX, avY = self.predictor.load_data(
                bios_name, locus, dsf=1, fill_y_prompt_spec=None, fill_prompt_mode="median"
            )
        else:
            X, Y, P, mX, mY, avX, avY = self.predictor.load_data(
                bios_name, locus, dsf=1, fill_y_prompt_spec=None, fill_prompt_mode="median"
            )
            seq = None
        
        # Get available indices
        available_indices = torch.where(avX[0, :] == 1)[0].tolist()
        if 35 in available_indices:  # Remove control
            available_indices.remove(35)
        
        num_tracks = len(available_indices)
        print(f"Available tracks: {num_tracks}")
        
        # Flatten to 2D
        P_flat = P.view(-1, P.shape[-1]).numpy()
        signal_length = P_flat.shape[0]
        
        # Prepare results dictionary
        results = {
            'biosample': bios_name,
            'num_tracks': num_tracks,
            'settings': {}
        }
        
        # 1. Observed P-val (only available assays)
        print("Extracting features from observed P-val...")
        P_avail = P_flat[:, available_indices]
        X_obs_pval = self.extract_pval_features(P_avail, rna_seq_data, assay_indices=available_indices)
        
        # 2. Denoised P-val (upsampling for available assays only)
        print("Running denoised predictions...")
        if self.predictor.DNA:
            n_den, p_den, mu_den, var_den, peak_den = self.predictor.predict(
                X, mX, mY, avX, seq, imp_target=[]
            )
        else:
            n_den, p_den, mu_den, var_den, peak_den = self.predictor.predict(
                X, mX, mY, avX, None, imp_target=[]
            )
        
        mu_den_flat = mu_den.view(-1, mu_den.shape[-1]).numpy()
        # Extract only available assays
        mu_den_avail = mu_den_flat[:, available_indices]
        X_den_pval = self.extract_pval_features(mu_den_avail, rna_seq_data, assay_indices=available_indices)
        
        # 3. Imputed+Denoised P-val (upsampling for all 35 assays)
        print("Running imputed+denoised predictions...")
        # Use same upsampling predictions but for all assays
        all_assay_indices = list(range(self.n_assays))
        X_impden_pval = self.extract_pval_features(mu_den_flat, rna_seq_data, assay_indices=all_assay_indices)
        
        # 4. Latent Z
        print("Extracting latent Z...")
        if self.predictor.DNA:
            Z = self.predictor.get_latent_z(X, mX, mY, avX, seq)
        else:
            Z = self.predictor.get_latent_z(X, mX, mY, avX, None)
        Z_flat = Z.view(-1, Z.shape[-1]).numpy()
        X_latent = self.extract_latent_features(Z_flat, rna_seq_data, signal_length)
        
        # Prepare target using log1p(TPM), indexed by geneID (as in quick_eval_rnaseq)
        # y_series = np.arcsinh(gene_info['TPM'])
        y_series = np.log1p(gene_info['TPM'])
        
        # Run nested CV for each setting and model (if multi-assay evaluation is enabled)
        if run_multi_assay:
            settings = {
                'observed_pval': X_obs_pval,
                'denoised_pval': X_den_pval,
                'impden_pval': X_impden_pval,
                'latent_z': X_latent
            }
            # model_list = ["linear", "ridge", "lasso", "svr", "xgb", "rf"]
            model_list = [
                "ridge", 
                "lasso", 
                # "svr", 
                # "rf"
                ]
            
            for setting_name, X_features in settings.items():
                if X_features.empty or len(X_features) == 0:
                    print(f"Warning: No features for {setting_name}")
                    results['settings'][setting_name] = {
                        model: {'pearson': np.nan, 'spearman': np.nan, 'r2': np.nan, 'mse': np.nan}
                        for model in model_list
                    }
                    continue
                
                # Align features and targets by geneID, preserving consistent ordering
                common_genes = X_features.index.intersection(y_series.index)
                if len(common_genes) == 0:
                    print(f"Warning: No common genes for {setting_name}")
                    results['settings'][setting_name] = {
                        model: {'pearson': np.nan, 'spearman': np.nan, 'r2': np.nan, 'mse': np.nan}
                        for model in model_list
                    }
                    continue
                
                X_aligned = X_features.loc[common_genes].values
                y_aligned = y_series.loc[common_genes].values
                
                results['settings'][setting_name] = {}
                for model in model_list:
                    print(f"Running nested CV for {setting_name} with {model} ({len(common_genes)} genes)...")
                    metrics = self.run_nested_cv(X_aligned, y_aligned, n_jobs=n_jobs, 
                                                model=model, feature_selection=feature_selection)
                    results['settings'][setting_name][model] = metrics
                    print(
                        f"  [{model}] Pearson={metrics['pearson']:.3f}, "
                        f"Spearman={metrics['spearman']:.3f}, R2={metrics['r2']:.3f}, "
                        f"MSE={metrics['mse']:.3f}"
                    )
        
        # Per-assay evaluation (if requested)
        per_assay_results = None
        if run_per_assay:
            print("\n=== Running per-assay evaluation ===")
            per_assay_results = self.evaluate_per_assay(
                bios_name, P_avail, mu_den_avail, mu_den_flat,
                rna_seq_data, available_indices, n_jobs=n_jobs
            )
            print(f"Per-assay evaluation completed: {len(per_assay_results)} results")
        
        # Return both multi-assay and per-assay results
        results['per_assay'] = per_assay_results
        return results


def plot_parity(results_df: pd.DataFrame, metric: str, output_path: Path):
    """
    Parity plots comparing observed P-val baseline vs other settings.
    Multi-panel layout: one column per regression model (linear, ridge, rf).
    Matches aesthetics from reference plotting code.
    """
    if 'model' not in results_df.columns:
        results_df = results_df.copy()
        results_df['model'] = 'linear'
    
    models = sorted(results_df['model'].unique())
    
    # Style per setting: color and marker
    setting_styles = {
        'denoised_pval':  {'marker': 'o', 'color': 'blue'},
        'impden_pval':    {'marker': 'X', 'color': 'green'},
        'latent_z':       {'marker': '^', 'color': 'red'},
    }
    
    settings_to_plot = ['denoised_pval', 'impden_pval', 'latent_z']
    
    # Create figure with square subplots
    subplot_size = 4
    n_cols = len(models)
    fig_width = subplot_size * n_cols
    fig_height = subplot_size
    
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height), 
                            sharex=True, sharey=True, squeeze=False)
    axes = axes[0]  # Flatten for easier indexing
    
    fig.suptitle(f"Parity Plots for {metric.capitalize()} Correlation", fontsize=16)
    fig.supxlabel("Observed Correlation", fontsize=14)
    fig.supylabel("Predicted Correlation", fontsize=14)
    
    # Global baseline range for consistent axes
    global_min, global_max = None, None
    
    for ax, model_name in zip(axes, models):
        df_m = results_df[results_df['model'] == model_name]
        baseline = df_m[df_m['setting'] == 'observed_pval'].set_index('biosample')[metric]
        if baseline.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            continue
        
        # Track global min/max
        if global_min is None:
            global_min = baseline.min()
            global_max = baseline.max()
        else:
            global_min = min(global_min, baseline.min())
            global_max = max(global_max, baseline.max())
        
        for setting in settings_to_plot:
            style = setting_styles.get(setting, {'marker': 'o', 'color': 'grey'})
            setting_data = df_m[df_m['setting'] == setting].set_index('biosample')[metric]
            common_bios = baseline.index.intersection(setting_data.index)
            if len(common_bios) == 0:
                continue
            
            x_vals = baseline.loc[common_bios]
            y_vals = setting_data.loc[common_bios]
            
            # Format label: convert denoised_pval -> Denoised Pval
            label = setting.replace('_', ' ').replace('pval', 'P-val').title()
            
            ax.scatter(
                x_vals, y_vals,
                label=label,
                marker=style['marker'], 
                color=style['color'], 
                alpha=0.7, 
                s=80,
                edgecolor='none'
            )
        
        ax.set_title(model_name.capitalize(), fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    if global_min is not None:
        pad = 0.05 * (global_max - global_min)
        lo, hi = global_min - pad, global_max + pad
        for ax in axes:
            ax.axline((0, 0), slope=1, color='grey', linestyle='--', label='y=x', alpha=0.7)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
    
    # Create and place the shared legend
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
              ncol=len(settings_to_plot)+1, fontsize=12)
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.92])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_vs_tracks(results_df: pd.DataFrame, metric: str, output_path: Path):
    """
    Plot performance vs number of tracks with regression lines and confidence intervals.
    Multi-panel layout: one column per regression model.
    Uses seaborn regplot for fitted lines with CI, matching reference aesthetics.
    
    Args:
        results_df: DataFrame with results
        metric: Metric name
        output_path: Path to save plot
    """
    if 'model' not in results_df.columns:
        results_df = results_df.copy()
        results_df['model'] = 'linear'
    
    models = sorted(results_df['model'].unique())
    
    # Style per setting: color, marker, and linestyle
    setting_styles = {
        'observed_pval':  {'marker': 'o', 'color': 'black', 'linestyle': ':'},
        'denoised_pval':  {'marker': 's', 'color': 'blue', 'linestyle': '--'},
        'impden_pval':    {'marker': 'X', 'color': 'green', 'linestyle': '-.'},
        'latent_z':       {'marker': '^', 'color': 'red', 'linestyle': '-'},
    }
    
    settings_to_plot = ['observed_pval', 'denoised_pval', 'impden_pval', 'latent_z']
    
    # Create figure
    n_cols = len(models)
    subplot_size = 5
    fig_width = subplot_size * n_cols
    fig_height = subplot_size
    
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height), 
                            sharex=True, sharey=True, squeeze=False)
    axes = axes[0]  # Flatten for easier indexing
    
    fig.suptitle(f"Performance vs. Number of Tracks ({metric.upper()})", fontsize=16)
    fig.supxlabel("Number of Tracks", fontsize=14)
    fig.supylabel(f"Performance ({metric.upper()})", fontsize=14)
    
    for ax, model_name in zip(axes, models):
        df_m = results_df[results_df['model'] == model_name]
        
        has_data = False
        for setting in settings_to_plot:
            style = setting_styles.get(setting, {'marker': 'o', 'color': 'grey', 'linestyle': '-'})
            setting_data = df_m[df_m['setting'] == setting].copy()
            
            # Need at least 2 unique num_tracks values to fit a line
            if not setting_data.empty and len(setting_data['num_tracks'].unique()) > 1:
                has_data = True
                
                # Format label: convert observed_pval -> Observed P-val
                label = setting.replace('_', ' ').replace('pval', 'P-val').title()
                
                # Use seaborn's regplot for robust linear fit with 95% CI
                sns.regplot(
                    x='num_tracks', y=metric, data=setting_data, ax=ax,
                    scatter=False,  # We'll add scatter separately for control
                    ci=95,  # 95% confidence interval
                    color=style['color'],
                    label=label
                )
                
                # Optional: add scatter points (commented out to match reference style)
                # ax.scatter(setting_data['num_tracks'], setting_data[metric],
                #           marker=style['marker'], color=style['color'], alpha=0.6, s=30)
        
        if not has_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title(model_name.capitalize(), fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Create and place the shared legend
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
              ncol=len(settings_to_plot), fontsize=12)
    
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_biosample_barplots(results_df: pd.DataFrame, metric: str, output_path: Path):
    """
    Create multi-panel bar plots showing per-assay performance for each biosample.
    One column layout with one row per biosample.
    
    Args:
        results_df: DataFrame with per-assay results
        metric: Metric name (pearson, spearman, r2, mse)
        output_path: Path to save plot
    """
    biosamples = sorted(results_df['biosample'].unique())
    n_biosamples = len(biosamples)
    
    # Single column layout - one row per biosample
    n_rows = n_biosamples
    n_cols = 1
    
    # Get all unique assays and create name mapping
    all_assays = sorted(results_df['assay_idx'].unique())
    n_assays = len(all_assays)
    
    # Create assay name mapping from the data
    assay_names = {}
    for assay_idx in all_assays:
        assay_name = results_df[results_df['assay_idx'] == assay_idx]['assay_name'].iloc[0]
        assay_names[assay_idx] = assay_name
    
    # Create figure with larger panels
    fig_width = 20  # Wide enough for all assay names
    fig_height = 4 * n_rows  # 4 inches per biosample
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), 
                            squeeze=False)
    axes = axes.flatten()
    
    # Color scheme: royalblue (observed), seagreen (denoised), salmon (imputed)
    category_colors = {
        'observed': 'royalblue',
        'denoised': 'limegreen',
        'imputed': 'salmon'
    }
    
    # Bar width and positions
    bar_width = 0.25
    
    for idx, biosample in enumerate(biosamples):
        ax = axes[idx]
        df_bios = results_df[results_df['biosample'] == biosample]
        
        # Prepare data for this biosample
        for assay_idx in all_assays:
            df_assay = df_bios[df_bios['assay_idx'] == assay_idx]
            
            # Position for this assay
            x_pos = assay_idx
            
            # Plot bars for each category
            for i, category in enumerate(['observed', 'denoised', 'imputed']):
                df_cat = df_assay[df_assay['category'] == category]
                if len(df_cat) > 0:
                    value = df_cat[metric].values[0]
                    offset = (i - 1) * bar_width
                    ax.bar(x_pos + offset, value, bar_width, 
                          color=category_colors[category], alpha=0.8,
                          label=category.capitalize() if assay_idx == all_assays[0] else '')
        
        # Add vertical dotted separator lines between assays
        for assay_idx in all_assays[:-1]:
            ax.axvline(assay_idx + 0.5, linestyle=':', color='grey', alpha=0.5, linewidth=1.5)
        
        # Formatting
        ax.set_title(biosample, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_xlim(-0.5, n_assays - 0.5)
        ax.set_xticks(all_assays)
        ax.set_xticklabels([assay_names[i] for i in all_assays], 
                          rotation=90, ha='center', fontsize=8)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Legend only for first subplot
        if idx == 0:
            ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
        
        # Only show x-axis label on bottom subplot
        if idx == n_biosamples - 1:
            ax.set_xlabel('Assay', fontsize=11)
    
    plt.suptitle(f'Per-Assay {metric.upper()} by Biosample', fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-biosample bar plot: {output_path}")


def plot_per_assay_boxplots(results_df: pd.DataFrame, metric: str, output_path: Path):
    """
    Create box plots aggregating per-assay performance across all biosamples.
    
    Args:
        results_df: DataFrame with per-assay results
        metric: Metric name (pearson, spearman, r2, mse)
        output_path: Path to save plot
    """
    # Get all unique assays
    all_assays = sorted(results_df['assay_idx'].unique())
    n_assays = len(all_assays)
    
    # Create assay name mapping from the data
    assay_names = {}
    for assay_idx in all_assays:
        assay_name = results_df[results_df['assay_idx'] == assay_idx]['assay_name'].iloc[0]
        assay_names[assay_idx] = assay_name
    
    # Color scheme: royalblue (observed), seagreen (denoised), salmon (imputed)
    category_colors = {
        'observed': 'royalblue',
        'denoised': 'seagreen',
        'imputed': 'salmon'
    }
    
    # Create figure - make it wide enough for assay names
    fig, ax = plt.subplots(1, 1, figsize=(max(20, n_assays * 0.6), 7))
    
    # Box width and positions
    box_width = 0.25
    positions_offset = {'observed': -box_width, 'denoised': 0, 'imputed': box_width}
    
    # Prepare data for boxplots
    all_box_data = []
    all_positions = []
    all_colors = []
    
    for assay_idx in all_assays:
        df_assay = results_df[results_df['assay_idx'] == assay_idx]
        
        for category in ['observed', 'denoised', 'imputed']:
            df_cat = df_assay[df_assay['category'] == category]
            if len(df_cat) > 0:
                values = df_cat[metric].values
                all_box_data.append(values)
                all_positions.append(assay_idx + positions_offset[category])
                all_colors.append(category_colors[category])
    
    # Create boxplots
    if len(all_box_data) > 0:
        bp = ax.boxplot(all_box_data, positions=all_positions, widths=box_width * 0.8,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(linewidth=1.2),
                       whiskerprops=dict(linewidth=1.2),
                       capprops=dict(linewidth=1.2),
                       medianprops=dict(linewidth=2, color='darkred'))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    # Add vertical dotted separator lines between assays
    for assay_idx in all_assays[:-1]:
        ax.axvline(assay_idx + 0.5, linestyle=':', color='grey', alpha=0.5, linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Assay', fontsize=13)
    ax.set_ylabel(metric.upper(), fontsize=13)
    ax.set_title(f'Per-Assay {metric.upper()} Across All Biosamples', fontsize=15, fontweight='bold')
    ax.set_xlim(-0.5, n_assays - 0.5)
    ax.set_xticks(all_assays)
    ax.set_xticklabels([assay_names[i] for i in all_assays], 
                       rotation=90, ha='center', fontsize=9)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='royalblue', alpha=0.7, label='Observed'),
        Patch(facecolor='seagreen', alpha=0.7, label='Denoised'),
        Patch(facecolor='salmon', alpha=0.7, label='Imputed')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-assay box plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='RNA-seq evaluation script')
    parser.add_argument('--model-dir', type=str, required=True, help='Path to model directory')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: model_dir/viz)')
    parser.add_argument('--resolution', type=int, default=25, help='Genomic resolution in bp')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs for CV')
    parser.add_argument('--biosample', type=str, default=None, help='Specific biosample (default: all with RNA-seq)')
    parser.add_argument('--feature-selection', type=str, default='pca', 
                       choices=['pca', 'ftest'],
                       help='Feature selection method: pca (PCA) or ftest (F-test SelectKBest)')
    parser.add_argument('--per-assay-only', action='store_true',
                       help='Run only per-assay analysis (skip multi-assay evaluation)')
    parser.add_argument('--run-per-assay', action='store_true',
                       help='Run per-assay analysis in addition to multi-assay evaluation')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(args.model_dir) / "viz"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = RNASeqEvaluator(args.model_dir, args.data_path, resolution=args.resolution)
    
    # Get biosamples with RNA-seq
    if args.biosample:
        biosamples = [args.biosample]
    else:
        biosamples = [bios for bios in evaluator.dataset.navigation.keys() 
                     if evaluator.dataset.has_rnaseq(bios)]
    
    print(f"Processing {len(biosamples)} biosamples with RNA-seq data")
    
    # Determine which evaluations to run
    run_multi_assay = not args.per_assay_only
    run_per_assay = args.per_assay_only or args.run_per_assay
    
    # Process each biosample
    all_results = []
    all_per_assay_results = []
    
    for bios in biosamples:
        try:
            result = evaluator.process_biosample(
                bios, n_jobs=args.n_jobs, 
                feature_selection=args.feature_selection,
                run_per_assay=run_per_assay,
                run_multi_assay=run_multi_assay
            )
            if result is not None:
                # Collect multi-assay results
                if run_multi_assay and 'settings' in result:
                    for setting, model_dict in result['settings'].items():
                        for model_name, metrics in model_dict.items():
                            all_results.append({
                                'biosample': result['biosample'],
                                'num_tracks': result['num_tracks'],
                                'setting': setting,
                                'model': model_name,
                                'pearson': metrics['pearson'],
                                'spearman': metrics['spearman'],
                                'r2': metrics['r2'],
                                'mse': metrics['mse']
                            })
                
                # Collect per-assay results
                if run_per_assay and result.get('per_assay') is not None:
                    all_per_assay_results.append(result['per_assay'])
        except Exception as e:
            print(f"Error processing {bios}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Add feature selection method to filename suffix
    fs_suffix = f"_{args.feature_selection}" if args.feature_selection != "pca" else ""
    
    # Multi-assay results and plots
    if run_multi_assay and len(all_results) > 0:
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        results_csv = output_dir / f"rnaseq_results{fs_suffix}.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"\nMulti-assay results saved to {results_csv}")
        
        # Generate plots
        metrics = ['pearson', 'spearman', 'r2', 'mse']
        
        for metric in metrics:
            # Parity plot
            parity_path = output_dir / f"rnaseq_parity_{metric}{fs_suffix}.svg"
            plot_parity(results_df, metric, parity_path)
            print(f"Saved parity plot: {parity_path}")
            
            # Performance vs tracks
            perf_path = output_dir / f"rnaseq_perf_vs_tracks_{metric}{fs_suffix}.svg"
            plot_performance_vs_tracks(results_df, metric, perf_path)
            print(f"Saved performance plot: {perf_path}")
    
    # Per-assay results and plots
    if run_per_assay and len(all_per_assay_results) > 0:
        # Combine all per-assay results
        per_assay_df = pd.concat(all_per_assay_results, ignore_index=True)
        
        # Save per-assay results
        per_assay_csv = output_dir / "rnaseq_per_assay_results.csv"
        per_assay_df.to_csv(per_assay_csv, index=False)
        print(f"\nPer-assay results saved to {per_assay_csv}")
        
        # Generate per-assay plots
        metrics = ['pearson', 'spearman', 'r2', 'mse']
        
        for metric in metrics:
            # Bar plots per biosample
            barplot_path = output_dir / f"rnaseq_per_assay_barplot_{metric}.svg"
            plot_per_biosample_barplots(per_assay_df, metric, barplot_path)
            
            # Box plots aggregated across biosamples
            boxplot_path = output_dir / f"rnaseq_per_assay_boxplot_{metric}.svg"
            plot_per_assay_boxplots(per_assay_df, metric, boxplot_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

