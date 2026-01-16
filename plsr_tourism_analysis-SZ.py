"""
PLSR Analysis of International Tourist Arrivals Determinants
Author: Simon Zhang
Date: 01/2026
Description: Complete PLSR analysis package with permutation testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import scipy.stats as stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TourismPLSR:   
    def __init__(self, data_path=None, X=None, y=None, random_state=42):
        """
        Initialize the PLSR analysis
        Parameters:
        -----------
        data_path : str or Path, optional
            Path to Excel file with data
        X : pandas DataFrame, optional
            Predictor variables
        y : pandas Series or DataFrame, optional
            Response variable
        random_state : int, optional
            Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        if data_path:
            self.load_data(data_path)
        elif X is not None and y is not None:
            self.X = X.copy()
            self.y = y.copy()
            self.preprocess_data()
        else:
            raise ValueError("Either data_path or (X, y) must be provided")
        
        # Results storage
        self.results = {}
        self.permutation_results = {}
        self.bootstrap_results = {}
        
    def load_data(self, data_path, sheet_name='Indexing'):
        """
        Load data from Excel file
        Parameters:
        -----------
        data_path : str or Path
            Path to Excel file
        sheet_name : str, optional
            Sheet name to read
        """
        print(f"Loading data from {data_path}...")
        data = pd.read_excel(data_path, sheet_name=sheet_name)
        
        # Extract variables based on your Table S1 structure
        self.X = data[[
            'Safety/Security Index (1-7)*',
            'Health/Hygiene Index (1-7)*', 
            'SST Readiness (1-7)*',
            'Price Competitiveness (1-7)*',
            'Air Transport Infra (1-7)*',
            'Environmental Sustainability (1-7)*'
        ]].copy()
        
        self.y = data['Ln(Arrivals)'].copy()
        
        # Store country names for reference
        self.countries = data['Country'].copy()
        
        print(f"Data loaded: {len(self.X)} observations, {self.X.shape[1]} predictors")
        self.preprocess_data()
    
    def preprocess_data(self):
        """Standardize the data"""
        self.X_raw = self.X.copy()
        self.y_raw = self.y.copy()
        
        # Standardize X
        self.scaler_X = StandardScaler()
        self.X_std = pd.DataFrame(
            self.scaler_X.fit_transform(self.X),
            columns=self.X.columns,
            index=self.X.index
        )
        
        # Standardize y
        self.scaler_y = StandardScaler()
        self.y_std = pd.Series(
            self.scaler_y.fit_transform(self.y.values.reshape(-1, 1)).flatten(),
            index=self.y.index
        )
        
        print("Data preprocessing completed.")
    
    def determine_optimal_components(self, max_components=6, cv_folds=10):
        """
        Determine optimal number of PLS components using cross-validation
        Parameters:
        -----------
        max_components : int, optional
            Maximum number of components to test
        cv_folds : int, optional
            Number of cross-validation folds
        Returns:
        --------
        optimal_n_comp : int
            Optimal number of components
        """
        print("\n" + "="*60)
        print("Determining optimal number of PLS components...")
        print("="*60)
        
        n_components_range = range(1, min(max_components, self.X.shape[1]) + 1)
        mse_scores = []
        q2_scores = []
        
        for n_comp in n_components_range:
            pls = PLSRegression(n_components=n_comp)
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Cross-validated predictions
            y_pred = cross_val_predict(pls, self.X_std, self.y_std, cv=kfold)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_std, y_pred)
            q2 = 1 - (np.sum((self.y_std - y_pred)**2) / np.sum((self.y_std - np.mean(self.y_std))**2))
            
            mse_scores.append(mse)
            q2_scores.append(q2)
            
            print(f"Components: {n_comp}, MSE: {mse:.4f}, Q²: {q2:.4f}")
        
        # Find optimal based on Q² (higher is better)
        q2_scores = np.array(q2_scores)
        optimal_idx = np.argmax(q2_scores)
        optimal_n_comp = n_components_range[optimal_idx]
        
        # Store results
        self.results['component_selection'] = {
            'n_components_range': list(n_components_range),
            'mse_scores': mse_scores,
            'q2_scores': q2_scores,
            'optimal_n_comp': optimal_n_comp
        }
        
        print(f"\nOptimal number of components: {optimal_n_comp}")
        print(f"Maximum Q²: {q2_scores[optimal_idx]:.4f}")
        
        return optimal_n_comp
    
    def calculate_vip_scores(self):
        """
        Calculate Variable Importance in Projection (VIP) scores
        Returns:
        --------
        vip_scores : pandas Series
            VIP scores for each predictor
        """
        print("\n" + "="*60)
        print("Calculating VIP scores...")
        print("="*60)
        
        model = self.results['final_model']['model']
        
        # Extract model parameters
        t = model.x_scores_           # X scores
        w = model.x_weights_          # X weights
        q = model.y_loadings_         # Y loadings
        
        p = w.shape[0]  # Number of predictors
        vip = np.zeros((p,))
        
        # Calculate sum of squares
        s = np.diag(t.T @ t @ q.T @ q).reshape(-1, 1)
        total_s = np.sum(s)
        
        # Calculate VIP for each predictor
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 
                             for j in range(model.n_components)])
            vip[i] = np.sqrt(p * (weight.T @ s) / total_s)
        
        # Create Series with variable names
        vip_scores = pd.Series(vip, index=self.X.columns, name='VIP_Score')
        
        # Sort by VIP
        vip_scores = vip_scores.sort_values(ascending=False)
        
        # Store results
        self.results['vip_scores'] = vip_scores
        
        print("\nVIP Scores (sorted):")
        for var, score in vip_scores.items():
            significance = "***" if score > 1.0 else "*" if score > 0.8 else ""
            print(f"  {var:40} VIP: {score:.3f} {significance}")
        
        return vip_scores
    
    def permutation_test(self, n_permutations=1000, cv_folds=10):
        """
        Perform permutation test for model significance
        Parameters:
        -----------
        n_permutations : int, optional
            Number of permutations
        cv_folds : int, optional
            Number of cross-validation folds
            
        Returns:
        --------
        p_value : float
            Permutation p-value
        """
        print("\n" + "="*60)
        print(f"Performing permutation test (N={n_permutations})...")
        print("="*60)
        
        n_comp = self.results['final_model']['n_components']
        
        # Calculate original score
        pls = PLSRegression(n_components=n_comp)
        original_score = cross_val_score(
            pls, self.X_std, self.y_std, 
            cv=cv_folds, 
            scoring='r2'
        ).mean()
        
        # Permutation loop
        perm_scores = np.zeros(n_permutations)
        
        for i in range(n_permutations):
            # Permute y values
            y_perm = np.random.permutation(self.y_std)
            
            # Calculate permuted score
            perm_score = cross_val_score(
                pls, self.X_std, y_perm, 
                cv=cv_folds, 
                scoring='r2'
            ).mean()
            
            perm_scores[i] = perm_score
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_permutations} permutations...")
        
        # Calculate p-value (one-tailed)
        p_value = (np.sum(perm_scores >= original_score) + 1) / (n_permutations + 1)
        
        # Calculate z-score
        z_score = (original_score - np.mean(perm_scores)) / np.std(perm_scores)
        
        # Store results
        self.permutation_results = {
            'original_score': original_score,
            'perm_scores': perm_scores,
            'p_value': p_value,
            'z_score': z_score,
            'perm_mean': np.mean(perm_scores),
            'perm_std': np.std(perm_scores),
            'n_permutations': n_permutations
        }
        
        print(f"\nPermutation test results:")
        print(f"  Original R²: {original_score:.4f}")
        print(f"  Permuted mean R²: {np.mean(perm_scores):.4f}")
        print(f"  Permuted std R²: {np.std(perm_scores):.4f}")
        print(f"  Z-score: {z_score:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        
        return p_value
    
    def bootstrap_analysis(self, n_bootstraps=1000, n_components=None):
        """
        Perform bootstrap analysis for coefficient stability
        
        Parameters:
        -----------
        n_bootstraps : int, optional
            Number of bootstrap samples
        n_components : int, optional
            Number of PLS components (uses optimal if None)
            
        Returns:
        --------
        bootstrap_coeffs : pandas DataFrame
            Bootstrap coefficient distributions
        """
        print("\n" + "="*60)
        print(f"Performing bootstrap analysis (N={n_bootstraps})...")
        print("="*60)
        
        if n_components is None:
            n_components = self.results['final_model']['n_components']
        
        bootstrap_coeffs = []
        bootstrap_intercepts = []
        
        for i in range(n_bootstraps):
            # Bootstrap sample
            X_resample, y_resample = resample(
                self.X_std, self.y_std,
                random_state=self.random_state + i
            )
            
            # Fit PLS model
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_resample, y_resample)
            
            # Store coefficients (scaled back to original)
            coeffs_scaled = pls.coef_.flatten() * (self.scaler_y.scale_ / self.scaler_X.scale_)
            bootstrap_coeffs.append(coeffs_scaled)
            
            # Store intercept (scaled back)
            intercept_scaled = (pls.y_mean_ - np.dot(pls.x_mean_, coeffs_scaled))[0]
            bootstrap_intercepts.append(intercept_scaled)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_bootstraps} bootstrap samples...")
        
        # Convert to arrays
        bootstrap_coeffs = np.array(bootstrap_coeffs)
        bootstrap_intercepts = np.array(bootstrap_intercepts)
        
        # Calculate statistics
        coeff_mean = np.mean(bootstrap_coeffs, axis=0)
        coeff_std = np.std(bootstrap_coeffs, axis=0)
        coeff_ci_lower = np.percentile(bootstrap_coeffs, 2.5, axis=0)
        coeff_ci_upper = np.percentile(bootstrap_coeffs, 97.5, axis=0)
        
        # Create DataFrame
        coeff_stats = pd.DataFrame({
            'Variable': self.X.columns,
            'Coefficient_Mean': coeff_mean,
            'Coefficient_Std': coeff_std,
            'CI_2.5%': coeff_ci_lower,
            'CI_97.5%': coeff_ci_upper,
            'Stability_Ratio': np.abs(coeff_mean) / coeff_std
        })
        
        # Store results
        self.bootstrap_results = {
            'coefficients': bootstrap_coeffs,
            'intercepts': bootstrap_intercepts,
            'coefficient_stats': coeff_stats,
            'n_bootstraps': n_bootstraps
        }
        
        print("\nBootstrap coefficient statistics:")
        print(coeff_stats.round(4))
        
        return coeff_stats
    
    def calculate_model_metrics(self, cv_folds=10):
        """
        Calculate comprehensive model metrics
        
        Parameters:
        -----------
        cv_folds : int, optional
            Number of cross-validation folds
            
        Returns:
        --------
        metrics : dict
            Dictionary of model metrics
        """
        print("\n" + "="*60)
        print("Calculating model metrics...")
        print("="*60)
        
        n_comp = self.results['final_model']['n_components']
        model = self.results['final_model']['model']
        
        # Cross-validated predictions
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        y_pred_cv = cross_val_predict(model, self.X_std, self.y_std, cv=kfold)
        
        # Full model predictions
        y_pred_full = model.predict(self.X_std)
        
        # Calculate metrics
        metrics = {
            # R² metrics
            'R2Y_train': r2_score(self.y_std, y_pred_full),
            'R2Y_cv': r2_score(self.y_std, y_pred_cv),
            
            # Q² (predictive ability)
            'Q2': 1 - (np.sum((self.y_std - y_pred_cv)**2) / 
                      np.sum((self.y_std - np.mean(self.y_std))**2)),
            
            # Error metrics
            'RMSEP': np.sqrt(mean_squared_error(self.y_std, y_pred_cv)),
            'MAEP': mean_absolute_error(self.y_std, y_pred_cv),
            
            # Component statistics
            'n_components': n_comp,
            'R2X_cumulative': self._calculate_r2x(model),
            'R2Y_cumulative': self._calculate_r2y(model)
        }
        
        # Component-wise metrics
        r2y_components, r2x_components = self._component_wise_statistics(n_comp)
        metrics['R2Y_by_component'] = r2y_components
        metrics['R2X_by_component'] = r2x_components
        
        # Store results
        self.results['model_metrics'] = metrics
        
        print("\nModel Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key:25}: {value:.4f}")
        
        return metrics
    
    def _calculate_r2x(self, model):
        """Calculate cumulative R²X"""
        X_reconstructed = model.x_scores_ @ model.x_loadings_.T
        ss_res = np.sum((self.X_std - X_reconstructed)**2)
        ss_tot = np.sum((self.X_std - np.mean(self.X_std, axis=0))**2)
        return 1 - (ss_res / ss_tot)
    
    def _calculate_r2y(self, model):
        """Calculate cumulative R²Y"""
        y_pred = model.predict(self.X_std)
        return r2_score(self.y_std, y_pred)
    
    def _component_wise_statistics(self, n_components):
        """Calculate component-wise statistics"""
        r2y_components = []
        r2x_components = []
        
        for i in range(1, n_components + 1):
            pls_temp = PLSRegression(n_components=i)
            pls_temp.fit(self.X_std, self.y_std)
            
            # R²Y for this component
            y_pred = pls_temp.predict(self.X_std)
            r2y = r2_score(self.y_std, y_pred)
            r2y_components.append(r2y)
            
            # R²X for this component
            X_reconstructed = pls_temp.x_scores_ @ pls_temp.x_loadings_.T
            ss_res = np.sum((self.X_std - X_reconstructed)**2)
            ss_tot = np.sum((self.X_std - np.mean(self.X_std, axis=0))**2)
            r2x = 1 - (ss_res / ss_tot)
            r2x_components.append(r2x)
        
        return r2y_components, r2x_components
    
    def run_full_analysis(self, max_components=6, n_permutations=1000, 
                         n_bootstraps=1000, cv_folds=10):
        """
        Run complete PLSR analysis pipeline
        
        Parameters:
        -----------
        max_components : int, optional
            Maximum components to test
        n_permutations : int, optional
            Number of permutations
        n_bootstraps : int, optional
            Number of bootstrap samples
        cv_folds : int, optional
            Number of CV folds
        """
        print("\n" + "="*60)
        print("STARTING COMPLETE PLSR ANALYSIS")
        print("="*60)
        
        # 1. Determine optimal components
        optimal_n_comp = self.determine_optimal_components(max_components, cv_folds)
        
        # 2. Fit final model
        final_model = PLSRegression(n_components=optimal_n_comp)
        final_model.fit(self.X_std, self.y_std)
        
        self.results['final_model'] = {
            'model': final_model,
            'n_components': optimal_n_comp,
            'x_loadings': final_model.x_loadings_,
            'y_loadings': final_model.y_loadings_,
            'x_scores': final_model.x_scores_,
            'y_scores': final_model.y_scores_,
            'coefficients': final_model.coef_.flatten()
        }
        
        # 3. Calculate VIP scores
        vip_scores = self.calculate_vip_scores()
        
        # 4. Calculate model metrics
        metrics = self.calculate_model_metrics(cv_folds)
        
        # 5. Permutation test
        p_value = self.permutation_test(n_permutations, cv_folds)
        
        # 6. Bootstrap analysis
        bootstrap_stats = self.bootstrap_analysis(n_bootstraps, optimal_n_comp)
        
        # 7. Calculate redundancy indices
        redundancy = self.calculate_redundancy()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return self.results
    
    def calculate_redundancy(self):
        """
        Calculate redundancy indices for PLS components
        
        Returns:
        --------
        redundancy_df : pandas DataFrame
            Redundancy indices for each component
        """
        model = self.results['final_model']['model']
        n_comp = model.n_components
        
        redundancy = []
        for h in range(n_comp):
            # Extract component scores
            t_h = model.x_scores_[:, h]
            
            # R²(Y, t_h): average squared correlation between Y and component h
            r2_y = np.corrcoef(self.y_std, t_h)[0, 1]**2
            
            # R²(X, t_h): proportion of X-variance explained by component h
            # This is the eigenvalue proportion
            x_var_total = np.var(self.X_std, axis=0).sum()
            x_var_explained = np.var(t_h) * model.x_loadings_[:, h].T @ model.x_loadings_[:, h]
            r2_x = x_var_explained / x_var_total
            
            # Redundancy index
            redundancy_h = r2_y * r2_x
            
            redundancy.append({
                'Component': h + 1,
                'R2_Y': r2_y,
                'R2_X': r2_x,
                'Redundancy': redundancy_h
            })
        
        redundancy_df = pd.DataFrame(redundancy)
        self.results['redundancy'] = redundancy_df
        
        return redundancy_df
    
    def create_summary_report(self, output_dir='results'):
        """
        Create comprehensive summary report
        
        Parameters:
        -----------
        output_dir : str, optional
            Output directory for results
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create summary DataFrame
        summary_data = []
        
        # Model metrics
        metrics = self.results['model_metrics']
        summary_data.append({'Metric': 'Optimal Components', 'Value': metrics['n_components']})
        summary_data.append({'Metric': 'R²Y (Training)', 'Value': metrics['R2Y_train']})
        summary_data.append({'Metric': 'R²Y (CV)', 'Value': metrics['R2Y_cv']})
        summary_data.append({'Metric': 'Q² (Predictive)', 'Value': metrics['Q2']})
        summary_data.append({'Metric': 'RMSEP', 'Value': metrics['RMSEP']})
        summary_data.append({'Metric': 'R²X (Cumulative)', 'Value': metrics['R2X_cumulative']})
        
        # Permutation results
        perm = self.permutation_results
        summary_data.append({'Metric': 'Permutation p-value', 'Value': perm['p_value']})
        summary_data.append({'Metric': 'Permutation Z-score', 'Value': perm['z_score']})
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save results to files
        summary_df.to_excel(f'{output_dir}/summary_metrics.xlsx', index=False)
        
        # VIP scores
        self.results['vip_scores'].to_excel(f'{output_dir}/vip_scores.xlsx')
        
        # Bootstrap coefficients
        self.bootstrap_results['coefficient_stats'].to_excel(
            f'{output_dir}/bootstrap_coefficients.xlsx', index=False
        )
        
        # Redundancy indices
        if 'redundancy' in self.results:
            self.results['redundancy'].to_excel(
                f'{output_dir}/redundancy_indices.xlsx', index=False
            )
        
        print(f"\nSummary report saved to {output_dir}/")
        
        return summary_df

# Main execution function
def main():
    """Main function to run the complete analysis"""
    
    # Initialize analysis
    print("Tourism PLSR Analysis Package")
    print("="*60)
    
    analyzer = TourismPLSR(data_path='Table S1-test.xlsx')
    
    # Run complete analysis
    results = analyzer.run_full_analysis(
        max_components=6,
        n_permutations=1000,
        n_bootstraps=1000,
        cv_folds=10
    )
    
    # Create summary report
    summary = analyzer.create_summary_report('results')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nResults have been saved to the 'results' directory.")
    print("\nKey findings:")
    print(f"- Optimal components: {analyzer.results['final_model']['n_components']}")
    print(f"- R²Y: {analyzer.results['model_metrics']['R2Y_cv']:.3f}")
    print(f"- Q²: {analyzer.results['model_metrics']['Q2']:.3f}")
    print(f"- Permutation p-value: {analyzer.permutation_results['p_value']:.4f}")
    
    # Print top predictors
    top_predictors = analyzer.results['vip_scores'].head(3)
    print("\nTop predictors (VIP > 1.0):")
    for var, score in top_predictors.items():
        print(f"  {var}: VIP = {score:.3f}")
    
    return analyzer


if __name__ == "__main__":
    # Run the analysis
    analyzer = main()
