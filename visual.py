import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from math import pi
import warnings

warnings.filterwarnings('ignore')

class AdvancedSegmentationVisualizer:
    def __init__(self, data_path_or_df, cluster_col='Cluster'):
        # Load data if path is provided, otherwise use DataFrame
        if isinstance(data_path_or_df, str):
            self.data = pd.read_parquet(data_path_or_df)
        else:
            self.data = data_path_or_df
            
        self.cluster_col = cluster_col
        self.n_clusters = self.data[self.cluster_col].nunique()
        self.cluster_labels = sorted(self.data[self.cluster_col].unique())
        
        # Design setup
        sns.set_theme(style="whitegrid")
        self.palette = sns.color_palette("Set2", self.n_clusters)

    def plot_cluster_sizes(self):
        """Displays the population count per cluster."""
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=self.data, x=self.cluster_col, palette=self.palette)
        
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height()):,}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 10), 
                        textcoords='offset points', fontweight='bold')
            
        plt.title('Customer Count per Segment', fontsize=16, pad=15)
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Total Customers', fontsize=12)
        plt.ylim(0, self.data[self.cluster_col].value_counts().max() * 1.15)
        plt.tight_layout()
        plt.show()

    def plot_behavioral_radar(self):
        """Plots a Radar chart for diverse structural metrics."""
        # Key psychological metrics to profile
        metrics = [
            'Average_Order_Value_AOV', 
            'Average_Units_Per_Transaction_UPT', 
            'Category_Width', 
            'Inferred_Discount_Affinity', 
            'Single_Item_Basket_Propensity'
        ]
        
        # Verify columns exist (fallback if data is missing some)
        valid_metrics = [m for m in metrics if m in self.data.columns]
        
        cluster_means = self.data.groupby(self.cluster_col)[valid_metrics].mean()
        
        # Normalize to 0-1 scale for comparative plotting
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(
            scaler.fit_transform(cluster_means),
            columns=valid_metrics,
            index=cluster_means.index
        )
        
        N = len(valid_metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        # Format labels
        formatted_labels = [m.replace('_', '\n') for m in valid_metrics]
        plt.xticks(angles[:-1], formatted_labels, size=11)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=9)
        plt.ylim(0, 1)
        
        for i, cluster in enumerate(self.cluster_labels):
            values = normalized_data.loc[cluster].values.flatten().tolist()
            values += values[:1]
            color = self.palette[i]
            
            ax.plot(angles, values, linewidth=2.5, linestyle='solid', 
                    label=f'Cluster {cluster}', color=color)
            ax.fill(angles, values, color=color, alpha=0.1)
            
        plt.title('Shopper DNA: Behavioral Profile (Normalized)', size=16, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()
        plt.show()

    def plot_risk_vs_value_matrix(self):
        """Plots Churn Risk vs. Average Order Value."""
        required_cols = ['Churn_Risk_Index_CRI', 'Average_Order_Value_AOV', 'Total_Lifetime_Spend']
        if not all(c in self.data.columns for c in required_cols):
            print("Missing columns for Risk vs Value Matrix. Skipping.")
            return

        plt.figure(figsize=(12, 8))
        
        # Remove extreme outliers for cleaner visualization (e.g., top 1%)
        p99_aov = self.data['Average_Order_Value_AOV'].quantile(0.99)
        p99_cri = self.data['Churn_Risk_Index_CRI'].quantile(0.99)
        plot_data = self.data[(self.data['Average_Order_Value_AOV'] < p99_aov) & 
                              (self.data['Churn_Risk_Index_CRI'] < p99_cri)].copy()
        
        # Sort so small bubbles render on top of big bubbles
        plot_data = plot_data.sort_values('Total_Lifetime_Spend', ascending=False)
        
        scatter = sns.scatterplot(
            data=plot_data, 
            x='Churn_Risk_Index_CRI', 
            y='Average_Order_Value_AOV',
            hue=self.cluster_col,
            size='Total_Lifetime_Spend',
            sizes=(20, 800),
            palette=self.palette,
            alpha=0.6,
            edgecolor='white'
        )
        
        # Draw critical threshold lines
        plt.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, label='High Churn Risk (>1.5x AIPT)')
        plt.axhline(y=plot_data['Average_Order_Value_AOV'].median(), color='gray', linestyle=':', alpha=0.5)
        
        plt.title('Strategic Matrix: Churn Risk vs. Order Value', fontsize=16, pad=15)
        plt.xlabel('Churn Risk Index (Days Since Purchase / Avg Inter-Purchase Time)', fontsize=12)
        plt.ylabel('Average Order Value ($)', fontsize=12)
        
        # Cleanup legend (extract just clusters, ignore sizes)
        handles, labels = scatter.get_legend_handles_labels()
        cluster_handles = handles[1:self.n_clusters+1]
        cluster_labels = labels[1:self.n_clusters+1]
        plt.legend(cluster_handles, cluster_labels, title="Cluster", loc='upper right')
        
        plt.tight_layout()
        plt.show()

    def plot_deep_feature_heatmap(self):
        """Heatmap of detailed structural averages across clusters."""
        heatmap_features = [
            'Total_Lifetime_Spend', 'Total_Lifetime_Transactions',
            'Average_Unit_Retail_AUR', 'Category_Concentration_Score',
            'Department_Penetration_Pct', 'Active_Days_Ratio',
            'Average_InterPurchase_Time_AIPT', 'Recency_Days'
        ]
        
        valid_features = [f for f in heatmap_features if f in self.data.columns]
        cluster_means = self.data.groupby(self.cluster_col)[valid_features].mean()
        
        # Scale columns individually to 0-1 so colors highlight the highest cluster per metric
        scaler = MinMaxScaler()
        scaled_means = pd.DataFrame(
            scaler.fit_transform(cluster_means),
            columns=cluster_means.columns,
            index=cluster_means.index
        )
        
        plt.figure(figsize=(14, 8))
        # Annotate with the raw mathematical means, but color based on scaled relative values
        sns.heatmap(scaled_means, annot=cluster_means.round(2), 
                    cmap='YlOrRd', fmt='g', linewidths=1, 
                    cbar_kws={'label': 'Relative Intensity (0 to 1)'})
        
        plt.title('Cluster Profiling Heatmap (Annotated with Raw Averages)', fontsize=16, pad=15)
        plt.ylabel('Cluster ID', fontsize=12)
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        plt.show()

    def generate_all(self):
        print("1. Plotting Cluster Population Sizes...")
        self.plot_cluster_sizes()
        print("2. Generating Shopper DNA Radar...")
        self.plot_behavioral_radar()
        print("3. Plotting Risk vs. Value Matrix...")
        self.plot_risk_vs_value_matrix()
        print("4. Generating Deep Feature Heatmap...")
        self.plot_deep_feature_heatmap()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Point this to the output of your pipeline
    data_file = "customer_segments_advanced.parquet"
    
    try:
        visualizer = AdvancedSegmentationVisualizer(data_file)
        visualizer.generate_all()
    except FileNotFoundError:
        print(f"Error: Could not find '{data_file}'. Please run the segmentation script first.")