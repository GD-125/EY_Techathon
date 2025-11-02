"""
Model Evaluation and Visualization Module
Generates professional evaluation reports with visualizations
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

logger = logging.getLogger(__name__)

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """
    Comprehensive model evaluation with professional visualizations
    """

    def __init__(self, output_dir: str = "evaluation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "model",
        feature_names: Optional[List[str]] = None,
        feature_importances: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with visualizations

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model
            feature_names: List of feature names
            feature_importances: Feature importance values

        Returns:
            Dictionary with evaluation metrics and plot paths
        """
        logger.info(f"Evaluating model: {model_name}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_id = f"{model_name}_{timestamp}"

        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)

        # Generate visualizations
        plots = {}

        # 1. Confusion Matrix
        cm_path = self._plot_confusion_matrix(
            y_true, y_pred,
            save_path=self.output_dir / "plots" / f"{report_id}_confusion_matrix.png"
        )
        plots['confusion_matrix'] = str(cm_path)

        # 2. ROC Curve
        roc_path = self._plot_roc_curve(
            y_true, y_pred_proba,
            save_path=self.output_dir / "plots" / f"{report_id}_roc_curve.png"
        )
        plots['roc_curve'] = str(roc_path)

        # 3. Precision-Recall Curve
        pr_path = self._plot_precision_recall_curve(
            y_true, y_pred_proba,
            save_path=self.output_dir / "plots" / f"{report_id}_pr_curve.png"
        )
        plots['precision_recall_curve'] = str(pr_path)

        # 4. Metrics Dashboard
        dashboard_path = self._plot_metrics_dashboard(
            metrics,
            save_path=self.output_dir / "plots" / f"{report_id}_dashboard.png"
        )
        plots['dashboard'] = str(dashboard_path)

        # 5. Feature Importance (if available)
        if feature_names and feature_importances is not None:
            fi_path = self._plot_feature_importance(
                feature_names, feature_importances,
                save_path=self.output_dir / "plots" / f"{report_id}_feature_importance.png"
            )
            plots['feature_importance'] = str(fi_path)

        # 6. Prediction Distribution
        dist_path = self._plot_prediction_distribution(
            y_true, y_pred_proba,
            save_path=self.output_dir / "plots" / f"{report_id}_prediction_dist.png"
        )
        plots['prediction_distribution'] = str(dist_path)

        # Generate comprehensive report
        report = {
            'model_name': model_name,
            'timestamp': timestamp,
            'report_id': report_id,
            'metrics': metrics,
            'plots': plots,
            'n_samples': len(y_true),
            'class_distribution': {
                'positive': int(np.sum(y_true)),
                'negative': int(len(y_true) - np.sum(y_true))
            }
        }

        # Save report
        report_path = self.output_dir / "reports" / f"{report_id}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation complete. Report saved: {report_path}")

        return report

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
            'average_precision': float(average_precision_score(y_true, y_pred_proba))
        }

        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics.update({
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        })

        return metrics

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Path
    ) -> Path:
        """Plot confusion matrix with annotations"""

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Approved', 'Rejected'],
            yticklabels=['Approved', 'Rejected'],
            cbar_kws={'label': 'Count'},
            ax=ax
        )

        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)

        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = (cm[i, j] / total) * 100
                ax.text(
                    j + 0.5, i + 0.7,
                    f'({percentage:.1f}%)',
                    ha='center', va='center',
                    color='gray', fontsize=9
                )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def _plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Path
    ) -> Path:
        """Plot ROC curve with AUC"""

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot ROC curve
        ax.plot(fpr, tpr, color='#2E86DE', linewidth=2.5,
                label=f'ROC Curve (AUC = {roc_auc:.4f})')

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2,
                label='Random Classifier')

        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def _plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Path
    ) -> Path:
        """Plot Precision-Recall curve"""

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot PR curve
        ax.plot(recall, precision, color='#10AC84', linewidth=2.5,
                label=f'PR Curve (AP = {avg_precision:.4f})')

        # Plot baseline
        baseline = np.sum(y_true) / len(y_true)
        ax.plot([0, 1], [baseline, baseline], color='gray', linestyle='--',
                linewidth=2, label=f'Baseline (Prevalence = {baseline:.4f})')

        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def _plot_metrics_dashboard(
        self,
        metrics: Dict[str, float],
        save_path: Path
    ) -> Path:
        """Plot comprehensive metrics dashboard"""

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Main Metrics (top row)
        main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for idx, metric in enumerate(main_metrics):
            ax = fig.add_subplot(gs[0, idx if idx < 3 else 0])
            value = metrics[metric]

            # Color based on performance
            if value >= 0.9:
                color = '#10AC84'
            elif value >= 0.8:
                color = '#F79F1F'
            else:
                color = '#EE5A6F'

            ax.text(0.5, 0.5, f'{value:.4f}',
                   ha='center', va='center',
                   fontsize=48, fontweight='bold', color=color)
            ax.text(0.5, 0.15, metric.replace('_', ' ').title(),
                   ha='center', va='center', fontsize=14)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')

        # 2. ROC AUC (middle left)
        ax = fig.add_subplot(gs[1, 0])
        value = metrics['roc_auc']
        color = '#2E86DE'
        ax.text(0.5, 0.5, f'{value:.4f}',
               ha='center', va='center',
               fontsize=48, fontweight='bold', color=color)
        ax.text(0.5, 0.15, 'ROC AUC',
               ha='center', va='center', fontsize=14)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')

        # 3. Confusion Matrix Metrics (middle center & right)
        cm_metrics = [
            ('True Positives', metrics['true_positives']),
            ('True Negatives', metrics['true_negatives']),
            ('False Positives', metrics['false_positives']),
            ('False Negatives', metrics['false_negatives'])
        ]

        ax = fig.add_subplot(gs[1, 1:])
        y_pos = np.arange(len(cm_metrics))
        values = [m[1] for m in cm_metrics]
        labels = [m[0] for m in cm_metrics]
        colors = ['#10AC84', '#10AC84', '#EE5A6F', '#EE5A6F']

        bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Count', fontsize=11)
        ax.set_title('Confusion Matrix Breakdown', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value, bar.get_y() + bar.get_height()/2,
                   f' {int(value)}',
                   va='center', fontsize=10, fontweight='bold')

        # 4. Additional Metrics (bottom row)
        additional = [
            ('Specificity', metrics['specificity']),
            ('Sensitivity', metrics['sensitivity']),
            ('Avg Precision', metrics['average_precision'])
        ]

        for idx, (name, value) in enumerate(additional):
            ax = fig.add_subplot(gs[2, idx])
            color = '#5F27CD'
            ax.text(0.5, 0.5, f'{value:.4f}',
                   ha='center', va='center',
                   fontsize=36, fontweight='bold', color=color)
            ax.text(0.5, 0.15, name,
                   ha='center', va='center', fontsize=12)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')

        fig.suptitle('Model Performance Dashboard',
                    fontsize=20, fontweight='bold', y=0.98)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def _plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        save_path: Path,
        top_n: int = 20
    ) -> Path:
        """Plot top N feature importances"""

        # Get top N features
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create horizontal bar chart
        y_pos = np.arange(len(top_features))
        colors = plt.cm.viridis(top_importances / top_importances.max())

        bars = ax.barh(y_pos, top_importances, color=colors, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances',
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_importances)):
            ax.text(value, bar.get_y() + bar.get_height()/2,
                   f' {value:.6f}',
                   va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def _plot_prediction_distribution(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Path
    ) -> Path:
        """Plot distribution of prediction probabilities"""

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Split by true class
        approved_probs = y_pred_proba[y_true == 0]
        rejected_probs = y_pred_proba[y_true == 1]

        # Plot histograms
        axes[0].hist(approved_probs, bins=50, alpha=0.7, color='#10AC84',
                    edgecolor='black', label='True Approved')
        axes[0].hist(rejected_probs, bins=50, alpha=0.7, color='#EE5A6F',
                    edgecolor='black', label='True Rejected')
        axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2,
                       label='Decision Threshold')
        axes[0].set_xlabel('Predicted Probability of Rejection', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Prediction Probability Distribution by True Class',
                         fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Plot box plots
        data_to_plot = [approved_probs, rejected_probs]
        bp = axes[1].boxplot(data_to_plot, labels=['True Approved', 'True Rejected'],
                             patch_artist=True, widths=0.6)

        colors = ['#10AC84', '#EE5A6F']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1].axhline(0.5, color='black', linestyle='--', linewidth=2,
                       label='Decision Threshold')
        axes[1].set_ylabel('Predicted Probability of Rejection', fontsize=11)
        axes[1].set_title('Prediction Probability Box Plots',
                         fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def compare_models(
        self,
        reports: List[Dict[str, Any]],
        save_comparison: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple model evaluation reports

        Args:
            reports: List of evaluation report dictionaries
            save_comparison: Whether to save comparison plots

        Returns:
            Comparison results with best model selection
        """
        logger.info(f"Comparing {len(reports)} models")

        if len(reports) < 2:
            logger.warning("Need at least 2 models to compare")
            return {'best_model': reports[0] if reports else None}

        # Create comparison DataFrame
        comparison_data = []
        for report in reports:
            metrics = report['metrics']
            comparison_data.append({
                'model': report['model_name'],
                'timestamp': report['timestamp'],
                'report_id': report['report_id'],
                **metrics
            })

        df_comparison = pd.DataFrame(comparison_data)

        # Calculate composite score for ranking
        # Weighted average: ROC AUC (40%), F1 (30%), Precision (20%), Recall (10%)
        df_comparison['composite_score'] = (
            df_comparison['roc_auc'] * 0.4 +
            df_comparison['f1_score'] * 0.3 +
            df_comparison['precision'] * 0.2 +
            df_comparison['recall'] * 0.1
        )

        # Sort by composite score
        df_comparison = df_comparison.sort_values('composite_score', ascending=False)

        # Identify best model
        best_model_idx = df_comparison.index[0]
        best_model_report = reports[best_model_idx]

        # Generate comparison plots if requested
        if save_comparison:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            comparison_path = self._plot_model_comparison(
                df_comparison,
                save_path=self.output_dir / "comparisons" / f"comparison_{timestamp}.png"
            )
        else:
            comparison_path = None

        comparison_result = {
            'best_model': best_model_report,
            'ranking': df_comparison[['model', 'composite_score', 'roc_auc',
                                     'f1_score', 'accuracy']].to_dict('records'),
            'comparison_plot': str(comparison_path) if comparison_path else None,
            'summary': {
                'total_models': len(reports),
                'best_model_name': best_model_report['model_name'],
                'best_composite_score': float(df_comparison.iloc[0]['composite_score']),
                'improvement_over_worst': float(
                    df_comparison.iloc[0]['composite_score'] -
                    df_comparison.iloc[-1]['composite_score']
                )
            }
        }

        # Save comparison report
        if save_comparison:
            comparison_report_path = (
                self.output_dir / "comparisons" / f"comparison_report_{timestamp}.json"
            )
            with open(comparison_report_path, 'w') as f:
                json.dump(comparison_result, f, indent=2)

        logger.info(f"Best model: {best_model_report['model_name']} "
                   f"(score: {comparison_result['summary']['best_composite_score']:.4f})")

        return comparison_result

    def _plot_model_comparison(
        self,
        df_comparison: pd.DataFrame,
        save_path: Path
    ) -> Path:
        """Plot model comparison charts"""

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score',
                             'roc_auc', 'composite_score']

        # 1. Composite Score Ranking (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        models = df_comparison['model'].values
        scores = df_comparison['composite_score'].values
        colors = plt.cm.RdYlGn(scores / scores.max())

        bars = ax1.barh(range(len(models)), scores, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(models)))
        ax1.set_yticklabels(models)
        ax1.invert_yaxis()
        ax1.set_xlabel('Composite Score', fontsize=11)
        ax1.set_title('Model Ranking by Composite Score',
                     fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar, score in zip(bars, scores):
            ax1.text(score, bar.get_y() + bar.get_height()/2,
                    f' {score:.4f}', va='center', fontsize=9, fontweight='bold')

        # 2. Metrics Comparison (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(models))
        width = 0.15

        for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1_score']):
            offset = (i - 1.5) * width
            values = df_comparison[metric].values
            ax2.bar(x + offset, values, width, label=metric.capitalize(), alpha=0.8)

        ax2.set_xlabel('Models', fontsize=11)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Metrics Comparison Across Models',
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 1.1])

        # 3. ROC AUC Comparison (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        auc_scores = df_comparison['roc_auc'].values
        colors = plt.cm.plasma(auc_scores / auc_scores.max())

        bars = ax3.bar(range(len(models)), auc_scores, color=colors, alpha=0.8)
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.set_ylabel('ROC AUC Score', fontsize=11)
        ax3.set_title('ROC AUC Comparison', fontsize=13, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, 1.1])

        # Add value labels
        for bar, score in zip(bars, auc_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{score:.4f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        # 4. Metrics Heatmap (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        metrics_matrix = df_comparison[metrics_to_compare].values.T

        im = ax4.imshow(metrics_matrix, aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=1)
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.set_yticks(range(len(metrics_to_compare)))
        ax4.set_yticklabels([m.replace('_', ' ').title()
                            for m in metrics_to_compare])
        ax4.set_title('Metrics Heatmap', fontsize=13, fontweight='bold')

        # Add values to heatmap
        for i in range(len(metrics_to_compare)):
            for j in range(len(models)):
                text = ax4.text(j, i, f'{metrics_matrix[i, j]:.3f}',
                              ha='center', va='center', color='black', fontsize=8)

        plt.colorbar(im, ax=ax4, label='Score')

        fig.suptitle('Comprehensive Model Comparison',
                    fontsize=18, fontweight='bold', y=0.98)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path
