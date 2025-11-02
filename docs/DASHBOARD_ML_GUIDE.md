```markdown
# Dashboard ML Integration Guide

## Overview

Your loan ERP system now includes professional ML capabilities with:

✅ **Professional Visualizations**
- Confusion Matrix with annotations
- ROC Curve with AUC scores
- Precision-Recall Curves
- Feature Importance Charts
- Prediction Distribution Plots
- Comprehensive Metrics Dashboard

✅ **Automatic Best Model Selection**
- Trains multiple models
- Compares performance with composite scoring
- Automatically selects and saves best model
- Dashboard loads best model on startup

✅ **File Upload & Analysis**
- Upload CSV/Excel files
- Batch analysis with detailed insights
- Export results
- Risk distribution analysis

---

## Quick Start

### 1. Train Models with Evaluation

Run the enhanced training script:

```bash
cd backend
python train_and_evaluate.py
```

**What it does:**
- Trains Random Forest and/or Gradient Boosting models
- Generates professional evaluation visualizations
- Compares models and selects the best one
- Saves best model info for dashboard

**Output:**
```
evaluation_reports/
├── plots/
│   ├── model_confusion_matrix.png
│   ├── model_roc_curve.png
│   ├── model_pr_curve.png
│   ├── model_dashboard.png
│   ├── model_feature_importance.png
│   └── model_prediction_dist.png
├── reports/
│   └── model_report.json
└── comparisons/
    └── comparison_*.png

models/
├── best_model_info.json  ← Dashboard uses this
└── [model directories]
```

### 2. View Evaluation Visualizations

After training, check the professional plots:

**Confusion Matrix** (`confusion_matrix.png`)
- Shows True/False Positives/Negatives
- Percentages and counts
- Color-coded heatmap

**ROC Curve** (`roc_curve.png`)
- AUC score displayed
- Reference diagonal line
- Professional styling

**Metrics Dashboard** (`dashboard.png`)
- All key metrics in one view
- Color-coded by performance
- Confusion matrix breakdown
- Additional metrics

**Feature Importance** (`feature_importance.png`)
- Top 20 most important features
- Ranked by importance score
- Color-coded bars

### 3. Start Dashboard with ML

The dashboard automatically loads the best model on startup.

---

## Dashboard API Endpoints

### Get Best Model Info

**GET** `/api/dashboard/ml/best-model`

```json
{
  "best_model_id": "RandomForest_HomeCredit_20250131_120000",
  "model_name": "RandomForest_HomeCredit",
  "composite_score": 0.9234,
  "metrics": {
    "accuracy": 0.9187,
    "precision": 0.8851,
    "recall": 0.9012,
    "f1_score": 0.8930,
    "roc_auc": 0.9471
  },
  "has_model": true
}
```

### Analyze Uploaded File

**POST** `/api/dashboard/ml/analyze-file`

Upload CSV/Excel file with loan applications.

**Response:**
```json
{
  "status": "success",
  "file_name": "applications.csv",
  "insights": {
    "total_analyzed": 1000,
    "approval_summary": {
      "total_approved": 750,
      "total_rejected": 250,
      "approval_rate": 0.75
    },
    "risk_analysis": {
      "average_risk": 0.234,
      "median_risk": 0.198,
      "risk_distribution": {
        "very_low": 300,
        "low": 450,
        "medium": 150,
        "high": 80,
        "very_high": 20
      }
    },
    "confidence_metrics": {
      "average_confidence": 0.856,
      "high_confidence_count": 820,
      "low_confidence_count": 45
    },
    "recommendations": {
      "high_risk_count": 100,
      "manual_review_recommended": 120,
      "fast_track_eligible": 680
    }
  },
  "sample_results": [...],
  "full_results_count": 1000
}
```

### Get Model Metrics

**GET** `/api/dashboard/ml/model-metrics`

```json
{
  "primary_metrics": {
    "accuracy": {
      "value": 0.9187,
      "label": "Accuracy",
      "format": "percentage"
    },
    "precision": {...},
    "recall": {...},
    "f1_score": {...}
  },
  "advanced_metrics": {
    "roc_auc": {...},
    "average_precision": {...}
  },
  "confusion_matrix": {
    "true_positives": 8542,
    "true_negatives": 982,
    "false_positives": 143,
    "false_negatives": 333
  },
  "model_info": {
    "name": "RandomForest_HomeCredit",
    "timestamp": "20250131_120000",
    "composite_score": 0.9234
  }
}
```

### Get Evaluation Plots

**GET** `/api/dashboard/ml/evaluation-plots`

```json
{
  "status": "success",
  "plots": {
    "confusion_matrix": {
      "path": "evaluation_reports/plots/model_confusion_matrix.png",
      "filename": "model_confusion_matrix.png",
      "url": "/api/dashboard/ml/plot/model_confusion_matrix.png"
    },
    "roc_curve": {...},
    "dashboard": {...},
    "feature_importance": {...}
  },
  "model_name": "RandomForest_HomeCredit",
  "timestamp": "20250131_120000"
}
```

### Serve Plot Image

**GET** `/api/dashboard/ml/plot/{filename}`

Returns PNG image for display in dashboard.

### Get System Statistics

**GET** `/api/dashboard/ml/statistics`

```json
{
  "models": {
    "total_models": 5,
    "has_active_model": true
  },
  "evaluations": {
    "total_reports": 3,
    "total_plots": 18
  },
  "exports": {
    "total_exports": 2
  },
  "latest_training": {
    "timestamp": "2025-01-31T12:00:00",
    "n_samples": 50000,
    "metrics": {...}
  }
}
```

---

## Frontend Integration Examples

### Display Model Metrics

```javascript
// Fetch metrics
const response = await fetch('/api/dashboard/ml/model-metrics');
const data = await response.json();

// Display accuracy
const accuracy = data.primary_metrics.accuracy.value;
document.getElementById('accuracy').textContent =
  `${(accuracy * 100).toFixed(2)}%`;

// Display confusion matrix
const cm = data.confusion_matrix;
renderConfusionMatrix(cm);
```

### Show Evaluation Plots

```javascript
// Get available plots
const plotsResponse = await fetch('/api/dashboard/ml/evaluation-plots');
const plots = await plotsResponse.json();

// Display ROC curve
const rocUrl = plots.plots.roc_curve.url;
document.getElementById('roc-plot').innerHTML =
  `<img src="${rocUrl}" alt="ROC Curve" />`;

// Display confusion matrix
const cmUrl = plots.plots.confusion_matrix.url;
document.getElementById('cm-plot').innerHTML =
  `<img src="${cmUrl}" alt="Confusion Matrix" />`;
```

### File Upload and Analysis

```javascript
// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('/api/dashboard/ml/analyze-file', {
  method: 'POST',
  body: formData
});

const result = await response.json();

// Display insights
const insights = result.insights;

// Show approval rate
const approvalRate = insights.approval_summary.approval_rate;
document.getElementById('approval-rate').textContent =
  `${(approvalRate * 100).toFixed(1)}%`;

// Show risk distribution chart
renderRiskDistribution(insights.risk_analysis.risk_distribution);

// Show recommendations
const fastTrack = insights.recommendations.fast_track_eligible;
const manualReview = insights.recommendations.manual_review_recommended;

document.getElementById('fast-track-count').textContent = fastTrack;
document.getElementById('manual-review-count').textContent = manualReview;
```

### React Component Example

```jsx
import React, { useState, useEffect } from 'react';

function MLDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [plots, setPlots] = useState(null);

  useEffect(() => {
    // Load metrics
    fetch('/api/dashboard/ml/model-metrics')
      .then(res => res.json())
      .then(data => setMetrics(data));

    // Load plots
    fetch('/api/dashboard/ml/evaluation-plots')
      .then(res => res.json())
      .then(data => setPlots(data));
  }, []);

  if (!metrics || !plots) return <div>Loading...</div>;

  return (
    <div className="ml-dashboard">
      <h2>Model Performance</h2>

      {/* Metrics Cards */}
      <div className="metrics-grid">
        {Object.entries(metrics.primary_metrics).map(([key, metric]) => (
          <MetricCard
            key={key}
            label={metric.label}
            value={metric.value}
            format={metric.format}
          />
        ))}
      </div>

      {/* Visualization Grid */}
      <div className="plots-grid">
        <PlotCard
          title="Confusion Matrix"
          imageUrl={plots.plots.confusion_matrix.url}
        />
        <PlotCard
          title="ROC Curve"
          imageUrl={plots.plots.roc_curve.url}
        />
        <PlotCard
          title="Metrics Dashboard"
          imageUrl={plots.plots.dashboard.url}
        />
        <PlotCard
          title="Feature Importance"
          imageUrl={plots.plots.feature_importance.url}
        />
      </div>

      {/* File Upload */}
      <FileAnalysisSection />
    </div>
  );
}

function MetricCard({ label, value, format }) {
  const displayValue = format === 'percentage'
    ? `${(value * 100).toFixed(2)}%`
    : value.toFixed(4);

  const colorClass = value >= 0.9 ? 'excellent' : value >= 0.8 ? 'good' : 'fair';

  return (
    <div className={`metric-card ${colorClass}`}>
      <div className="metric-value">{displayValue}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

function PlotCard({ title, imageUrl }) {
  return (
    <div className="plot-card">
      <h3>{title}</h3>
      <img src={imageUrl} alt={title} />
    </div>
  );
}
```

---

## Model Comparison

When you train multiple models (option 3 in training script), you get:

### Comparison Visualization

**File:** `evaluation_reports/comparisons/comparison_*.png`

**Includes:**
1. **Composite Score Ranking** - Bar chart showing best to worst
2. **Metrics Comparison** - Side-by-side comparison of all metrics
3. **ROC AUC Comparison** - Bar chart of AUC scores
4. **Metrics Heatmap** - Color-coded matrix of all metrics

### Composite Scoring

Models are ranked using weighted average:
- ROC-AUC: 40%
- F1 Score: 30%
- Precision: 20%
- Recall: 10%

Best model is automatically saved to `models/best_model_info.json`.

---

## Professional Visualizations

### 1. Confusion Matrix
- **Format:** PNG, 300 DPI
- **Features:**
  - Heatmap with blue color scale
  - Counts and percentages
  - Labeled axes
  - Professional styling

### 2. ROC Curve
- **Format:** PNG, 300 DPI
- **Features:**
  - AUC score in legend
  - Diagonal reference line
  - Grid for readability
  - Publication-quality

### 3. Precision-Recall Curve
- **Format:** PNG, 300 DPI
- **Features:**
  - Average Precision score
  - Baseline reference
  - Green color scheme

### 4. Metrics Dashboard
- **Format:** PNG, 1600x1000, 300 DPI
- **Features:**
  - 6-panel layout
  - Color-coded metrics
  - Confusion matrix breakdown
  - All key metrics visible

### 5. Feature Importance
- **Format:** PNG, 300 DPI
- **Features:**
  - Top 20 features
  - Horizontal bar chart
  - Importance scores
  - Color gradient

### 6. Prediction Distribution
- **Format:** PNG, 300 DPI
- **Features:**
  - Histograms by true class
  - Box plots
  - Decision threshold line
  - Probability analysis

---

## File Analysis Workflow

### Step 1: Upload File

```javascript
const formData = new FormData();
formData.append('file', file);

const response = await fetch('/api/dashboard/ml/analyze-file', {
  method: 'POST',
  body: formData
});
```

### Step 2: Get Insights

```javascript
const data = await response.json();

const insights = data.insights;
// approval_summary, risk_analysis, confidence_metrics, recommendations
```

### Step 3: Display Results

```javascript
// Show approval rate
showApprovalRate(insights.approval_summary.approval_rate);

// Show risk distribution
renderRiskChart(insights.risk_analysis.risk_distribution);

// Show recommendations
displayRecommendations(insights.recommendations);
```

### Step 4: Export (Optional)

```javascript
// Get detailed export
const exportResponse = await fetch('/api/dashboard/ml/analyze-file-detailed', {
  method: 'POST',
  body: formData
});

const exportData = await exportResponse.json();
const downloadUrl = exportData.download_url;

// Provide download link
window.location.href = downloadUrl;
```

---

## Dashboard Layout Recommendations

### Main Dashboard View

```
┌─────────────────────────────────────────────────────┐
│  ML Model Dashboard                                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │
│  │Accuracy  │ │Precision │ │ Recall   │ │F1 Score│  │
│  │  91.87%  │ │  88.51%  │ │  90.12%  │ │ 89.30% │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘  │
│                                                     │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │  Confusion Matrix    │  │  ROC Curve           │ │
│  │  [Plot Image]        │  │  [Plot Image]        │ │
│  └──────────────────────┘  └──────────────────────┘ │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Feature Importance                          │   │
│  │  [Plot Image]                                │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  File Upload & Analysis                     │    │
│  │  [Upload Button] [Analyze Button]           │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### Analysis Results View

```
┌─────────────────────────────────────────────────────┐
│  Analysis Results - applications.csv                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Total Analyzed: 1,000 applications                │
│                                                     │
│  Approval Summary:                                  │
│  ├─ Approved: 750 (75.0%)                          │
│  └─ Rejected: 250 (25.0%)                          │
│                                                     │
│  Risk Distribution:                                 │
│  [Bar Chart]                                        │
│                                                     │
│  Recommendations:                                   │
│  ├─ Fast Track Eligible: 680                       │
│  └─ Manual Review Needed: 120                      │
│                                                     │
│  [Export Results] [View Details]                   │
└─────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Train your models:**
   ```bash
   python train_and_evaluate.py
   ```

2. **Check visualizations:**
   - Open `evaluation_reports/plots/` folder
   - View professional evaluation charts

3. **Integrate with dashboard:**
   - Use provided API endpoints
   - Display metrics and plots
   - Add file upload functionality

4. **Test file analysis:**
   - Upload sample CSV/Excel
   - View analysis insights
   - Export results

**Your dashboard now has enterprise-grade ML capabilities!** 🎉
```
