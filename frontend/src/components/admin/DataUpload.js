import { Link } from 'react-router-dom';
import React, { useState } from 'react';
import axios from 'axios';
import './DataUpload.css';

const DataUpload = ({ onLogout }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState('');

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Accept CSV and all Excel formats (.xls, .xlsx, .xlsm, .xlsb)
      const fileName = file.name.toLowerCase();
      if (fileName.endsWith('.csv') ||
          fileName.endsWith('.xlsx') ||
          fileName.endsWith('.xls') ||
          fileName.endsWith('.xlsm') ||
          fileName.endsWith('.xlsb')) {
        setSelectedFile(file);
        setError('');
        setUploadResult(null);
        setAnalysisResult(null);
      } else {
        setError('Please select a CSV or Excel file (.csv, .xls, .xlsx, .xlsm, .xlsb)');
        setSelectedFile(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('/api/data/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setUploadResult(response.data);
      setUploading(false);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
      setUploading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!uploadResult) {
      setError('Please upload a file first');
      return;
    }

    setAnalyzing(true);
    setError('');

    try {
      const response = await axios.post('/api/data/analyze', {
        file_id: uploadResult.file_id
      });

      setAnalysisResult(response.data);
      setAnalyzing(false);
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
      setAnalyzing(false);
    }
  };

  const handleDownloadReport = (format = 'json') => {
    if (!analysisResult) {
      setError('No analysis results to download');
      return;
    }

    try {
      let content, filename, mimeType;

      if (format === 'json') {
        // Download as JSON
        content = JSON.stringify(analysisResult, null, 2);
        filename = `loan_analysis_report_${new Date().toISOString().split('T')[0]}.json`;
        mimeType = 'application/json';
      } else if (format === 'csv') {
        // Convert predictions to CSV
        const predictions = analysisResult.predictions || [];
        if (predictions.length === 0) {
          setError('No predictions data to export');
          return;
        }

        // CSV Headers
        const headers = ['Application ID', 'Prediction', 'Confidence', 'Credit Score', 'Loan Amount', 'Income', 'Reasoning'];
        const csvRows = [headers.join(',')];

        // CSV Data
        predictions.forEach(pred => {
          const row = [
            pred.application_id,
            pred.prediction,
            (pred.confidence * 100).toFixed(2) + '%',
            pred.credit_score,
            pred.loan_amount,
            pred.income,
            `"${pred.explainability?.reasoning || 'N/A'}"`
          ];
          csvRows.push(row.join(','));
        });

        content = csvRows.join('\n');
        filename = `loan_predictions_${new Date().toISOString().split('T')[0]}.csv`;
        mimeType = 'text/csv';
      } else if (format === 'txt') {
        // Download as formatted text report
        const report = generateTextReport(analysisResult);
        content = report;
        filename = `loan_analysis_report_${new Date().toISOString().split('T')[0]}.txt`;
        mimeType = 'text/plain';
      }

      // Create blob and download
      const blob = new Blob([content], { type: mimeType });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download error:', err);
      setError('Failed to download report. Please try again.');
    }
  };

  const generateTextReport = (data) => {
    const lines = [];
    lines.push('═══════════════════════════════════════════════════════════');
    lines.push('           LOAN DATA ANALYSIS REPORT');
    lines.push('═══════════════════════════════════════════════════════════');
    lines.push(`Generated: ${new Date().toLocaleString()}`);
    lines.push('');

    lines.push('--- SUMMARY STATISTICS ---');
    lines.push(`Total Records: ${data.total_records}`);
    lines.push(`Data Quality Score: ${(data.quality_score * 100).toFixed(1)}%`);
    lines.push(`Missing Values: ${data.missing_values}`);
    lines.push(`Duplicates: ${data.duplicates}`);
    lines.push(`Approval Rate: ${(data.approval_rate * 100).toFixed(1)}%`);
    lines.push('');

    lines.push('--- MODEL PERFORMANCE METRICS ---');
    lines.push(`Accuracy: ${(data.model_metrics.accuracy * 100).toFixed(2)}%`);
    lines.push(`Precision: ${(data.model_metrics.precision * 100).toFixed(2)}%`);
    lines.push(`Recall: ${(data.model_metrics.recall * 100).toFixed(2)}%`);
    lines.push(`F1-Score: ${(data.model_metrics.f1_score * 100).toFixed(2)}%`);
    lines.push('');

    lines.push('--- PREDICTIONS (Top 10) ---');
    const predictions = data.predictions || [];
    predictions.slice(0, 10).forEach((pred, idx) => {
      lines.push('');
      lines.push(`${idx + 1}. Application ID: ${pred.application_id}`);
      lines.push(`   Prediction: ${pred.prediction}`);
      lines.push(`   Confidence: ${(pred.confidence * 100).toFixed(1)}%`);
      lines.push(`   Credit Score: ${pred.credit_score}`);
      lines.push(`   Loan Amount: $${pred.loan_amount.toLocaleString()}`);
      lines.push(`   Income: $${pred.income.toLocaleString()}`);
      lines.push(`   Reasoning: ${pred.explainability?.reasoning || 'N/A'}`);
    });

    lines.push('');
    lines.push('═══════════════════════════════════════════════════════════');
    lines.push('                    END OF REPORT');
    lines.push('═══════════════════════════════════════════════════════════');

    return lines.join('\n');
  };

  return (
    <div className="page-container">
      <div className="sidebar">
        <h2>Loan ERP</h2>
        <div className="nav-menu">
          <Link to="/dashboard" className="nav-item">Dashboard</Link>
          <Link to="/chat" className="nav-item">Chat Assistant</Link>
          <Link to="/analytics" className="nav-item">Analytics</Link>
          <Link to="/upload" className="nav-item active">Data Upload</Link>
        </div>
        <button onClick={onLogout} className="logout-btn">Logout</button>
      </div>

      <div className="main-content">
        <h1>Data Upload & Analysis</h1>

        <div className="content-card">
          <h2>Upload Dataset</h2>
          <p className="description">
            Upload loan application data in CSV or Excel format for training and analysis.
            Supported formats: .csv, .xls, .xlsx, .xlsm, .xlsb
          </p>

          <div className="upload-section">
            <div className="file-input-wrapper">
              <input
                type="file"
                id="file-input"
                accept=".csv,.xls,.xlsx,.xlsm,.xlsb,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                onChange={handleFileSelect}
                className="file-input"
              />
              <label htmlFor="file-input" className="file-label">
                📁 Choose File
              </label>
              {selectedFile && (
                <span className="file-name">{selectedFile.name}</span>
              )}
            </div>

            <button
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
              className="upload-btn"
            >
              {uploading ? 'Uploading...' : '⬆️ Upload'}
            </button>

            {uploadResult && (
              <button
                onClick={handleAnalyze}
                disabled={analyzing}
                className="analyze-btn"
              >
                {analyzing ? 'Analyzing...' : '🔍 Analyze Data'}
              </button>
            )}
          </div>

          {error && <div className="error-message">{error}</div>}

          {uploadResult && (
            <div className="success-message">
              ✓ File uploaded successfully!
              <br />
              Rows: {uploadResult.rows} | Columns: {uploadResult.columns}
            </div>
          )}
        </div>

        {analysisResult && (
          <>
            <div className="content-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>Data Analysis Results</h2>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button
                    onClick={() => handleDownloadReport('csv')}
                    className="upload-btn"
                    style={{ background: '#10b981' }}
                  >
                    📊 Download CSV
                  </button>
                  <button
                    onClick={() => handleDownloadReport('txt')}
                    className="upload-btn"
                    style={{ background: '#667eea' }}
                  >
                    📄 Download Report
                  </button>
                  <button
                    onClick={() => handleDownloadReport('json')}
                    className="upload-btn"
                    style={{ background: '#f59e0b' }}
                  >
                    📦 Download JSON
                  </button>
                </div>
              </div>

              <div className="analysis-stats">
                <div className="stat-item">
                  <span className="stat-label">Total Records</span>
                  <span className="stat-value">{analysisResult.total_records}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Data Quality Score</span>
                  <span className="stat-value">{(analysisResult.quality_score * 100).toFixed(1)}%</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Missing Values</span>
                  <span className="stat-value">{analysisResult.missing_values}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Duplicates</span>
                  <span className="stat-value">{analysisResult.duplicates}</span>
                </div>
              </div>
            </div>

            <div className="content-card">
              <h2>Model Predictions with Explainability</h2>

              {analysisResult.predictions && analysisResult.predictions.slice(0, 5).map((pred, idx) => (
                <div key={idx} className="prediction-card">
                  <div className="prediction-header">
                    <h3>Application #{pred.application_id}</h3>
                    <span className={`prediction-badge ${pred.prediction === 'APPROVED' ? 'approved' : 'rejected'}`}>
                      {pred.prediction}
                    </span>
                  </div>

                  <div className="prediction-details">
                    <div className="detail-item">
                      <strong>Confidence:</strong>
                      <div className="confidence-bar">
                        <div
                          className="confidence-fill"
                          style={{ width: `${pred.confidence * 100}%` }}
                        ></div>
                        <span className="confidence-text">{(pred.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    <div className="detail-item">
                      <strong>Credit Score:</strong> {pred.credit_score}
                    </div>

                    <div className="detail-item">
                      <strong>Loan Amount:</strong> ${pred.loan_amount.toLocaleString()}
                    </div>

                    <div className="detail-item">
                      <strong>Income:</strong> ${pred.income.toLocaleString()}
                    </div>
                  </div>

                  {pred.explainability && (
                    <div className="explainability-section">
                      <h4>🔍 Explainability Analysis</h4>

                      <div className="explanation-text">
                        <strong>Reasoning:</strong>
                        <p>{pred.explainability.reasoning}</p>
                      </div>

                      <div className="feature-importance">
                        <strong>Key Factors (Feature Importance):</strong>
                        <div className="factors-list">
                          {pred.explainability.feature_importance.map((factor, fIdx) => (
                            <div key={fIdx} className="factor-item">
                              <span className="factor-name">{factor.feature}</span>
                              <div className="factor-bar">
                                <div
                                  className="factor-fill"
                                  style={{
                                    width: `${factor.importance * 100}%`,
                                    background: factor.impact === 'positive' ? '#10b981' : '#ef4444'
                                  }}
                                ></div>
                              </div>
                              <span className="factor-value">{(factor.importance * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {pred.explainability.shap_values && (
                        <div className="shap-section">
                          <strong>SHAP Values:</strong>
                          <p className="shap-desc">
                            Shows how each feature contributes to the final prediction
                          </p>
                          <div className="shap-chart">
                            {Object.entries(pred.explainability.shap_values).map(([feature, value]) => (
                              <div key={feature} className="shap-item">
                                <span>{feature}</span>
                                <span className={value > 0 ? 'positive' : 'negative'}>
                                  {value > 0 ? '+' : ''}{value.toFixed(3)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {pred.explainability.recommendations && (
                        <div className="recommendations">
                          <strong>💡 Recommendations:</strong>
                          <ul>
                            {pred.explainability.recommendations.map((rec, rIdx) => (
                              <li key={rIdx}>{rec}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="content-card">
              <h2>Model Performance Metrics</h2>
              <div className="metrics-grid">
                <div className="metric-card">
                  <h3>Accuracy</h3>
                  <p className="metric-value">{(analysisResult.model_metrics.accuracy * 100).toFixed(2)}%</p>
                </div>
                <div className="metric-card">
                  <h3>Precision</h3>
                  <p className="metric-value">{(analysisResult.model_metrics.precision * 100).toFixed(2)}%</p>
                </div>
                <div className="metric-card">
                  <h3>Recall</h3>
                  <p className="metric-value">{(analysisResult.model_metrics.recall * 100).toFixed(2)}%</p>
                </div>
                <div className="metric-card">
                  <h3>F1-Score</h3>
                  <p className="metric-value">{(analysisResult.model_metrics.f1_score * 100).toFixed(2)}%</p>
                </div>
              </div>
            </div>
          </>
        )}

      </div>
    </div>
  );
};

export default DataUpload;
