import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Dashboard.css';

const UserDashboard = ({ userRole, userId, onLogout }) => {
  const [stats, setStats] = useState({
    myApplications: 0,
    approved: 0,
    rejected: 0,
    underReview: 0
  });
  const [myApplications, setMyApplications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedApplication, setSelectedApplication] = useState(null);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    fetchUserData();
  }, [userId]);

  const fetchUserData = async () => {
    try {
      // In production, fetch from: `/api/applications/user/${userId}`
      // Mock user-specific data
      setTimeout(() => {
        const userApps = [
          { id: 'LA101', type: 'Personal Loan', amount: 50000, status: 'approved', date: '2025-10-20', interestRate: '10.5%' },
          { id: 'LA102', type: 'Home Loan', amount: 500000, status: 'under_review', date: '2025-10-25', interestRate: 'TBD' },
          { id: 'LA103', type: 'Car Loan', amount: 300000, status: 'rejected', date: '2025-10-15', interestRate: 'N/A' }
        ];

        setMyApplications(userApps);

        setStats({
          myApplications: userApps.length,
          approved: userApps.filter(a => a.status === 'approved').length,
          rejected: userApps.filter(a => a.status === 'rejected').length,
          underReview: userApps.filter(a => a.status === 'under_review').length
        });

        setLoading(false);
      }, 500);
    } catch (error) {
      console.error('Error fetching user data:', error);
      setLoading(false);
    }
  };

  const getStatusBadge = (status) => {
    const statusClasses = {
      approved: 'badge-success',
      rejected: 'badge-danger',
      under_review: 'badge-info'
    };
    return <span className={`status-badge ${statusClasses[status]}`}>{status.replace('_', ' ')}</span>;
  };

  const handleViewDetails = (app) => {
    setSelectedApplication(app);
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setSelectedApplication(null);
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <div className="sidebar">
        <h2>Loan ERP</h2>
        <div className="nav-menu">
          <Link to="/dashboard" className="nav-item active">My Applications</Link>
          <Link to="/chat" className="nav-item">Chat Assistant</Link>
          <Link to="/analytics" className="nav-item">Analytics</Link>
        </div>
        <button onClick={onLogout} className="logout-btn">Logout</button>
      </div>

      <div className="main-content">
        <h1>My Loan Applications</h1>

        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
              📄
            </div>
            <div className="stat-details">
              <h3>Total Applications</h3>
              <p className="stat-value">{stats.myApplications}</p>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)' }}>
              ✓
            </div>
            <div className="stat-details">
              <h3>Approved</h3>
              <p className="stat-value">{stats.approved}</p>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)' }}>
              ✗
            </div>
            <div className="stat-details">
              <h3>Rejected</h3>
              <p className="stat-value">{stats.rejected}</p>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)' }}>
              ⏳
            </div>
            <div className="stat-details">
              <h3>Under Review</h3>
              <p className="stat-value">{stats.underReview}</p>
            </div>
          </div>
        </div>

        <div className="content-card">
          <h2>My Applications</h2>
          <div className="table-container">
            <table className="applications-table">
              <thead>
                <tr>
                  <th>Application ID</th>
                  <th>Loan Type</th>
                  <th>Amount</th>
                  <th>Status</th>
                  <th>Applied Date</th>
                  <th>Interest Rate</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {myApplications.map((app) => (
                  <tr key={app.id}>
                    <td><strong>{app.id}</strong></td>
                    <td>{app.type}</td>
                    <td>₹{app.amount.toLocaleString()}</td>
                    <td>{getStatusBadge(app.status)}</td>
                    <td>{app.date}</td>
                    <td>{app.interestRate}</td>
                    <td>
                      <button className="action-btn view-btn" onClick={() => handleViewDetails(app)}>View Details</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {myApplications.length === 0 && (
            <div style={{ textAlign: 'center', padding: '40px', color: '#6b7280' }}>
              <p>No applications found. Start by applying for a new loan!</p>
              <Link to="/chat">
                <button className="action-btn view-btn" style={{ marginTop: '20px' }}>
                  Apply for Loan
                </button>
              </Link>
            </div>
          )}
        </div>

        <div className="content-card">
          <h2>Need Help?</h2>
          <p style={{ color: '#6b7280', marginBottom: '15px' }}>
            Have questions about your application? Our AI assistant is here to help!
          </p>
          <Link to="/chat">
            <button className="action-btn view-btn">
              Chat with AI Assistant
            </button>
          </Link>
        </div>

        {showModal && selectedApplication && (
          <div className="modal-overlay" onClick={closeModal}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h2>Application Details</h2>
                <button className="modal-close" onClick={closeModal}>&times;</button>
              </div>
              <div className="modal-body">
                <div className="detail-row">
                  <span className="detail-label">Application ID:</span>
                  <span className="detail-value">{selectedApplication.id}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Loan Type:</span>
                  <span className="detail-value">{selectedApplication.type}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Amount:</span>
                  <span className="detail-value">₹{selectedApplication.amount.toLocaleString()}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Status:</span>
                  <span className="detail-value">{getStatusBadge(selectedApplication.status)}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Applied Date:</span>
                  <span className="detail-value">{selectedApplication.date}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Interest Rate:</span>
                  <span className="detail-value">{selectedApplication.interestRate}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Processing Time:</span>
                  <span className="detail-value">
                    {selectedApplication.status === 'approved' ? '2-3 business days' :
                     selectedApplication.status === 'rejected' ? 'N/A' :
                     '3-5 business days'}
                  </span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Remarks:</span>
                  <span className="detail-value">
                    {selectedApplication.status === 'approved' ? 'Your loan application has been approved. Documents will be sent to your email.' :
                     selectedApplication.status === 'rejected' ? 'Application rejected due to insufficient credit score. Please try again after 3 months.' :
                     'Your application is under review. You will be notified within 3-5 business days.'}
                  </span>
                </div>
              </div>
              <div className="modal-footer">
                <button className="action-btn view-btn" onClick={closeModal}>Close</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UserDashboard;
