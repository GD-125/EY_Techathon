import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Dashboard.css';

const Dashboard = ({ userRole, onLogout }) => {
  const [stats, setStats] = useState({
    totalApplications: 0,
    pendingReview: 0,
    approved: 0,
    rejected: 0,
    totalAmount: 0
  });
  const [recentApplications, setRecentApplications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedApplication, setSelectedApplication] = useState(null);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Mock data for demo
      setTimeout(() => {
        setStats({
          totalApplications: 1247,
          pendingReview: 43,
          approved: 892,
          rejected: 312,
          totalAmount: 45678900
        });

        setRecentApplications([
          { id: 'LA001', name: 'John Doe', amount: 50000, status: 'pending', date: '2025-10-28' },
          { id: 'LA002', name: 'Jane Smith', amount: 75000, status: 'approved', date: '2025-10-28' },
          { id: 'LA003', name: 'Bob Johnson', amount: 30000, status: 'under_review', date: '2025-10-27' },
          { id: 'LA004', name: 'Alice Brown', amount: 100000, status: 'approved', date: '2025-10-27' },
          { id: 'LA005', name: 'Charlie Wilson', amount: 45000, status: 'rejected', date: '2025-10-26' }
        ]);

        setLoading(false);
      }, 500);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setLoading(false);
    }
  };

  const getStatusBadge = (status) => {
    const statusClasses = {
      pending: 'badge-warning',
      approved: 'badge-success',
      rejected: 'badge-danger',
      under_review: 'badge-info',
      withdrawn: 'badge-warning'
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

  const handleApprove = async (applicationId) => {
    try {
      // In production, make API call: await axios.post(`/api/applications/${applicationId}/approve`);

      // Update local state
      setRecentApplications(prevApps =>
        prevApps.map(app =>
          app.id === applicationId ? { ...app, status: 'approved' } : app
        )
      );

      // Update stats only if status was pending
      const application = recentApplications.find(app => app.id === applicationId);
      if (application && application.status === 'pending') {
        setStats(prevStats => ({
          ...prevStats,
          pendingReview: prevStats.pendingReview - 1,
          approved: prevStats.approved + 1
        }));
      }

      // Update selected application if it's the current one
      if (selectedApplication && selectedApplication.id === applicationId) {
        setSelectedApplication(prevSelected => ({ ...prevSelected, status: 'approved' }));
      }

      // Close modal after action
      closeModal();
      alert('Application approved successfully!');
    } catch (error) {
      console.error('Error approving application:', error);
      alert('Failed to approve application. Please try again.');
    }
  };

  const handleReject = async (applicationId) => {
    try {
      // In production, make API call: await axios.post(`/api/applications/${applicationId}/reject`);

      // Update local state
      setRecentApplications(prevApps =>
        prevApps.map(app =>
          app.id === applicationId ? { ...app, status: 'rejected' } : app
        )
      );

      // Update stats only if status was pending
      const application = recentApplications.find(app => app.id === applicationId);
      if (application && application.status === 'pending') {
        setStats(prevStats => ({
          ...prevStats,
          pendingReview: prevStats.pendingReview - 1,
          rejected: prevStats.rejected + 1
        }));
      }

      // Update selected application if it's the current one
      if (selectedApplication && selectedApplication.id === applicationId) {
        setSelectedApplication(prevSelected => ({ ...prevSelected, status: 'rejected' }));
      }

      // Close modal after action
      closeModal();
      alert('Application rejected successfully!');
    } catch (error) {
      console.error('Error rejecting application:', error);
      alert('Failed to reject application. Please try again.');
    }
  };

  const handleWithdraw = async (applicationId) => {
    try {
      // In production, make API call: await axios.post(`/api/applications/${applicationId}/withdraw`);

      // Update local state
      setRecentApplications(prevApps =>
        prevApps.map(app =>
          app.id === applicationId ? { ...app, status: 'withdrawn' } : app
        )
      );

      // Update stats only if status was pending
      const application = recentApplications.find(app => app.id === applicationId);
      if (application && application.status === 'pending') {
        setStats(prevStats => ({
          ...prevStats,
          pendingReview: prevStats.pendingReview - 1
        }));
      }

      // Update selected application if it's the current one
      if (selectedApplication && selectedApplication.id === applicationId) {
        setSelectedApplication(prevSelected => ({ ...prevSelected, status: 'withdrawn' }));
      }

      // Close modal after action
      closeModal();
      alert('Application withdrawn successfully!');
    } catch (error) {
      console.error('Error withdrawing application:', error);
      alert('Failed to withdraw application. Please try again.');
    }
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
          <Link to="/dashboard" className="nav-item active">Dashboard</Link>
          <Link to="/chat" className="nav-item">Chat Assistant</Link>
          <Link to="/analytics" className="nav-item">Analytics</Link>
          {userRole === 'admin' && (
            <Link to="/upload" className="nav-item">Data Upload</Link>
          )}
        </div>
        <button onClick={onLogout} className="logout-btn">Logout</button>
      </div>

      <div className="main-content">
        <h1>Dashboard</h1>

        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
              📊
            </div>
            <div className="stat-details">
              <h3>Total Applications</h3>
              <p className="stat-value">{stats.totalApplications}</p>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)' }}>
              ⏳
            </div>
            <div className="stat-details">
              <h3>Pending Review</h3>
              <p className="stat-value">{stats.pendingReview}</p>
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
        </div>

        <div className="content-card">
          <h2>Recent Applications</h2>
          <div className="table-container">
            <table className="applications-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Applicant Name</th>
                  <th>Amount</th>
                  <th>Status</th>
                  <th>Date</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {recentApplications.map((app) => (
                  <tr key={app.id}>
                    <td>{app.id}</td>
                    <td>{app.name}</td>
                    <td>${app.amount.toLocaleString()}</td>
                    <td>{getStatusBadge(app.status)}</td>
                    <td>{app.date}</td>
                    <td>
                      <button className="action-btn view-btn" onClick={() => handleViewDetails(app)}>View</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
                  <span className="detail-label">Applicant Name:</span>
                  <span className="detail-value">{selectedApplication.name}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Amount:</span>
                  <span className="detail-value">${selectedApplication.amount.toLocaleString()}</span>
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
                  <span className="detail-label">Email:</span>
                  <span className="detail-value">{selectedApplication.email || 'N/A'}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Phone:</span>
                  <span className="detail-value">{selectedApplication.phone || 'N/A'}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Processing Time:</span>
                  <span className="detail-value">
                    {selectedApplication.status === 'approved' ? '2-3 business days' :
                     selectedApplication.status === 'rejected' ? 'N/A' :
                     '3-5 business days'}
                  </span>
                </div>
              </div>
              <div className="modal-footer">
                <button className="action-btn view-btn" onClick={closeModal}>Close</button>
                {selectedApplication.status === 'pending' && (
                  <>
                    <button
                      className="action-btn"
                      style={{background: '#10b981', marginLeft: '10px'}}
                      onClick={() => handleApprove(selectedApplication.id)}
                    >
                      Approve
                    </button>
                    <button
                      className="action-btn"
                      style={{background: '#ef4444', marginLeft: '10px'}}
                      onClick={() => handleReject(selectedApplication.id)}
                    >
                      Reject
                    </button>
                    <button
                      className="action-btn"
                      style={{background: '#f59e0b', marginLeft: '10px'}}
                      onClick={() => handleWithdraw(selectedApplication.id)}
                    >
                      Withdraw
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
