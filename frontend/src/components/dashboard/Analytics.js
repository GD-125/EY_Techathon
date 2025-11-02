import { Link } from 'react-router-dom';
import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './Analytics.css';

const Analytics = ({ userRole, onLogout }) => {
  const [analyticsData, setAnalyticsData] = useState({
    monthlyTrends: [],
    statusDistribution: [],
    agentPerformance: []
  });

  useEffect(() => {
    // Mock analytics data - different for admin vs user
    if (userRole === 'admin') {
      // Admin sees all system data
      setAnalyticsData({
        monthlyTrends: [
          { month: 'Jan', applications: 120, approved: 85, rejected: 35 },
          { month: 'Feb', applications: 145, approved: 102, rejected: 43 },
          { month: 'Mar', applications: 160, approved: 115, rejected: 45 },
          { month: 'Apr', applications: 178, approved: 130, rejected: 48 },
          { month: 'May', applications: 195, approved: 145, rejected: 50 },
          { month: 'Jun', applications: 210, approved: 158, rejected: 52 }
        ],
        statusDistribution: [
          { name: 'Approved', value: 892, color: '#10b981' },
          { name: 'Rejected', value: 312, color: '#ef4444' },
          { name: 'Pending', value: 43, color: '#f59e0b' }
        ],
        agentPerformance: [
          { agent: 'Sales', tasks: 450, success: 425 },
          { agent: 'Verification', tasks: 380, success: 360 },
          { agent: 'Underwriting', tasks: 320, success: 295 },
          { agent: 'Sanction', tasks: 280, success: 270 }
        ]
      });
    } else {
      // Regular user sees only their data
      setAnalyticsData({
        monthlyTrends: [
          { month: 'Jan', applications: 0, approved: 0, rejected: 0 },
          { month: 'Feb', applications: 0, approved: 0, rejected: 0 },
          { month: 'Mar', applications: 1, approved: 0, rejected: 1 },
          { month: 'Apr', applications: 0, approved: 0, rejected: 0 },
          { month: 'May', applications: 1, approved: 1, rejected: 0 },
          { month: 'Jun', applications: 1, approved: 0, rejected: 0 }
        ],
        statusDistribution: [
          { name: 'Approved', value: 1, color: '#10b981' },
          { name: 'Rejected', value: 1, color: '#ef4444' },
          { name: 'Under Review', value: 1, color: '#f59e0b' }
        ],
        agentPerformance: []  // Users don't see agent performance
      });
    }
  }, [userRole]);

  return (
    <div className="page-container">
      <div className="sidebar">
        <h2>Loan ERP</h2>
        <div className="nav-menu">
          <Link to="/dashboard" className="nav-item">Dashboard</Link>
          <Link to="/chat" className="nav-item">Chat Assistant</Link>
          <Link to="/analytics" className="nav-item active">Analytics</Link>
          {userRole === 'admin' && (
            <Link to="/upload" className="nav-item">Data Upload</Link>
          )}
        </div>
        <button onClick={onLogout} className="logout-btn">Logout</button>
      </div>

      <div className="main-content">
        <h1>{userRole === 'admin' ? 'System Analytics Dashboard' : 'My Application Analytics'}</h1>

        <div className="analytics-grid">
          <div className="content-card chart-card">
            <h2>Monthly Application Trends</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={analyticsData.monthlyTrends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="applications" stroke="#667eea" strokeWidth={2} />
                <Line type="monotone" dataKey="approved" stroke="#10b981" strokeWidth={2} />
                <Line type="monotone" dataKey="rejected" stroke="#ef4444" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="content-card chart-card">
            <h2>Application Status Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={analyticsData.statusDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {analyticsData.statusDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {userRole === 'admin' && analyticsData.agentPerformance.length > 0 && (
            <div className="content-card chart-card full-width">
              <h2>Agent Performance</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analyticsData.agentPerformance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="agent" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="tasks" fill="#667eea" />
                  <Bar dataKey="success" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="content-card">
          <h2>Key Insights</h2>
          <div className="insights-grid">
            <div className="insight-card">
              <h3>📈 Approval Rate</h3>
              <p className="insight-value">71.5%</p>
              <p className="insight-desc">Average approval rate across all applications</p>
            </div>
            <div className="insight-card">
              <h3>⏱️ Processing Time</h3>
              <p className="insight-value">2.3 days</p>
              <p className="insight-desc">Average time from application to decision</p>
            </div>
            <div className="insight-card">
              <h3>💰 Avg Loan Amount</h3>
              <p className="insight-value">$58,450</p>
              <p className="insight-desc">Average approved loan amount</p>
            </div>
            <div className="insight-card">
              <h3>🎯 Success Rate</h3>
              <p className="insight-value">92.1%</p>
              <p className="insight-desc">Agent task completion success rate</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
