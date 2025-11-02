import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import Login from './components/auth/Login';
import Dashboard from './components/dashboard/Dashboard';
import UserDashboard from './components/dashboard/UserDashboard';
import ChatInterface from './components/chat/ChatInterface';
import DataUpload from './components/admin/DataUpload';
import Analytics from './components/dashboard/Analytics';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userRole, setUserRole] = useState(null);

  const handleLogin = (role) => {
    setIsAuthenticated(true);
    setUserRole(role);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUserRole(null);
  };

  return (
    <Router>
      <div className="App">
        <Routes>
          <Route
            path="/login"
            element={
              isAuthenticated ?
              <Navigate to="/dashboard" /> :
              <Login onLogin={handleLogin} />
            }
          />
          <Route
            path="/dashboard"
            element={
              isAuthenticated ?
              (userRole === 'admin' ?
                <Dashboard userRole={userRole} onLogout={handleLogout} /> :
                <UserDashboard userRole={userRole} userId="user123" onLogout={handleLogout} />
              ) :
              <Navigate to="/login" />
            }
          />
          <Route
            path="/chat"
            element={
              isAuthenticated ?
              <ChatInterface userRole={userRole} onLogout={handleLogout} /> :
              <Navigate to="/login" />
            }
          />
          <Route
            path="/upload"
            element={
              isAuthenticated && userRole === 'admin' ?
              <DataUpload onLogout={handleLogout} /> :
              <Navigate to="/login" />
            }
          />
          <Route
            path="/analytics"
            element={
              isAuthenticated ?
              <Analytics userRole={userRole} onLogout={handleLogout} /> :
              <Navigate to="/login" />
            }
          />
          <Route path="/" element={<Navigate to="/login" />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
