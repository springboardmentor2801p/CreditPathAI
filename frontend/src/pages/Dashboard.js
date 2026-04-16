import React, { useContext, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';

import { INBUILT_USERS, INBUILT_HISTORY, INBUILT_GOALS } from '../datadb/mockData';

function Dashboard() {
    const { user } = useContext(AuthContext);
    const [history, setHistory] = useState([]);
    const [allUsers, setAllUsers] = useState([]);
    const [goals, setGoals] = useState([]);
    const [newGoal, setNewGoal] = useState({ title: '', amount: '', saved: '' });
    const [expandedId, setExpandedId] = useState(null);
    const [allGoalsMap, setAllGoalsMap] = useState({});
    const navigate = useNavigate();

    useEffect(() => {
        if (!user) {
            navigate('/auth');
            return;
        }

        if (user.role === 'Admin') {
            // Load all users from localStorage and filter OUT admins
            const registered = JSON.parse(localStorage.getItem('creditpath_users') || '[]');
            const manualUsers = registered.filter(u => u.role !== 'Admin' && u.email !== user.email);
            setAllUsers(manualUsers); // Only manual users for the Real Signups table

            // Admin aggregates all user histories found in localStorage
            let dynamicHistory = [];
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith('creditpath_history_')) {
                    try {
                        const email = key.replace('creditpath_history_', '');
                        const data = JSON.parse(localStorage.getItem(key) || '[]');
                        // Attach unique ID and user email to each item for aggregation
                        const itemsWithUser = data.map((item, idx) => ({ ...item, userEmail: email, uniqueId: `${email}-${idx}-${item.timestamp}` }));
                        dynamicHistory = dynamicHistory.concat(itemsWithUser);
                    } catch (e) { console.error("Error parsing history for key", key); }
                }
            }

            // Combine Inbuilt + Dynamic and Sort
            const allHistory = [...INBUILT_HISTORY, ...dynamicHistory].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            setHistory(allHistory);

            // Aggregate all user goals
            let goalsMapping = { ...INBUILT_GOALS };
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith('creditpath_goals_')) {
                    try {
                        const email = key.replace('creditpath_goals_', '');
                        const data = JSON.parse(localStorage.getItem(key) || '[]');
                        goalsMapping[email] = data;
                    } catch (e) { console.error("Error parsing goals for key", key); }
                }
            }
            setAllGoalsMap(goalsMapping);
        } else {
            const saved = localStorage.getItem(`creditpath_history_${user.email}`);
            if (saved) {
                setHistory(JSON.parse(saved).reverse());
            } else {
                setHistory([]);
            }

            // Load goals
            const savedGoals = localStorage.getItem(`creditpath_goals_${user.email}`);
            if (savedGoals) setGoals(JSON.parse(savedGoals));
        }
    }, [user, navigate]);

    const addGoal = (e) => {
        e.preventDefault();
        if (!newGoal.title || !newGoal.amount) return;
        const updated = [...goals, { ...newGoal, id: Date.now() }];
        setGoals(updated);
        localStorage.setItem(`creditpath_goals_${user.email}`, JSON.stringify(updated));
        setNewGoal({ title: '', amount: '', saved: '' });
    };

    const deleteGoal = (id) => {
        const updated = goals.filter(g => g.id !== id);
        setGoals(updated);
        localStorage.setItem(`creditpath_goals_${user.email}`, JSON.stringify(updated));
    };

    const clearHistory = () => {
        if (window.confirm("Clear your prediction history?")) {
            localStorage.removeItem(`creditpath_history_${user.email}`);
            setHistory([]);
        }
    };

    const downloadCSV = (data, filename) => {
        if (!data || data.length === 0) return;
        const headers = Object.keys(data[0]).join(',');
        const rows = data.map(obj => Object.values(obj).map(val => `"${val}"`).join(','));
        const csvContent = [headers, ...rows].join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    if (!user) return null; // Component will redirect in useEffect

    // If Admin, render aggregate view
    if (user.role === 'Admin') {
        return (
            <div className="predict-page" style={{ maxWidth: 1000 }}>
                <div className="predict-header">
                    <h1 style={{ fontFamily: "'Inter', sans-serif", fontSize: "2rem" }}>
                        🛠️ Admin Control Center
                    </h1>
                    <p>Welcome back, Admin {user.name}. View aggregate system analytics below.</p>
                </div>

                <div className="predict-card">
                    <div className="metric-cards">
                        <div className="metric-card">
                            <div className="metric-label">Total Users</div>
                            <div className="metric-value">{allUsers.length + INBUILT_USERS.length}</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">Total Predictions</div>
                            <div className="metric-value">{history.length}</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">High Risk Flagged</div>
                            <div className="metric-value" style={{ color: 'var(--danger)' }}>{history.filter(h => h.risk === 'High Risk' || h.risk === 'Rejected' || h.risk === 'High').length}</div>
                        </div>
                    </div>

                    <div style={{ marginTop: 30 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                            <h3 style={{ borderLeft: '4px solid var(--accent)', paddingLeft: 12, margin: 0 }}>👥 Real Manual Signups</h3>
                            <button
                                onClick={() => downloadCSV(allUsers, 'creditpath_real_users.csv')}
                                style={{ padding: '6px 12px', fontSize: '0.8rem', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--bg-card)', cursor: 'pointer' }}
                            >
                                📥 Export CSV
                            </button>
                        </div>
                        {allUsers.length === 0 ? <p style={{ color: 'var(--text-muted)' }}>No manual signups yet.</p> : (
                            <div style={{ overflowX: 'auto' }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 30, fontSize: '0.9rem' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '2px solid var(--border)', color: 'var(--text-muted)' }}>
                                            <th style={{ padding: '12px 8px', textAlign: 'left' }}>Name</th>
                                            <th style={{ padding: '12px 8px', textAlign: 'left' }}>Email</th>
                                            <th style={{ padding: '12px 8px', textAlign: 'left' }}>Role</th>
                                            <th style={{ padding: '12px 8px', textAlign: 'right' }}>Joined</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {allUsers.map((u, i) => (
                                            <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                                                <td style={{ padding: '16px 8px', fontWeight: 600 }}>{u.name}</td>
                                                <td style={{ padding: '16px 8px' }}>{u.email}</td>
                                                <td style={{ padding: '16px 8px' }}>
                                                    <span style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: 4, background: 'rgba(16, 185, 129, 0.1)', color: 'var(--success)' }}>
                                                        {u.role}
                                                    </span>
                                                </td>
                                                <td style={{ padding: '16px 8px', textAlign: 'right', color: 'var(--text-muted)' }}>
                                                    {new Date(u.createdAt).toLocaleDateString()}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>



                    <div style={{ marginTop: 30 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                            <h3 style={{ borderLeft: '4px solid var(--accent-gold)', paddingLeft: 12, margin: 0 }}>📊 All System History</h3>
                            <button
                                onClick={() => downloadCSV(history, 'creditpath_system_history.csv')}
                                style={{ padding: '6px 12px', fontSize: '0.8rem', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--bg-card)', cursor: 'pointer' }}
                            >
                                📥 Export CSV
                            </button>
                        </div>
                        {history.length === 0 ? <p>No data recorded.</p> : (
                            <div style={{ overflowX: 'auto' }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 12, fontSize: '0.85rem' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '2px solid var(--border)', color: 'var(--text-muted)' }}>
                                            <th style={{ padding: '12px 8px', textAlign: 'left' }}>Date</th>
                                            <th style={{ padding: '12px 8px', textAlign: 'left' }}>User</th>
                                            <th style={{ padding: '12px 8px', textAlign: 'left' }}>User Goal</th>
                                            <th style={{ padding: '12px 8px', textAlign: 'left' }}>Loan Category</th>
                                            <th style={{ padding: '12px 8px', textAlign: 'left' }}>Type</th>
                                            <th style={{ padding: '12px 8px', textAlign: 'right' }}>Amount</th>
                                            <th style={{ padding: '12px 8px', textAlign: 'center' }}>Decision / Risk</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {history.map((item, index) => {
                                            const userGoal = allGoalsMap[item.userEmail]?.[0]?.title || 'N/A';
                                            const uInfo = [...INBUILT_USERS, ...JSON.parse(localStorage.getItem('creditpath_users') || '[]')].find(u => u.email === item.userEmail);
                                            return (
                                                <tr key={item.uniqueId || index} style={{ borderBottom: '1px solid var(--border)', background: index % 2 === 0 ? 'transparent' : 'rgba(0,0,0,0.02)' }}>
                                                    <td style={{ padding: '16px 8px' }}>{new Date(item.timestamp).toLocaleDateString()}</td>
                                                    <td style={{ padding: '16px 8px' }}>
                                                        <div style={{ fontWeight: 600 }}>{uInfo?.name || 'Inbuilt User'}</div>
                                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{item.userEmail}</div>
                                                    </td>
                                                    <td style={{ padding: '16px 8px', color: 'var(--accent)', fontWeight: 500 }}>
                                                        {userGoal}
                                                    </td>
                                                    <td style={{ padding: '16px 8px', textTransform: 'capitalize' }}>
                                                        {item.loan_type?.replace('_', ' ') || 'General'}
                                                    </td>
                                                    <td style={{ padding: '16px 8px', fontWeight: 600 }}>
                                                        {item.type === 'user' ? '👤 User' : '🏦 Bank'}
                                                    </td>
                                                    <td style={{ padding: '16px 8px', textAlign: 'right' }}>₹{Number(item.amount).toLocaleString('en-IN')}</td>
                                                    <td style={{ padding: '16px 8px', textAlign: 'center' }}>
                                                        <span className={`risk-badge ${item.risk?.toLowerCase() === 'low' || item.risk === 'Approved' ? 'low' :
                                                            item.risk?.toLowerCase() === 'medium' || item.risk === 'Conditionally Approved' ? 'medium' : 'high'
                                                            }`}>
                                                            {item.risk}
                                                        </span>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="predict-page" style={{ maxWidth: 900 }}>
            <div className="predict-header" style={{ textAlign: 'left' }}>
                <h1 style={{ fontFamily: "'Inter', sans-serif", fontSize: "2rem" }}>
                    👋 Welcome back, {user.name.split(' ')[0]}!
                </h1>
                <p>Member Since: {new Date(user.createdAt || user.loggedInAt).toLocaleDateString()}</p>
            </div>

            {/* 🎯 Financial Goals Section */}
            <div className="predict-card" style={{ marginBottom: 30 }}>
                <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 20 }}>
                    <div className="title-icon" style={{ background: 'rgba(99, 102, 241, 0.15)' }}>🎯</div>
                    <h2 style={{ fontFamily: "'Inter', sans-serif", margin: 0 }}>Personal Financial Goals</h2>
                </div>

                <form onSubmit={addGoal} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px', marginBottom: 24 }}>
                    <input className="form-input" placeholder="Goal (e.g. Home Downpayment)" value={newGoal.title} onChange={e => setNewGoal({ ...newGoal, title: e.target.value })} />
                    <input className="form-input" type="number" placeholder="Target Amount (₹)" value={newGoal.amount} onChange={e => setNewGoal({ ...newGoal, amount: e.target.value })} />
                    <input className="form-input" type="number" placeholder="Already Saved (₹)" value={newGoal.saved} onChange={e => setNewGoal({ ...newGoal, saved: e.target.value })} />
                    <button type="submit" className="btn-primary" style={{ width: 'auto' }}>Add Goal</button>
                </form>

                {goals.length === 0 ? <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>You haven't set any financial goals yet. What are you saving for?</p> : (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
                        {goals.map(goal => {
                            const progress = Math.min(100, (Number(goal.saved || 0) / Number(goal.amount)) * 100);
                            return (
                                <div key={goal.id} style={{ padding: 16, background: 'var(--bg-secondary)', borderRadius: 12, border: '1px solid var(--border)' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                                        <span style={{ fontWeight: 600 }}>{goal.title}</span>
                                        <button onClick={() => deleteGoal(goal.id)} style={{ background: 'none', border: 'none', color: 'var(--danger)', cursor: 'pointer', fontSize: '0.8rem' }}>Delete</button>
                                    </div>
                                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                                        ₹{Number(goal.saved || 0).toLocaleString()} of ₹{Number(goal.amount).toLocaleString()}
                                    </div>
                                    <div style={{ height: 8, background: 'var(--border)', borderRadius: 4, overflow: 'hidden' }}>
                                        <div style={{ width: `${progress}%`, height: '100%', background: 'var(--accent)', transition: 'width 0.5s ease-out' }} />
                                    </div>
                                    <div style={{ textAlign: 'right', fontSize: '0.75rem', marginTop: 4, color: 'var(--accent)', fontWeight: 600 }}>{progress.toFixed(0)}%</div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            <div className="predict-card">
                <div className="predict-card-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                        <div className="title-icon" style={{ background: 'rgba(16, 185, 129, 0.15)' }}>📜</div>
                        <h2 style={{ fontFamily: "'Inter', sans-serif" }}>Prediction History</h2>
                    </div>
                    {history.length > 0 && (
                        <button onClick={clearHistory} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', textDecoration: 'underline', fontSize: '0.85rem' }}>
                            Clear History
                        </button>
                    )}
                </div>

                {history.length === 0 ? (
                    <div className="info-box" style={{ textAlign: "center", padding: "40px 20px" }}>
                        <div style={{ fontSize: 40, marginBottom: 12 }}>📭</div>
                        <h4>No history found</h4>
                        <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>Run your first prediction on the User or Bank tab to see it here!</p>
                    </div>
                ) : (
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 12, fontSize: '0.9rem' }}>
                            <thead>
                                <tr style={{ borderBottom: '2px solid var(--border)', color: 'var(--text-muted)' }}>
                                    <th style={{ padding: '12px 8px', textAlign: 'left' }}>Date</th>
                                    <th style={{ padding: '12px 8px', textAlign: 'left' }}>Type</th>
                                    <th style={{ padding: '12px 8px', textAlign: 'right' }}>Amount</th>
                                    <th style={{ padding: '12px 8px', textAlign: 'center' }}>Status</th>
                                    <th style={{ padding: '12px 8px', textAlign: 'right' }}>Action</th>
                                </tr>
                            </thead>
                            <tbody>
                                {history.map((item, index) => (
                                    <React.Fragment key={index}>
                                        <tr style={{ borderBottom: '1px solid var(--border)', transition: 'background 0.2s' }} className="history-row">
                                            <td style={{ padding: '16px 8px', color: 'var(--text-secondary)' }}>
                                                {new Date(item.timestamp).toLocaleDateString()}
                                            </td>
                                            <td style={{ padding: '16px 8px', fontWeight: 600 }}>
                                                {item.type === 'user' ? '👤 User' : '🏦 Bank'}
                                            </td>
                                            <td style={{ padding: '16px 8px', textAlign: 'right', fontWeight: 600 }}>
                                                ₹{Number(item.amount).toLocaleString('en-IN')}
                                            </td>
                                            <td style={{ padding: '16px 8px', textAlign: 'center' }}>
                                                <span className={`risk-badge ${item.risk === 'Low' || item.risk === 'Approved' ? 'low' :
                                                    item.risk === 'Medium' || item.risk === 'Conditionally Approved' ? 'medium' : 'high'
                                                    }`}>
                                                    {item.risk}
                                                </span>
                                            </td>
                                            <td style={{ padding: '16px 8px', textAlign: 'right' }}>
                                                <button
                                                    onClick={() => setExpandedId(expandedId === index ? null : index)}
                                                    style={{ background: 'transparent', border: 'none', color: 'var(--accent)', cursor: 'pointer', fontWeight: 600 }}
                                                >
                                                    {expandedId === index ? 'Close ▲' : 'Details ▼'}
                                                </button>
                                            </td>
                                        </tr>
                                        {expandedId === index && (
                                            <tr style={{ background: 'var(--bg-secondary)' }}>
                                                <td colSpan="5" style={{ padding: '16px', borderBottom: '1px solid var(--border)' }}>
                                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px' }}>
                                                        <div>
                                                            <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: 4 }}>Time</div>
                                                            <div style={{ fontWeight: 600 }}>{new Date(item.timestamp).toLocaleTimeString()}</div>
                                                        </div>
                                                        <div>
                                                            <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: 4 }}>Calculation ID</div>
                                                            <div style={{ fontWeight: 600 }}>#{item.timestamp.split('T')[0].replace(/-/g, '')}-{index}</div>
                                                        </div>
                                                        <div>
                                                            <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: 4 }}>Summary</div>
                                                            <div style={{ color: 'var(--text-secondary)' }}>
                                                                Prediction completed successfully with {item.risk} risk profile for a ₹{Number(item.amount).toLocaleString('en-IN')} application.
                                                            </div>
                                                        </div>
                                                    </div>
                                                </td>
                                            </tr>
                                        )}
                                    </React.Fragment>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}

export default Dashboard;

