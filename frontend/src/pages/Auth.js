import React, { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';

function Auth() {
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState({ name: '', email: '', password: '', role: 'Borrower' });
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const { login, signup } = useContext(AuthContext);
    const navigate = useNavigate();

    const [isMfaStep, setIsMfaStep] = useState(false);
    const [mfaInput, setMfaInput] = useState('');

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
        setError('');
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (isLogin) {
            const res = login(formData.email, formData.password, true);
            if (res.status === 'valid_credentials') {
                setIsMfaStep(true);
                setSuccess("Demo MFA: Enter code '1234' to login.");
                setError('');
            } else {
                setError(res.message);
            }
        } else {
            if (!formData.name || !formData.email || !formData.password) {
                return setError("Please fill in all fields");
            }
            const res = signup(formData.name, formData.email, formData.password, formData.role);
            if (res.success) {
                setSuccess("Account created! Please log in.");
                setIsLogin(true);
                setFormData({ ...formData, password: '' });
            } else {
                setError(res.message);
            }
        }
    };

    const handleMfaSubmit = (e) => {
        e.preventDefault();
        if (mfaInput === '1234') {
            const res = login(formData.email, formData.password);
            if (res.success) {
                navigate('/dashboard');
            } else {
                setError("Login failed. Please try again.");
                setIsMfaStep(false);
            }
        } else {
            setError("Invalid MFA code. For this demo, use '1234'.");
        }
    };

    return (
        <div className="predict-page" style={{ maxWidth: 450, marginTop: 60 }}>
            <div className="predict-card">
                <div className="predict-card-title" style={{ justifyContent: 'center', marginBottom: 24 }}>
                    <div className="title-icon">{isLogin ? '🔐' : '✍️'}</div>
                    <h2 style={{ fontFamily: "'Inter', sans-serif" }}>{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
                </div>

                {error && <div className="error-box" style={{ marginBottom: 20 }}>{error}</div>}
                {success && <div className="info-box success-box" style={{ marginBottom: 20 }}>{success}</div>}

                {isMfaStep ? (
                    <form onSubmit={handleMfaSubmit} className="form-group" style={{ gap: 20 }}>
                        <div style={{ textAlign: 'center' }}>
                            <label>Enter 4-Digit Security PIN</label>
                            <input
                                className="form-input"
                                placeholder="0 0 0 0"
                                value={mfaInput}
                                onChange={(e) => setMfaInput(e.target.value)}
                                required
                                maxLength="4"
                                style={{ textAlign: 'center', fontSize: '1.8rem', letterSpacing: '12px', marginTop: 12, fontWeight: 800, border: '2px solid var(--accent)' }}
                                autoFocus
                            />
                            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 12 }}>
                                We've simulated a secure code sent to your device.
                            </p>
                        </div>
                        <button type="submit" className="btn-primary">Verify & Login</button>
                        <button type="button" onClick={() => setIsMfaStep(false)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.85rem' }}>
                            Back to Login
                        </button>
                    </form>
                ) : (
                    <form onSubmit={handleSubmit} className="form-group" style={{ gap: 20 }}>
                        {!isLogin && (
                            <div>
                                <label>Full Name</label>
                                <input
                                    className="form-input" name="name" placeholder="John Doe"
                                    value={formData.name} onChange={handleChange} required
                                />
                            </div>
                        )}
                        <div>
                            <label>Email Address</label>
                            <input
                                className="form-input" type="email" name="email" placeholder="john@example.com"
                                value={formData.email} onChange={handleChange} required
                            />
                        </div>
                        <div>
                            <label>Password</label>
                            <input
                                className="form-input" type="password" name="password" placeholder="••••••••"
                                value={formData.password} onChange={handleChange} required
                            />
                        </div>

                        {!isLogin && (
                            <div>
                                <label>Account Type</label>
                                <select className="form-input" name="role" value={formData.role} onChange={handleChange}>
                                    <option value="Borrower">👤 Borrower</option>
                                    <option value="Bank Agent">🏦 Bank Agent</option>
                                </select>
                            </div>
                        )}

                        <button type="submit" className="btn-primary" style={{ marginTop: 10 }}>
                            {isLogin ? 'Log In' : 'Sign Up'}
                        </button>

                        {isLogin && (
                            <button
                                type="button"
                                onClick={() => {
                                    setFormData({ ...formData, email: 'admin@creditpath.com', password: 'admin123' });
                                    const res = login('admin@creditpath.com', 'admin123', true);
                                    if (res.status === 'valid_credentials') {
                                        setIsMfaStep(true);
                                        setSuccess("Admin Bypass Enabled: Enter demo MFA code '1234'.");
                                        setError('');
                                    }
                                }}
                                style={{
                                    marginTop: '8px',
                                    padding: '12px',
                                    background: 'var(--bg-secondary)',
                                    color: 'var(--text-primary)',
                                    border: '1px solid var(--border)',
                                    borderRadius: '8px',
                                    cursor: 'pointer',
                                    fontWeight: '600',
                                    width: '100%'
                                }}
                            >
                                🔑 Direct Admin Login
                            </button>
                        )}
                    </form>
                )}

                <div style={{ textAlign: 'center', marginTop: 24, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    {isLogin ? "Don't have an account?" : "Already have an account?"}
                    <button
                        onClick={() => setIsLogin(!isLogin)}
                        style={{
                            background: 'none', border: 'none', color: 'var(--accent)',
                            fontWeight: 600, cursor: 'pointer', marginLeft: 6, textDecoration: 'underline'
                        }}
                    >
                        {isLogin ? 'Register now' : 'Log in here'}
                    </button>
                </div>
            </div>
        </div>
    );
}

export default Auth;
