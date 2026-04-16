import React, { createContext, useState, useEffect } from 'react';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);

    useEffect(() => {
        // Load auth
        const savedUser = localStorage.getItem('creditpath_user');
        if (savedUser) {
            setUser(JSON.parse(savedUser));
        }
    }, []);

    const signup = (name, email, password, role) => {
        const users = JSON.parse(localStorage.getItem('creditpath_users') || '[]');
        if (users.find(u => u.email === email)) {
            return { success: false, message: "Email already registered!" };
        }
        const newUser = { name, email, password, role, createdAt: new Date().toISOString() };
        users.push(newUser);
        localStorage.setItem('creditpath_users', JSON.stringify(users));
        return { success: true };
    };

    const login = (email, password, verifyOnly = false) => {
        // Direct Admin Bypass
        if (email === 'admin@creditpath.com' && password === 'admin123') {
            if (verifyOnly) return { success: true, status: 'valid_credentials' };
            const sessionUser = { name: 'System Admin', email, role: 'Admin', loggedInAt: new Date().toISOString() };
            setUser(sessionUser);
            localStorage.setItem('creditpath_user', JSON.stringify(sessionUser));
            return { success: true };
        }

        const users = JSON.parse(localStorage.getItem('creditpath_users') || '[]');
        const foundUser = users.find(u => u.email === email && u.password === password);

        if (foundUser) {
            if (verifyOnly) return { success: true, status: 'valid_credentials' };

            const sessionUser = { ...foundUser, loggedInAt: new Date().toISOString() };
            delete sessionUser.password; // Don't store password in session
            setUser(sessionUser);
            localStorage.setItem('creditpath_user', JSON.stringify(sessionUser));
            return { success: true };
        }
        return { success: false, message: "Invalid email or password" };
    };

    const logout = () => {
        setUser(null);
        localStorage.removeItem('creditpath_user');
    };

    return (
        <AuthContext.Provider value={{ user, login, signup, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

