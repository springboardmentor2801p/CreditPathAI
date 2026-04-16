import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import About from "./pages/About";
import Predict from "./pages/Predict";
import Contact from "./pages/Contact";
import Dashboard from "./pages/Dashboard";
import Auth from "./pages/Auth";
import { AuthProvider } from "./context/AuthContext";
import Chatbot from "./components/Chatbot";
import "./index.css";

function App() {
  return (
    <AuthProvider>
      <Router>
        <Navbar />
        <div className="page-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/predict/user" element={<Predict type="user" />} />
            <Route path="/predict/bank" element={<Predict type="bank" />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/auth" element={<Auth />} />
            {/* Catch-all or default route could go here */}
          </Routes>
        </div>
        <Chatbot />
      </Router>
    </AuthProvider>
  );
}

export default App;