import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import About from "./pages/About";
import Predict from "./pages/Predict";
import Contact from "./pages/Contact";

function App() {
  return (
    <Router>
      <div style={{
        background: "#f0f9ff",
        minHeight: "100vh",
        fontFamily: "Segoe UI"
      }}>
        <Navbar />

        <div style={{ padding: "30px" }}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/predict/user" element={<Predict type="user" />} />
            <Route path="/predict/bank" element={<Predict type="bank" />} />
            <Route path="/contact" element={<Contact />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;