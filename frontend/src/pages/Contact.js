import React from "react";

function Contact() {
  return (
    <div style={{
      maxWidth: "600px",
      margin: "auto",
      padding: "40px",
      background: "#fef9c3",
      borderRadius: "15px",
      boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
    }}>

      <h1 style={{ color: "#92400e" }}>Contact Us</h1>

      <p style={{ marginTop: "15px" }}>
        Have questions or feedback? Reach out to us!
      </p>

      <div style={{ marginTop: "20px", lineHeight: "2" }}>
        <p><b>Email:</b> support@creditpath.ai</p>
        <p><b>Phone:</b> +91 9876543210</p>
        <p><b>Location:</b> India</p>
      </div>

      <h3 style={{ marginTop: "25px" }}>📩 Send Message</h3>

      <input placeholder="Your Name" style={inputStyle} /><br />
      <input placeholder="Your Email" style={inputStyle} /><br />
      <textarea placeholder="Your Message" style={inputStyle} /><br />

      <button style={buttonStyle}>
        Send Message
      </button>

    </div>
  );
}

const inputStyle = {
  width: "100%",
  padding: "10px",
  marginTop: "10px",
  borderRadius: "8px",
  border: "1px solid #ccc"
};

const buttonStyle = {
  marginTop: "15px",
  padding: "10px 20px",
  background: "#ca8a04",
  color: "white",
  border: "none",
  borderRadius: "8px",
  cursor: "pointer"
};

export default Contact;