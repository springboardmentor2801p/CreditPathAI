import React, { useState, useRef, useEffect, useContext } from 'react';
import { AuthContext } from '../context/AuthContext';

const PRESET_ANSWERS = {
    // Project Specific - A to Z
    "what is creditpath": "CreditPath AI is an advanced financial tool that uses machine learning to predict loan approval chances and provides actionable financial insights for both borrowers and banks.",
    "how does creditpath work": "Borrowers provide their financial details to get a risk assessment. Banks use it to analyze borrower data, predict default probabilities, and get recovery recommendations.",
    "who uses creditpath": "It's designed for both retail borrowers (to check loan eligibility) and bank analysts (to assess borrower risk and minimize loan defaults).",
    
    // User / Borrower Questions
    "what is dti": "DTI (Debt-to-Income Ratio) compares your monthly debt payments to your gross monthly income. A lower DTI (usually under 36-40%) is preferable for loan approval.",
    "what is ltv": "LTV (Loan-to-Value) is the ratio of the loan amount to the value of the asset purchased. Banks prefer an LTV of 80% or lower.",
    "how to improve score": "To improve your credit score: Pay EMIs on time, keep credit card utilization below 30%, and avoid applying for multiple new loans in a short time.",
    "how to improve credit": "To improve your score: Pay EMIs on time, keep credit card utilization below 30%, and avoid applying for too many loans in a short time.",
    "is my data safe": "Yes! For this demonstration, your data is stored locally in your browser's storage and only processed securely.",
    "what is credit score": "A credit score is a 3-digit number (usually 300 to 900) representing your creditworthiness. A score above 750 is considered excellent and gets you better interest rates.",
    "what is emi": "EMI stands for Equated Monthly Installment. It's the fixed amount you pay to the bank every month to clear your loan.",
    "what is collateral": "Collateral is an asset (like a house or car) that you pledge to a lender to secure a loan. If you default, the lender can seize it.",
    "loan rate": "Loan interest rates depend heavily on your credit score and the current market. Generally, home loans range from 8-10% while personal loans are 10-15%+.",
    "default risk": "Default risk is the probability that a borrower will be unable to make their required payments on their debt obligations.",
    
    // Bank / Analyst Questions
    "how to assess risk": "To assess risk, enter the borrower's loan amount, income, credit score, LTV, and DTI. CreditPath's machine learning model will calculate the expected loss and default probability.",
    "what is expected loss": "Expected Loss is a financial metric used by banks. It is calculated as Probability of Default (PD) × Exposure at Default (EAD) × Loss Given Default (LGD).",
    "how does the bank engine work": "The bank engine uses an LGBM (Light Gradient Boosting Machine) model trained on historical data to predict if a borrower will default, and recommends the next best action.",
    "what is recovery channel": "If a loan is flagged as high risk, CreditPath recommends a recovery channel (e.g., Email, Phone Call, or Field Visit) to minimize potential losses.",
    "conditionally approved": "Conditionally Approved means the borrower has medium risk. The bank may require additional collateral, a co-signer, or a higher interest rate.",
    "high risk customer": "For high-risk customers, the system flags the expected loss. It is recommended to reject the application or require strict collateral and utilize high-priority recovery channels like Field Visits.",

    // Greetings & Pleasantries
    "hello": "Hello! I'm Pathy, your CreditPath AI assistant. You can ask me to 'predict risk' or learn about financial terms.",
    "hi": "Hi there! Ready to analyze some credit risk? Ask me to 'predict user risk' or 'predict bank risk'.",
    "hey": "Hey! How can I assist you with your financial questions?",
    "thank you": "You're very welcome! Feel free to ask if you need anything else.",
    "thanks": "Happy to help! Let me know if you have any other questions.",
    "bye": "Goodbye! Have a great day and stay financially healthy!",
    "good morning": "Good morning! Hope you're having a productive day. How can I help?",
    "good afternoon": "Good afternoon! What financial details can we look into today?",
    "good evening": "Good evening! Planning your finances for tomorrow?",
    "who are you": "I'm Pathy, your intelligent CreditPath assistant! Ask me to 'predict risk' or explain terms like DTI and LTV.",
    "help": "I can explain financial concepts and even predict risk! Try 'What is DTI?' or 'Predict bank risk: amount 200000, income 50000, score 720, ltv 80, dti 30'.",

    // Simple Conversations
    "yes": "Awesome! What would you like to know next?",
    "yeah": "Awesome! What would you like to know next?",
    "yep": "Great! Let me know if any financial jargon confuses you along the way.",
    "ok": "Got it! Ask me if you need explanations on things like EMI or Collateral.",
    "okay": "Got it! Ask me if you need explanations on things like EMI or Collateral.",
    "sure": "Perfect! Feel free to ask me questions while you fill out your loan application.",
    "no": "No problem! I'll be right here whenever you have questions about your credit health.",
};

function Chatbot() {
    const { user } = useContext(AuthContext);
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { text: "👋 Hi! I'm Pathy. Ask me anything about credit scores, banking terms, or ask me to literally 'predict risk'!", isBot: true }
    ]);
    const [input, setInput] = useState("");
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        if (isOpen) {
            scrollToBottom();
        }
    }, [messages, isOpen]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = input.trim();
        setMessages(prev => [...prev, { text: userMsg, isBot: false }]);
        setInput("");

        const lower = userMsg.toLowerCase();

        // 1. Check for prediction intent
        if (lower.includes('predict') && (lower.includes('risk') || lower.includes('loan') || lower.includes('bank') || lower.includes('user'))) {
            if (!user) {
                setMessages(prev => [...prev, { text: "🔒 Secure Feature: You need to log in to your account first to access risk predictions.", isBot: true }]);
                return;
            }
            if (lower.includes('bank')) {
                const amountMatch = lower.match(/(?:amount|loan)[:\s]*(\d+)/);
                const incomeMatch = lower.match(/income[:\s]*(\d+)/);
                const scoreMatch = lower.match(/score[:\s]*(\d+)/);
                const ltvMatch = lower.match(/(?:ltv|value)[:\s]*(\d+)/);
                const dtiMatch = lower.match(/(?:dti|dtir|ratio)[:\s]*(\d+)/);

                if (amountMatch && incomeMatch && scoreMatch && ltvMatch && dtiMatch) {
                    const bankData = {
                        loan_amount: parseFloat(amountMatch[1]),
                        income: parseFloat(incomeMatch[1]),
                        Credit_Score: parseFloat(scoreMatch[1]),
                        LTV: parseFloat(ltvMatch[1]),
                        dtir1: parseFloat(dtiMatch[1])
                    };
                    
                    setMessages(prev => [...prev, { text: "⏳ Calculating bank risk based on your inputs...", isBot: true }]);
                    
                    try {
                        const res = await fetch('http://localhost:8000/bank-risk', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(bankData)
                        });
                        const data = await res.json();
                        
                        if (data.error) throw new Error(data.error);
                        
                        const reply = `🏦 Bank Risk Prediction Result:\n\n• Status: ${data.loan_status}\n• Default Probability: ${(data.default_probability * 100).toFixed(1)}%\n• Expected Loss: ₹${data.expected_loss.toLocaleString('en-IN')}\n• Recommended Priority: ${data.bank_decision.priority}\n• Recovery Channel: ${data.bank_decision.recovery_channel}`;
                        setMessages(prev => [...prev, { text: reply, isBot: true }]);
                    } catch (err) {
                        setMessages(prev => [...prev, { text: "❌ Sorry, I couldn't process the risk prediction. Is the backend server running?", isBot: true }]);
                    }
                    return;
                } else {
                    const reply = "To predict Bank Risk, I need all of these values: amount, income, score, ltv, and dti.\n\nExample:\n'Predict bank risk: amount 200000, income 50000, score 720, ltv 80, dti 30'";
                    setTimeout(() => setMessages(prev => [...prev, { text: reply, isBot: true }]), 600);
                    return;
                }
            } else {
                // User Risk
                const typeMatch = lower.match(/(?:type|loan type)[:\s]*([a-z]+)/);
                const incomeMatch = lower.match(/income[:\s]*(\d+)/);
                const scoreMatch = lower.match(/score[:\s]*(\d+)/);
                const amountMatch = lower.match(/(?:amount|loan)[:\s]*(\d+)/);
                const missedMatch = lower.match(/(?:missed|payments)[:\s]*(\d+)/);

                if (incomeMatch && scoreMatch && amountMatch && missedMatch) {
                    const userData = {
                        loan_type: typeMatch ? typeMatch[1] : 'personal',
                        income: parseFloat(incomeMatch[1]),
                        credit_score: parseFloat(scoreMatch[1]),
                        loan_amount: parseFloat(amountMatch[1]),
                        missed_payments: parseInt(missedMatch[1])
                    };
                    
                    setMessages(prev => [...prev, { text: "⏳ Analyzing your profile for loan eligibility...", isBot: true }]);
                    
                    try {
                        const res = await fetch('http://localhost:8000/user-risk', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(userData)
                        });
                        const data = await res.json();
                        
                        const reply = `👤 User Risk Prediction Result:\n\n• Risk Level: ${data.risk_level}\n• Advice: ${data.recommendation_summary[0]}\n• Top Tip: ${data.tips[0]}`;
                        setMessages(prev => [...prev, { text: reply, isBot: true }]);
                    } catch (err) {
                        setMessages(prev => [...prev, { text: "❌ Sorry, I couldn't process the user risk prediction at the moment.", isBot: true }]);
                    }
                    return;
                } else {
                    const reply = "To predict User Risk, I need these values: type, income, score, amount, and missed.\n\nExample:\n'Predict risk: type home, income 80000, score 740, amount 300000, missed 0'";
                    setTimeout(() => setMessages(prev => [...prev, { text: reply, isBot: true }]), 600);
                    return;
                }
            }
        }

        // 2. Simple matching for FAQs
        setTimeout(() => {
            let reply = "I'm not quite sure about that. Try asking 'What is DTI?', 'How does CreditPath work?', or you can type 'Predict bank risk: amount 200000, income 50000, score 720, ltv 80, dti 30' to get real-time calculations.";

            for (let key in PRESET_ANSWERS) {
                if (lower.includes(key)) {
                    reply = PRESET_ANSWERS[key];
                    break;
                }
            }

            setMessages(prev => [...prev, { text: reply, isBot: true }]);
        }, 600);
    };

    return (
        <>
            {/* Floating Button */}
            <div
                onClick={() => setIsOpen(!isOpen)}
                style={{
                    position: 'fixed', bottom: 30, right: 30,
                    width: 60, height: 60, borderRadius: '50%',
                    background: 'var(--accent)', color: 'white',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '1.5rem', cursor: 'pointer', zIndex: 1000,
                    boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
                    transition: 'transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)'
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
                {isOpen ? '❌' : '💬'}
            </div>

            {/* Chat Window */}
            {isOpen && (
                <div style={{
                    position: 'fixed', bottom: 100, right: 30,
                    width: 380, height: 500, borderRadius: 16,
                    background: 'var(--bg-card)', border: '1px solid var(--border)',
                    display: 'flex', flexDirection: 'column',
                    zIndex: 1000, boxShadow: 'var(--shadow-lg)',
                    overflow: 'hidden', animation: 'slideUp 0.3s ease-out'
                }}>
                    {/* Header */}
                    <div style={{ padding: '16px 20px', background: 'var(--accent)', color: 'white' }}>
                        <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>Pathy Assistant AI</div>
                        <div style={{ fontSize: '0.75rem', opacity: 0.9 }}>Online • Ask me anything</div>
                    </div>

                    {/* Messages */}
                    <div style={{ flex: 1, padding: 16, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
                        {messages.map((msg, idx) => (
                            <div key={idx} style={{
                                maxWidth: '85%', padding: '12px 16px', borderRadius: 14,
                                fontSize: '0.9rem', lineHeight: 1.5,
                                alignSelf: msg.isBot ? 'flex-start' : 'flex-end',
                                background: msg.isBot ? 'var(--bg-secondary)' : 'var(--accent)',
                                color: msg.isBot ? 'var(--text-primary)' : 'white',
                                boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
                                whiteSpace: 'pre-wrap'
                            }}>
                                {msg.text}
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <form onSubmit={handleSend} style={{ padding: 16, borderTop: '1px solid var(--border)', display: 'flex', gap: 8 }}>
                        <input
                            value={input} onChange={(e) => setInput(e.target.value)}
                            placeholder="Type a message..."
                            style={{ flex: 1, border: '1px solid var(--border)', borderRadius: '20px', padding: '10px 16px', fontSize: '0.95rem', outline: 'none' }}
                        />
                        <button type="submit" style={{ background: 'var(--accent)', color: 'white', border: 'none', borderRadius: '50%', width: 40, height: 40, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem' }}>
                            ➜
                        </button>
                    </form>
                </div>
            )}

            {/* Animation Styles */}
            <style>{`
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
        </>
    );
}

export default Chatbot;
