import React, { useState } from "react";
import BorrowerForm from "./components/BorrowerForm";
import RiskMeter from "./components/RiskMeter";
import "./App.css";

function App() {

  const [result, setResult] = useState(null);

  return (
    <div className="dashboard">

      <h1>CreditPath AI Risk Dashboard</h1>

      <BorrowerForm setResult={setResult} />

      {result && (
        <div className="results">
          <RiskMeter result={result}/>
        </div>
      )}

    </div>
  );
}

export default App;