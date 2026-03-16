import React, { useState } from "react";
import { getRiskScore } from "../services/api";

function BorrowerForm({ setResult }) {

  const [formData, setFormData] = useState({
    purpose: "debtconsolidation",
    isJointApplication: 0,
    loanAmount: "",
    term: "36 months",
    interestRate: "",
    monthlyPayment: "",
    grade: "A1",
    residentialState: "CA",
    yearsEmployment: "1 year",
    homeOwnership: "rent",
    annualIncome: "",
    incomeVerified: 0,
    dtiRatio: "",
    lengthCreditHistory: "",
    numTotalCreditLines: "",
    numOpenCreditLines: "",
    numOpenCreditLines1Year: "",
    revolvingBalance: "",
    revolvingUtilizationRate: "",
    numDerogatoryRec: "",
    numDelinquency2Years: "",
    numChargeoff1year: "",
    numInquiries6Mon: ""
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const response = await getRiskScore(formData);
    setResult(response.data);
  };

  return (
    <form onSubmit={handleSubmit}>

      {/* Borrower Information */}
      <div className="form-section">
        <h2>Borrower Information</h2>

        <div className="grid">

          <div className="field">
            <label>Purpose</label>
            <select name="purpose" onChange={handleChange}>
              <option value="debtconsolidation">Debt Consolidation</option>
              <option value="creditcard">Credit Card</option>
              <option value="homeimprovement">Home Improvement</option>
            </select>
          </div>

          <div className="field">
            <label>Joint Application</label>
            <select name="isJointApplication" onChange={handleChange}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>

          <div className="field">
            <label>State</label>
            <input name="residentialState" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Years Employment</label>
            <select name="yearsEmployment" onChange={handleChange}>
              <option value="1 year">1 year</option>
              <option value="5 years">5 years</option>
              <option value="10+ years">10+ years</option>
            </select>
          </div>

          <div className="field">
            <label>Home Ownership</label>
            <select name="homeOwnership" onChange={handleChange}>
              <option value="rent">Rent</option>
              <option value="mortgage">Mortgage</option>
              <option value="own">Own</option>
            </select>
          </div>

          <div className="field">
            <label>Annual Income</label>
            <input type="number" name="annualIncome" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Income Verified</label>
            <select name="incomeVerified" onChange={handleChange}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>

        </div>
      </div>

      {/* Loan Information */}
      <div className="form-section">
        <h2>Loan Information</h2>

        <div className="grid">

          <div className="field">
            <label>Loan Amount</label>
            <input type="number" name="loanAmount" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Term</label>
            <select name="term" onChange={handleChange}>
              <option value="36 months">36 months</option>
              <option value="60 months">60 months</option>
            </select>
          </div>

          <div className="field">
            <label>Interest Rate</label>
            <input type="number" name="interestRate" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Monthly Payment</label>
            <input type="number" name="monthlyPayment" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Grade</label>
            <select name="grade" onChange={handleChange}>
              <option value="A1">A1</option>
              <option value="B2">B2</option>
              <option value="C3">C3</option>
              <option value="E3">E3</option>
            </select>
          </div>

        </div>
      </div>

      {/* Credit Behaviour */}
      <div className="form-section">
        <h2>Credit Behaviour</h2>

        <div className="grid">

          <div className="field">
            <label>DTI Ratio</label>
            <input type="number" name="dtiRatio" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Credit History Length</label>
            <input type="number" name="lengthCreditHistory" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Total Credit Lines</label>
            <input type="number" name="numTotalCreditLines" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Open Credit Lines</label>
            <input type="number" name="numOpenCreditLines" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Open Credit Lines (1yr)</label>
            <input type="number" name="numOpenCreditLines1Year" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Revolving Balance</label>
            <input type="number" name="revolvingBalance" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Revolving Utilization</label>
            <input type="number" name="revolvingUtilizationRate" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Derogatory Records</label>
            <input type="number" name="numDerogatoryRec" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Delinquencies (2yr)</label>
            <input type="number" name="numDelinquency2Years" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Chargeoffs (1yr)</label>
            <input type="number" name="numChargeoff1year" onChange={handleChange}/>
          </div>

          <div className="field">
            <label>Inquiries (6 months)</label>
            <input type="number" name="numInquiries6Mon" onChange={handleChange}/>
          </div>

        </div>
      </div>

      <button type="submit">Calculate Risk</button>

    </form>
  );
}

export default BorrowerForm;