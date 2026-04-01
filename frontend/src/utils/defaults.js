export const BLANK_FORM = {
  loanAmount: '',
  interestRate: '',
  monthlyPayment: '',
  term_months: '',
  yearsEmployment: '',
  annualIncome: '',

  isJointApplication: '',
  incomeVerified: '',
  dtiRatio: '',
  revolvingBalance: '',
  revolvingUtilizationRate: '',
  lengthCreditHistory: '',
  numTotalCreditLines: '',
  numOpenCreditLines: '',
  numOpenCreditLines1Year: '',
  numDerogatoryRec: '',
  numDelinquency2Years: '',
  numChargeoff1year: '',
  numInquiries6Mon: '',
  grade_score: '',
  
  loan_to_income_ratio: '',
  payment_to_income_ratio: '',
  repayment_velocity: '',
  loan_amortization_rate: '',
  open_credit_ratio: '',
  recent_credit_velocity: '',
  inquiry_intensity: '',
  delinquency_density: '',
  derogatory_density: '',
  estimated_credit_limit: '',
  credit_utilization_recomputed: '',
  log_loanAmount: '',
  log_annualIncome: '',
  log_revolvingBalance: '',
  
  purpose_business: '',
  purpose_debtconsolidation: '',
  purpose_education: '',
  purpose_healthcare: '',
  purpose_homeimprovement: '',
  purpose_other: '',
  homeOwnership_own: '',
  homeOwnership_rent: '',
  threshold: '',
};

export function computePayload(form) {
  const get = (key, fallback) => {
    if (form[key] === '' || form[key] === undefined || form[key] === null) {
      if (typeof fallback === 'number') {
        // Return 2 decimal places for neatness if necessary, but keep it numeric
        return Number.isInteger(fallback) ? fallback : Number(fallback.toFixed(3));
      }
      return fallback;
    }
    return Number(form[key]);
  };

  const loanAmount = get('loanAmount', 350000); // sensible generic default for display
  const annualIncome = Math.max(1, get('annualIncome', 500000));
  const monthlyPayment = get('monthlyPayment', loanAmount * 0.03); 
  const term = get('term_months', 36);
  const emp = get('yearsEmployment', 5);
  const revBal = get('revolvingBalance', annualIncome * 0.1);

  const numTot = get('numTotalCreditLines', 6);
  const numOpn = get('numOpenCreditLines', 3);

  const payload = {
    loanAmount: loanAmount,
    interestRate: get('interestRate', 12.0),
    monthlyPayment: monthlyPayment,
    term_months: term,
    annualIncome: annualIncome,
    yearsEmployment: Math.round(emp),
    
    isJointApplication: get('isJointApplication', 0),
    incomeVerified: get('incomeVerified', 1),
    dtiRatio: get('dtiRatio', (monthlyPayment * 12 * 1.5) / annualIncome),
    revolvingBalance: revBal,
    revolvingUtilizationRate: get('revolvingUtilizationRate', 0.3),
    
    lengthCreditHistory: Math.round(get('lengthCreditHistory', emp + 3)),
    numTotalCreditLines: Math.round(numTot),
    numOpenCreditLines: Math.round(numOpn),
    numOpenCreditLines1Year: Math.round(get('numOpenCreditLines1Year', 1)),
    
    numDerogatoryRec: Math.round(get('numDerogatoryRec', 0)),
    numDelinquency2Years: Math.round(get('numDelinquency2Years', 0)),
    numChargeoff1year: Math.round(get('numChargeoff1year', 0)),
    numInquiries6Mon: Math.round(get('numInquiries6Mon', 0)),
    
    grade_score: get('grade_score', 4),
    
    loan_to_income_ratio: get('loan_to_income_ratio', loanAmount / annualIncome),
    payment_to_income_ratio: get('payment_to_income_ratio', (monthlyPayment * 12) / annualIncome),
    repayment_velocity: get('repayment_velocity', monthlyPayment / loanAmount),
    loan_amortization_rate: Math.max(0.001, get('loan_amortization_rate', 1 / term)),
    open_credit_ratio: get('open_credit_ratio', numTot > 0 ? numOpn / numTot : 0.5),
    recent_credit_velocity: get('recent_credit_velocity', 1),
    inquiry_intensity: get('inquiry_intensity', 0),
    delinquency_density: get('delinquency_density', 0),
    derogatory_density: get('derogatory_density', 0),
    estimated_credit_limit: get('estimated_credit_limit', revBal / 0.3),
    credit_utilization_recomputed: get('credit_utilization_recomputed', 0.3),
    
    log_loanAmount: get('log_loanAmount', Math.log(loanAmount + 1)),
    log_annualIncome: get('log_annualIncome', Math.log(annualIncome + 1)),
    log_revolvingBalance: get('log_revolvingBalance', Math.log(revBal + 1)),
    
    purpose_business: get('purpose_business', 0),
    purpose_debtconsolidation: get('purpose_debtconsolidation', 1),
    purpose_education: get('purpose_education', 0),
    purpose_healthcare: get('purpose_healthcare', 0),
    purpose_homeimprovement: get('purpose_homeimprovement', 0),
    purpose_other: get('purpose_other', 0),
    
    homeOwnership_own: get('homeOwnership_own', 0),
    homeOwnership_rent: get('homeOwnership_rent', 1),
    threshold: get('threshold', 0.50),
  };
  
  return payload;
}
