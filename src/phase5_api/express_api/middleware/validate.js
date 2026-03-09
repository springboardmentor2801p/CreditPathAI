const { body, validationResult } = require("express-validator");

// ── Valid enum values derived from production data ────────────────────────────
const VALID_STATES = [
  "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
  "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
  "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
  "TX","UT","VT","VA","WA","WV","WI","WY","DC",
];

const VALID_EMPLOYMENT = [
  "< 1 year", "1 year", "2-5 years", "6-9 years", "10+ years",
];

const VALID_OWNERSHIP = ["rent", "own", "mortgage"];

const VALID_PURPOSE = [
  "debtconsolidation", "homeimprovement", "business", "medical",
  "majorpurchase", "smallbusiness", "car", "vacation", "wedding",
  "movingandrelo", "education", "renewable_energy", "other",
];

const VALID_TERM = ["36 months", "48 months", "60 months"];

// ── Borrower validators ───────────────────────────────────────────────────────
const borrowerRules = [
  body("borrower.residentialState")
    .isString().trim().toUpperCase()
    .isIn(VALID_STATES)
    .withMessage("Invalid US state code"),

  body("borrower.yearsEmployment")
    .isString().trim()
    .isIn(VALID_EMPLOYMENT)
    .withMessage(`Must be one of: ${VALID_EMPLOYMENT.join(", ")}`),

  body("borrower.homeOwnership")
    .isString().trim().toLowerCase()
    .isIn(VALID_OWNERSHIP)
    .withMessage("Must be: rent | own | mortgage"),

  body("borrower.annualIncome")
    .isFloat({ min: 1 })
    .withMessage("Annual income must be a positive number"),

  body("borrower.incomeVerified")
    .isInt({ min: 0, max: 1 })
    .withMessage("incomeVerified must be 0 or 1"),

  body("borrower.dtiRatio")
    .isFloat({ min: 0 })
    .withMessage("DTI ratio must be >= 0"),

  body("borrower.lengthCreditHistory")
    .isInt({ min: 0 })
    .withMessage("Length of credit history must be >= 0"),

  body("borrower.numTotalCreditLines")
    .isInt({ min: 0 })
    .withMessage("Must be >= 0"),

  body("borrower.numOpenCreditLines")
    .isInt({ min: 0 })
    .withMessage("Must be >= 0"),

  body("borrower.numOpenCreditLines1Year")
    .isInt({ min: 0 })
    .withMessage("Must be >= 0"),

  body("borrower.revolvingBalance")
    .isFloat({ min: 0 })
    .withMessage("Revolving balance must be >= 0"),

  body("borrower.revolvingUtilizationRate")
    .isFloat({ min: 0, max: 150 })
    .withMessage("Revolving utilization rate must be 0-150"),

  body("borrower.numDerogatoryRec")
    .isInt({ min: 0 })
    .withMessage("Must be >= 0"),

  body("borrower.numDelinquency2Years")
    .isInt({ min: 0 })
    .withMessage("Must be >= 0"),

  body("borrower.numChargeoff1year")
    .isInt({ min: 0 })
    .withMessage("Must be >= 0"),

  body("borrower.numInquiries6Mon")
    .isInt({ min: 0 })
    .withMessage("Must be >= 0"),
];

// ── Loan validators ───────────────────────────────────────────────────────────
const loanRules = [
  body("loan.purpose")
    .isString().trim().toLowerCase()
    .isIn(VALID_PURPOSE)
    .withMessage(`Invalid purpose. Valid: ${VALID_PURPOSE.join(", ")}`),

  body("loan.isJointApplication")
    .isInt({ min: 0, max: 1 })
    .withMessage("isJointApplication must be 0 or 1"),

  body("loan.loanAmount")
    .isFloat({ min: 1 })
    .withMessage("Loan amount must be > 0"),

  body("loan.term")
    .isString().trim()
    .isIn(VALID_TERM)
    .withMessage("Term must be: 36 months | 48 months | 60 months"),

  body("loan.interestRate")
    .isFloat({ min: 0.1 })
    .withMessage("Interest rate must be > 0"),

  body("loan.monthlyPayment")
    .isFloat({ min: 1 })
    .withMessage("Monthly payment must be > 0"),

  body("loan.grade")
    .isString().trim().toUpperCase()
    .matches(/^[A-E][1-5]$/)
    .withMessage("Grade must be A1-E5 format (e.g. C3, A1, E2)"),
];

// ── Error formatter ───────────────────────────────────────────────────────────
const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(422).json({
      success: false,
      error:   "Validation failed",
      details: errors.array().map((e) => ({
        field:   e.path,
        message: e.msg,
        value:   e.value,
      })),
    });
  }
  next();
};

module.exports = {
  validatePredictRequest: [...borrowerRules, ...loanRules, handleValidationErrors],
};
