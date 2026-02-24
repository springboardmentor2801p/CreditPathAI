
# Database Setup & Data Migration — CreditPathAI

## Overview

The raw dataset (`loan_default_final_dataset.csv`) was migrated into a structured SQLite database (`loans.db`) to enable SQL-based validation, EDA, and seamless integration with the ML pipeline.

---

## Implementation Steps

###  Database Initialization

The SQLite database was created from the project directory:

```bash
sqlite3 loans.db
```

This generated the local database file:

```
loans.db
```

---

###  Schema Definition

A structured table `loans` was created with 48 columns, including:

* Financial attributes (e.g., `loan_amount`, `income`, `LTV`)
* Credit indicators (`Credit_Score`)
* Engineered risk flags
* Target variable: `Status`

All columns were explicitly typed (`INTEGER`, `REAL`, `TEXT`) to ensure schema consistency before modeling.

---

###  CSV Import

The dataset was imported using SQLite CSV mode:

```sql
.mode csv
.separator ","
.import loan_default_final_dataset.csv loans
```

Since SQLite imports the header row as data, it was removed post-import:

```sql
DELETE FROM loans WHERE loan_limit = 'loan_limit';
```

---

###  Verification

Data integrity was validated using:

```sql
SELECT COUNT(*) FROM loans;
```

**Result: 148,669 records successfully stored.**

Schema structure was confirmed using:

```sql
PRAGMA table_info(loans);
```

---

## Final Outcome

* `loans.db` created
* 48 structured features
* 148,669 verified records
* Database ready for EDA and ML modeling
