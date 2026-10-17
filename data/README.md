# German Credit Dataset (Statlog German Credit Data)

## Overview
This directory contains the authentic benchmark **German Credit Dataset** originally compiled by Prof. Dr. Hans Hofmann at the University of Hamburg and hosted on the UCI Machine Learning Repository and OpenML (ID: 31, `credit-g`).

The dataset contains 1,000 instances of historical credit agreements evaluated for risk classification.

## Dataset Schema
- **Rows**: 1,000 credit applicant records
- **Features**: 20 attributes (7 numerical, 13 categorical)
- **Target**: `class` (Binary: `good` vs `bad`)
  - `good`: 700 instances (70.0%) — Low credit risk / creditworthy
  - `bad`: 300 instances (30.0%) — High credit risk / loan default risk

### Features Description
1. `checking_status`: Status of existing checking account (`<0`, `0<=X<200`, `>=200`, `no checking`)
2. `duration`: Credit duration in months (numerical)
3. `credit_history`: Past repayment track record (`critical/other existing credit`, `existing credits paid but no delays`, `all paid`, `delay in paying off in the past`, `no credits/all paid`)
4. `purpose`: Intended loan utilization (`radio/tv`, `education`, `furniture/equipment`, `new car`, `used car`, `business`, `domestic appliance`, `repairs`, `other`, `retraining`)
5. `credit_amount`: Total loan principal in Deutsche Mark (numerical)
6. `savings_status`: Value of savings bonds or accounts (`<100`, `100<=X<500`, `500<=X<1000`, `>=1000`, `no known savings`)
7. `employment`: Length of present employment tenure (`unemployed`, `<1`, `1<=X<4`, `4<=X<7`, `>=7`)
8. `installment_commitment`: Installment rate in percentage of disposable income (1, 2, 3, 4)
9. `personal_status`: Marital status and sex (`male single`, `female div/dep/mar`, `male mar/wid`, `male div/sep`)
10. `other_parties`: Other debtors or guarantors (`none`, `co applicant`, `guarantor`)
11. `residence_since`: Duration at current residence in years (1, 2, 3, 4)
12. `property_magnitude`: Most significant asset owned (`real estate`, `building society/life insurance`, `car`, `no known property`)
13. `age`: Age in years (numerical)
14. `other_payment_plans`: Other installment liabilities (`bank`, `stores`, `none`)
15. `housing`: Housing arrangement (`own`, `rent`, `for free`)
16. `existing_credits`: Number of existing credit contracts at this institution (numerical)
17. `job`: Employment classification (`skilled`, `unskilled resident`, `high qualif/self emp/mgmt`, `unemp/unskilled non res`)
18. `num_dependents`: Number of maintenance dependents (1 or 2)
19. `own_telephone`: Registered telephone number (`yes`, `none`)
20. `foreign_worker`: Foreign worker status (`yes`, `no`)
21. `class`: Ground truth credit risk label (`good` = Good Credit Risk, `bad` = Poor Credit Risk)

## Label Mapping & Asymmetric Cost
In credit risk underwriting, errors are asymmetric:
- **False Positive (Type I Error)**: Predicting an applicant is "Good Credit Risk" when they are actually "Poor Credit Risk". This results in loan default, causing direct capital loss (empirically rated as 5x more costly in the original German credit cost matrix).
- **False Negative (Type II Error)**: Predicting an applicant is "Poor Credit Risk" when they are actually "Good Credit Risk". This results in lost interest margin and friction.

The system explicitly documents and handles this trade-off during model training and evaluation.
