# Reduced Order Modeling of Turbulent Flows

This repository contains implementations and analysis tools for reduced-order modeling (ROM) and data-driven system identification applied to turbulent flow datasets. The project focuses on extracting low-dimensional dynamical representations from high-dimensional velocity field data using established and modern techniques.

This work was carried out as part of the **AS5401 – Data Driven Modelling of Complex Aerospace Systems** course at the **Indian Institute of Technology Madras**.

---

## Project Objectives

- Learn low-dimensional dynamical models from high-dimensional turbulent flow data  
- Analyze temporal correlations to guide appropriate time-step selection  
- Implement and compare data-driven reduced-order modeling techniques  
- Evaluate reconstruction and prediction performance of reduced-order models  

---

## Methods Implemented

The following methods are implemented in this repository:

- **Operator Inference (OpInf)**  
  Regression-based identification of reduced-order operators directly from data.

- **Sparse Identification of Nonlinear Dynamics (SINDy)**  
  Sparse regression framework to identify governing equations of the dynamics.

- **Discrete-Time System Identification**  
  Construction of discrete-time reduced-order dynamical systems.

- **Autocorrelation Analysis**  
  Temporal correlation analysis to identify dominant flow time scales and guide sampling decisions.

---

## Dataset

- **Source:** Johns Hopkins Turbulence Database (JHTDB)  
- **Data Type:** Velocity field snapshots  

Due to size constraints, raw datasets are not included in this repository. Scripts assume that snapshot matrices are loaded into the workspace prior to execution.

---

## Repository Structure

```
.
├── src/          # MATLAB source codes for ROM and system identification
├── notebooks/    # Jupyter notebooks for mean flow analysis and post-processing
├── results/      # Representative figures and tables
├── paper/        # Final project report (PDF)
└── README.md     # Project overview and instructions
```

---

## How to Use This Repository

1. Download or clone the repository  
2. Add the `src/` directory to the MATLAB path  
3. Load the required snapshot data into the MATLAB workspace  
4. Execute the desired scripts from the `src/` folder  
5. Use the notebooks in the `notebooks/` directory for analysis and visualization  

Each script contains internal documentation describing inputs, outputs, and computational steps.

---

## Results and Discussion

Key results, including reconstruction accuracy, temporal prediction behavior, and comparative performance of different reduced-order modeling techniques, are discussed in detail in the project report available in the `paper/` directory.

Representative plots and figures are stored in the `results/` folder.

---

## Project Report

- **Title:** Reduced Order Modeling of Turbulent Flows  
- **Course:** AS5401 – Advanced Fluid Mechanics  
- **Institute:** Indian Institute of Technology Madras  

The report includes:
- Mathematical formulation of the methods  
- Numerical implementation details  
- Error and performance analysis  
- Physical interpretation of the results  

---

## References

Peherstorfer, B., Willcox, K., and Gunzburger, M.  
*Data-driven operator inference for nonintrusive projection-based model reduction.*  
SIAM Journal on Scientific Computing, 2016.

Brunton, S. L., Proctor, J. L., and Kutz, J. N.  
*Discovering governing equations from data by sparse identification of nonlinear dynamical systems.*  
Proceedings of the National Academy of Sciences, 2016.

---

## Author

**Dhyan G**  
Dual Degree (B.Tech + M.Tech)  
Aerospace Engineering & Computational Engineering  
Indian Institute of Technology Madras  
