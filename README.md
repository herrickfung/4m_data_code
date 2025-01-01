# Data and codes for "Similarities and differences in the effects of different stimulus manipulations on accuracy and confidence".


**Project Description**:  

This project examines the similarities and differences between five different stimulus manipulations across two experiments on accuracy and confidence. 

---

## Installation

To set up this project locally, follow these steps:

### 1. Clone the repository:
Clone the repository to your local machine and navigate into the project folder:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies:
1. Python3.9
2. scipy==1.11.4
3. matplotlib==3.6.3
4. numpy==1.23.5
5. pandas==1.5.3
6. pingouin==0.5.3
7. rpy2==3.5.7

To install these dependencies, 
```bash
pip install -r requirements.txt
```

Additionally, R 4.3 is required for certain analyses. You can download it from:
https://cran.r-project.org
To run the analysis, the following R packages are required:
1. afex
```R
install.packages("afex")
```

### 3. Contents:

The repository includes the following structure:

- **`data_code/data/`**: Contains the lightly processed raw data from both experiments. These files are in a format ready for analysis.

- **`data_code/function/`**: Contains helper functions for the analysis scripts.

- **`data_code/results/`**: Contains the processed data, statistical results, and all figures in the article produced by the analysis script.

- **`data_code/analyze_expt_1.py`**: Analysis script for Experiment 1.
- **`data_code/analyze_expt_2.py`**: Analysis script for Experiment 2.

- **`requirements.txt`**: Lists all the required Python dependencies for the project.


