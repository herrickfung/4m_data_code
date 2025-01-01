# Data and codes for "Similarities and differences in the effects of different stimulus manipulations on accuracy and confidence".


## Project Description

This project examines the similarities and differences between five different stimulus manipulations across two experiments on accuracy and confidence. 

These results are published in XXX (URL).

---

## Installation

To set up this project locally, follow these steps:

### 1. Clone the repository:
Clone the repository to your local machine and navigate into the project folder:
```bash
git clone https://github.com/herrickfung/4m_data_code.git
cd 4m_data_code
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

---

## Contents

The repository includes the following structure:

- **`data_code/data/`**: Contains the lightly processed raw data from both experiments. These files are in a format ready for analysis.

- **`data_code/function/`**: Contains helper functions for the analysis scripts.

- **`data_code/results/`**: Contains the processed data, statistical results, and all figures in the article produced by the analysis script.

- **`data_code/analyze_expt_1.py`**: Analysis script for Experiment 1.
- **`data_code/analyze_expt_2.py`**: Analysis script for Experiment 2.

- **`requirements.txt`**: Lists all the required Python dependencies for the project.

---

## Data Column README

| Column Name         | Description                                                                                       | Example        |
|---------------------|---------------------------------------------------------------------------------------------------|----------------|
| **subject_ID**       | Unique identifier for each participant in the experiment. All trials belonging to the same subject will share the same subject_ID. | `1`            |
| **trial_no**         | Sequential number of the trial within a subject. This helps to track the order of the trials.      | `1`            |
| **stim_condition**   | The experimental condition under which the stimulus was presented. This variable categorizes the type of stimulus shown during the trial. | `4`            |
| **answer**           | The participant's response to the stimulus, coded as a numerical value representing the choice the participant made in response to the stimulus. | `0`            |
| **percept_resp**     | The participant’s perceptual response to the stimulus, also coded numerically. This can reflect how the participant interpreted or perceived the stimulus. | `1`            |
| **conf_resp**        | The confidence response, coded numerically, indicating the participant’s confidence level in their answer. Higher values often indicate higher confidence. | `2`            |
| **correct**          | Indicates whether the participant’s response was correct. A value of `1` means correct, and `0` means incorrect. | `0`            |
| **percept_rt**       | The reaction time (RT) for the perceptual response, measured in milliseconds. This shows how long it took the participant to respond to the stimulus. | `541`          |
| **conf_rt**          | The reaction time (RT) for the confidence response, measured in milliseconds. This reflects how long it took the participant to decide on their confidence level after responding perceptually. | `2267`         |
| **stim_size**        | The size of the stimulus presented during the trial, coded as a numerical value (could represent dimensions like width or height in some unit). | `150`          |
| **stim_dur**         | The duration in milliseconds for which the stimulus was presented. This indicates how long the stimulus remained on screen. | `500`          |
| **stim_contrast**    | The contrast level of the stimulus, typically ranging from `0` (no co
