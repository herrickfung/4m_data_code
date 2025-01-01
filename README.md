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

| Column Name         | Description                                                                                       |
|---------------------|---------------------------------------------------------------------------------------------------|
| **subject_ID**       | Unique identifier for each participant in the experiment. All trials belonging to the same subject will share the same subject_ID. |
| **trial_no**         | Sequential number of the trial within a subject.       |
| **stim_condition** / **condition**  | The experimental condition coded numerically.  |
| **text_stim_condition** | A textual description of the experimental condition associated with the trial. |
| **binary_condition** | Binary conversion of the experimental condition for Experiment 2. Each digit will refer to size, spatial frequency, noise, and tilt offset, respectively; 0 is Easy, 1 is Hard. E.g. 1100 would indicate hard setting for size and spatial frequency, and easy setting for noise and tilt offset.|
| **answer**           | The stimulus tilt coded numerically. 1 = tilt more to left, 0 = tilt more to right. |
| **percept_resp**     | The subject’s perceptual response to the stimulus. 1 = tilt more to left, 0 = tilt more to right. |
| **conf_resp**        | The subject's confidence response to the stimulus, ranged from 1 to 4. 1 = low confidence, 4 = high confidence |
| **correct**          | Accuracy of the subject's perceptual response. 1 = correct, 0 = incorrect. |
| **percept_rt**       | The reaction time (RT) for the perceptual response, measured in milliseconds. |
| **conf_rt**          | The reaction time (RT) for the confidence response, measured in milliseconds.  |
| **stim_size**        | The size of the stimulus presented during the trial, in pixel value. |
| **stim_dur**         | The duration in milliseconds for which the stimulus was presented.  |
| **stim_sf**          | The spatial frequency level of the stimulus, in cycles per degree. |
| **stim_contrast**    | The noise contrast level of the stimulus, ranging from 0 to 1, 0 = no noise, 1 = complete noise |
| **stim_tilt_diff**   | The tilt offset level of the stimulus, in degree. |

