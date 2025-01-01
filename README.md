# Data and codes for "Similarities and differences in the effects of different stimulus manipulations on accuracy and confidence".


**Project Description**:  

Visual stimuli can vary in multiple dimensions that affect accuracy and confidence in a perceptual decision-making task. 
However, previous studies have typically included just one or at most two manipulations, 
leaving it unclear how a wider set of manipulations may relate to each other. 
Here, we examine the similarities and differences between five different stimulus manipulations across two experiments. 
Subjects indicated whether a tilted Gabor patch was oriented clockwise or counterclockwise from 45°. 
In Experiment 1, we independently manipulated the size, duration, noise level, and tilt offset of the stimuli. 
In Experiment 2, we employed a 2x2x2x2 design jointly manipulating size, spatial frequency, noise level, and tilt offset of the stimuli. 
We found that manipulations of size, duration, noise level, and spatial frequency had remarkably similar effects on accuracy and confidence. 
In contrast, the tilt offset manipulation stood out by affecting accuracy more strongly than confidence. 
In addition, tilt offset exhibited supraadditive interaction with all other manipulations for both accuracy and confidence, 
whereas the remaining manipulations exhibited either no interactions or subadditive interactions with each other. 
Furthermore, tilt offset was also the only manipulation for which confidence in incorrect trials decreased with increasing difficulty, 
while all other manipulations exhibited the opposite trend. 
Overall, our results reveal a startling similarity between the effects of four very different stimulus manipulations 
and a prominent difference with a fifth manipulation. These results may allow us to predict how future, 
yet untested manipulations would affect accuracy and confidence.


---

## Table of Contents

1. [Installation](#installation)
2. [Requirements](#requirements)


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

Additionally, R is required for certain analyses. You can download it from:
https://cran.r-project.org
To run the analysis, the following R packages are required:
1. afex

```bash
pip install -r requirements.txt
