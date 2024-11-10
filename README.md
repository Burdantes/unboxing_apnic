# What’s in the Dataset? Unboxing the APNIC per AS User Population Datase
Official Git Repository for the  ``What’s in the Dataset? Unboxing the APNIC per AS User Population Dataset. ''

## Installation
### Step 1: Install virtualenv (if not installed)

First, make sure you have virtualenv installed. If not, you can install it using pip:
   
 ```bash
    pip install virtualenv
  ```
### Step 2: Create a Virtual Environment

To create a virtual environment, run the following command in your project directory:

```bash
    virtualenv venv
 ```
### Step 3: Activate the Virtual Environment

To activate the virtual environment, run the following command:

```bash
    source venv/bin/activate
 ```

### Step 4: Install the Required Libraries

To install the required libraries, run the following command:

```bash
    pip install -r py-requirements.txt
 ```

## Usage
We mainly use Jupyter notebooks for data analysis and visualization. The following steps will guide you through the process of analyzing the data and generating figures.
A large portion of our script relies on AnonCDN dataset which we are not allowed to share sadly. 

### Downloading APNIC dataset longitudinally 

### Downloading PeeringDB dataset 

Run the following command to download the PeeringDB dataset:
```
python src/PeeringDB.py
```

You can change the date of the dataset by changing the date in the script.

### Comparing Broadband Dataset with APNIC

Run the following command to compare the APNIC dataset with the Broadband dataset:
```
python src/broadband_comparison.py
```


2. Analyze the data with:
    ```bash
    jupyter notebook src/analysis.ipynb
    ```
3. Generate figures:
    ```bash
    python src/visualization.py
   ```

### Step 5: Comparing AnonCDN with APNIC

Sadly, we cannot share the AnonCDN dataset. However, we can share our correlations score in a txt file for traffic ([cdn_correlations_score_traffic_volume.txt](results%2Fcdn_correlations_score_traffic_volume.txt)) and for User-Agent ([cdn_correlations_score_user_agents.txt](results%2Fcdn_correlations_score_user_agents.txt)).
#### File Format

Each line in the file is formatted as follows:
   
 ```
<CC_ISO2>: Kendall Tau: <Kendall Tau value>, Pearson: <Pearson correlation coefficient>, Linear Coefficient: <Linear regression coefficient>
```
a. *CC_ISO2*: The two-letter country code of the country.

b. *Kendall Tau*: The Kendall Tau rank correlation coefficient between two variables. It measures the ordinal association, assessing how well the relationship between two variables can be described using a monotonic function. Values range from -1 (perfect negative association) to +1 (perfect positive association), with 0 indicating no association.

c. *Pearson*: The Pearson correlation coefficient between two variables. It measures the linear correlation, quantifying the strength and direction of the linear relationship. Values range from -1 to +1, where: +1: Perfect positive linear relationship.  0: No linear relationship. -1: Perfect negative linear relationship.

d. *Linear Coefficient*: The slope from a linear regression analysis. It represents the expected change in the dependent variable for a one-unit change in the independent variable.

# TO-DO List

This file contains a list of tasks and features that are pending or planned for future releases of the project.

## Pending Tasks

1. 
   - [ ] Finalize the export of the temporal study by the ACM IMC 2024 conference.
   - [ ] Automate aggregation mechanism to improve stability for sharing before the first week of November. 
   - [ ] Share list of trustworthy countries in a txt file according to AnonCDN comparison.


## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

