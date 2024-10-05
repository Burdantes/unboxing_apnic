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


# TO-DO List

This file contains a list of tasks and features that are pending or planned for future releases of the project.

## Pending Tasks

1. 
   - [ ] Finalize the export of the code by the ACM IMC 2024 conference.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

