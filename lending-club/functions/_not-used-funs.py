"""Functions removed from other files."""


# Data types
# Function to convert 0/1 to No/Yes
def convert_01_to_no_yes(x):
    dtype_no_yes = pd.CategoricalDtype(categories=["No", "Yes"], ordered=True)
    return x.replace({0: "No", 1: "Yes"}).astype(dtype_no_yes)


# Function to convert False/True to No/Yes
def convert_bool_to_no_yes(x):
    return x.astype(int).pipe(convert_01_to_no_yes)


# Function to convert No/Yes to 0/1
def convert_no_yes_to_01(x):
    return x.replace({"No": 0, "Yes": 1}).astype(np.int8)


# Function to convert False/True to 0/1
def convert_bool_to_01(x):
    return x.astype(np.int8)


# Function to convert No/Yes to -1/1
def convert_no_yes_to_mp1(x):
    return x.replace({"No": -1, "Yes": 1}).astype(np.int8)
