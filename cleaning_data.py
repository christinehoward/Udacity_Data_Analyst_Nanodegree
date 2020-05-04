# remove extra unnecessary letters

# example: remove bb before every animal name using string slicing
df['Animal'] = df['Animal'].str[2:]
# cuts after the first 2 letters

# replace incorrect symbol with correct

# example: replace ! with . in body weight and brain weight columns
df['Body Weight (kg)'] = df['Body Weight (kg)'].str.replace('!', '.')
df['Brain Weight (g)'] = df['Brain Weight (g)'].str.replace('!', '.')

# Convert zip code column's data type from a float to a string using astype
astype(str)
# Remove the '.0' using string slicing
str[:-2]
# Pad 4 digit zip codes with a leading 0
str.pad(5, fillchar='0') # we want 5 digits, when there aren't 5 digits, fill the first digit with 0

patients_clean.zip_code = patients_clean.zip_code.astype(str).str[:-2].str.pad(5, fillchar='0')


# First make a copy of the dfs
patients_clean = patients.copy()
treatments_clean = treatments.copy()
adverse_reactions_clean = adverse_reactions.copy()

# Missing data

# treatments: Missing records (280 instead of 350)

# Define
# Import the cut treatments into a DataFrame and concatenate it with the original treatments DataFrame.

# Code:
treatments_cut = pd.read_csv('treatments_cut.csv')
treatments_clean = pd.concat([treatments_clean, treatments_cut], ignore_index=True)
# ignore_index=True, do not use the index values along the concatenation axis. The resulting axis will be labeled 0, …, n - 1. 


# treatments: Missing HbA1c changes and Inaccurate HbA1c changes (leading 4s mistaken as 9s)

# Define
# Recalculate the hba1c_change column: hba1c_start minus hba1c_end.

# Code:
treatments_clean.hba1c_change = (treatments_clean.hba1c_start - treatments_clean.hba1c_end)


# First addressing missing data, then cleaning for tidiness

# Tidiness

# Contact column in patients table contains two variables: phone number and email

# Define
# Extract the phone number and email variables from the contact column using regular expressions and pandas' str.extract method. Drop the contact column when done.

# Code RENAN
patients_clean['phone_number'] = patients_clean.contact.str.extract('((?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})', expand=True)
# [a-zA-Z] to signify emails in this dataset all start and end with letters
patients_clean['email'] = patients_clean.contact.str.extract('([a-zA-Z][a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+[a-zA-Z])', expand=True)
# Note: axis=1 denotes that we are referring to a column, not a row
patients_clean = patients_clean.drop('contact', axis=1)

# Test
# confirm contact column is gone
list(patients_clean)

patients_clean.phone_number.sample()

# confirm no emails start with an integer
patients_clean.email.sort_values().head()


# Three variables in two columns in treatments table (treatment, start dose and end dose)

# Define
# Melt the auralin and novodra columns to a treatment and a dose column (dose will still contain both start and end dose at this point). Then split the dose column on ' - ' to obtain start_dose and end_dose columns. Drop the intermediate dose column.

# Code RENAN
treatments_clean = pd.melt(treatments_clean, id_vars=['given_name', 'surname', 'hba1c_start', 'hba1c_end', 'hba1c_change'],
                            var_name='treatment', value_name='dose')
treatments_clean = treatments_clean[treatments_clean.dose != "-"]
treatments_clean['dose_start'], treatments_clean['dose_end'] = treatments_clean['dose'].str.split(' - ', 1).str
treatments_clean = treatments_clean.drop('dose', axis=1)


# Adverse reaction should be part of the treatments table

# Define
# Merge the adverse_reaction column to the treatments table, joining on given name and surname.

# Code
treatments_clean = pd.merge(treatments_clean, adverse_reactions_clean,
                            on=['given_name', 'surname'], how='left')


# Given name and surname columns in patients table duplicated in treatments and adverse_reactions tables and Lowercase given names and surnames

# Define
# Adverse reactions table is no longer needed so ignore that part. Isolate the patient ID and names in the patients table, then convert these names to lower case to join with treatments. Then drop the given name and surname columns in the treatments table (so these being lowercase isn't an issue anymore).

# Code
id_names = patients_clean[['patient_id', 'given_name', 'surname']]
id_names.given_name = id_names.given_name.str.lower() # changing name to lowercase
id_names.surname = id_names.surname.str.lower() # changing name to lowercase
treatments_clean = pd.merge(treatments_clean, id_names, on=['given_name', 'surname']) # getting rid of name columns, replacing with id
treatments_clean = treatments_clean.drop(['given_name', 'surname'], axis=1)

# Patient ID should be the only duplicate column

all_columns = pd.Series(list(patients_clean) + list(treatments_clean))
all_columns[all_columns.duplicated()]


# Cleaning for quality

