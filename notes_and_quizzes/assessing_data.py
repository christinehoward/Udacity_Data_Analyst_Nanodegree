# Finding rows with null values

patients[patients['address'].isnull()]
# which rows in the address column have null values?

patients[patients.address.duplicated()]
# duplicated rows

weight_lbs = patients[patients.surname == 'Zaitseva'].weight * 2.20462 # changing kg to lbs
height_in = patients[patients.surname == 'Zaitseva'].height
bmi_check = 703 * weight_lbs / (height_in * height_in)
bmi_check
# computing BMI

patients[patients.surname == 'Zaitseva'].bmi 
# checking patient's BMI in dataset

sum(treatments.auralin.isnull())
# receives total rows with null values in auralin column in treatments df

all_columns = pd.Series(list(patients) + list(treatments) + list(adverse_reactions))
all_columns[all_columns.duplicated()]
# finding duplicate column names

