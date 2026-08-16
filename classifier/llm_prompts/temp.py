import pandas as pd
import re
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv("/u/spa-d4/grad/mfe261/Projects/TeacherApproved/classifier/annotated_data_privacy/final_classified.csv") 


#df_meta_2 = pd.read_csv("/u/siddique-d1/Moghis/mobilerec/apps_info_deduplicated.csv")


#df_meta = pd.read_csv("/u/siddique-d1/Moghis/mobilerec/app_meta.csv") 

#df['app_name'] = df.merge(
#    df_meta[['app_package', 'app_name']],
#    on='app_package',
#    how='left'
#)['app_name']

#df["permissions"] = df.merge(
#    df_meta_2[['app_package', 'permission']],
#    on='app_package',
#    how='left'
#)['permission']

#output_path = "/u/spa-d4/grad/mfe261/Projects/TeacherApproved/classifier/annotated_data_privacy/final_classified_with_appname_permissions.csv"
#df.to_csv(output_path, index=False)



# 1. Load the CSV file
df = pd.read_csv("/u/siddique-d1/Moghis/mobilerec/mobilerec_final.csv")   # change to your filename

df_meta_2 = pd.read_csv("/u/siddique-d1/Moghis/mobilerec/apps_info_deduplicated.csv")

df['app_name'] = df.merge(
    df_meta_2[['app_package', 'app_name']],
    on='app_package',
    how='left'
)['app_name']

df["permissions"] = df.merge(
    df_meta_2[['app_package', 'permission']],
    on='app_package',
    how='left'
)['permission']


df['date'] = pd.to_datetime(df['date'])

cutoff = pd.Timestamp("2021-03-18")

df_before_equal = df[df['date'] <= cutoff]
df_after = df[df['date'] > cutoff]

df_before_equal.to_csv("/u/siddique-d1/Moghis/mobilerec/mobilerec_before_or_on_2021_03_18.csv", index=False)
df_after.to_csv("/u/siddique-d1/Moghis/mobilerec/mobilerec_after_2021_03_18.csv", index=False)

sample_fraction = 0.1   # change to 0.05 for 5%, etc.
df_sample = df.sample(frac=sample_fraction, random_state=42)

# 2. Specify the review column name
review_col = "review"   # replace if your column name is different

# 3. Define all Privacy/Security keywords (from your list and reference)
keywords = [
    "privacy", "security", "safe", "secure", "permission", "identity", 
    "personal", "virus", "malware", "malicious", "access", "fishy", 
    "phishing", "fishing", "stealth", "steal", "thief", "creepy",
    "intrusive", "personal info"  # from general keywords
]

# 4. Build a regex pattern for whole-word, case-insensitive matching
pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"

# 5. Filter rows where the review text contains any keyword
filtered_df = df[df[review_col].str.contains(pattern, case=False, na=False, regex=True)]

# 6. (Optional) Add a column showing which keyword(s) were found
def find_matches(text):
    text = str(text).lower()
    return [k for k in keywords if k.lower() in text]

filtered_df["matched_keywords"] = filtered_df[review_col].progress_apply(find_matches)

filtered_df