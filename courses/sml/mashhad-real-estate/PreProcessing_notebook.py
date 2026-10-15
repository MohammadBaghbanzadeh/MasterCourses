# Exported notebook code; may contain IPython commands.
# %%
!pip install pandas numpy

# %%
import pandas as pd
import numpy as np
import re

# %%
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def normalize_text(x):
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.translate(PERSIAN_DIGITS)
    s = s.replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"\s+", " ", s).strip()

def clean_price(x):
    s = normalize_text(x)
    s = re.sub(r"[^\d]", "", s)
    return pd.to_numeric(s, errors="coerce")

def clean_age(x):
    s = normalize_text(x)
    if "تازه" in s:
        return 0
    s = s.replace("سال ساخت", "")
    return pd.to_numeric(s, errors="coerce")

def clean_floor(x):
    s = normalize_text(x)
    if "همکف" in s:
        return 0
    return pd.to_numeric(s, errors="coerce")

# %%
def extract_features(desc):
    s = normalize_text(desc)

    return pd.Series({
        "parking": int("پارکینگ" in s and "ندارد" not in s),
        "anbari": int("انباری" in s and "ندارد" not in s),
        "asansor": int("آسانسور" in s and "ندارد" not in s),
        "renovated": int("نوساز" in s or "بازسازی" in s),
        "shop": int("مغازه" in s),
        "south": int("جنوبی" in s),
        "north": int("شمالی" in s),
    })

# %%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()

# %%
train["price"] = train["price"].apply(clean_price)
train["age"] = train["age"].apply(clean_age)
train["floor"] = train["floor"].apply(clean_floor)

train["type"] = train["type"].apply(lambda x: 0 if "آپارتمان" in str(x) else 1)

train["region"] = train["neighborhood"].apply(
    lambda x: int(str(x).split()[1]) if len(str(x).split()) > 1 else np.nan
)

# %%
features = train["description"].fillna("").apply(extract_features)
train = pd.concat([train, features], axis=1)

train.head()

# %%
train = train.dropna(subset=["price", "area", "bedrooms", "latitude", "longitude"])
train = train.reset_index(drop=True)

# %%
train.to_csv("train_processed.csv", index=False)
print("Saved!")

# %%


# %%

