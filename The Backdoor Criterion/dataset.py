import numpy as np
import pandas as pd


SEED = 42


def generate_data(n: int = 2_000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    u_ses  = rng.normal(0, 1, n)
    u_edu  = rng.normal(0, 1, n)
    u_job  = rng.normal(0, 1, n)
    u_inc  = rng.normal(0, 1, n)

    ses = 50 + 10 * u_ses

    education = 12 + 0.15 * ses + 2 * u_edu
    education = np.clip(education, 0, 25)

    job_level = np.clip(np.round(0.3 * education - 2 + u_job), 0, 5)

    income = 20_000 + 2_000 * education + 800 * ses + 5_000 * u_inc

    df = pd.DataFrame({
        "SES": np.round(ses, 2),
        "Education": np.round(education, 2),
        "JobLevel": job_level.astype(int),
        "Income": np.round(income, 2),
    })
    return df


if __name__ == "__main__":
    df = generate_data()
    print(df.describe().T.round(2))
    df.to_csv("data.csv", index=False)
