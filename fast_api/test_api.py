import sqlite3
import pandas as pd
import requests

db_path = r'z:\CreditPathAI\csv2database\creditpathai.db'
conn = sqlite3.connect(db_path)

try:
    # First describe the schema to see if loanStatus is present
    schema_df = pd.read_sql('PRAGMA table_info(processed_loans)', conn)
    has_status = 'loanStatus' in schema_df['name'].values
    print("Has loanStatus column:", has_status)
    
    if has_status:
        df_0 = pd.read_sql('SELECT * FROM processed_loans WHERE loanStatus = 0 LIMIT 5', conn)
        df_1 = pd.read_sql('SELECT * FROM processed_loans WHERE loanStatus = 1 LIMIT 5', conn)
    else:
        print("loanStatus column missing in processed_loans. Trying to find a source file.")
        
        # fallback to training data if available
        # we can try to load test data from the advanced training folder
        try:
            df = pd.read_csv(r'z:\CreditPathAI\training\advanced\test_data.csv')
            df_0 = df[df['loanStatus'] == 0].head(5)
            df_1 = df[df['loanStatus'] == 1].head(5)
            print("Loaded rows from test_data.csv")
        except FileNotFoundError:
            try:
                # wait, let me check preprocessed data 
                df = pd.read_csv(r'z:\CreditPathAI\preprocessing\engineered_unscaled.csv')
                df_0 = df[df['loanStatus'] == 0].head(5)
                df_1 = df[df['loanStatus'] == 1].head(5)
                print("Loaded rows from engineered_unscaled.csv")
            except Exception as e:
                print("Could not load from CSV fallbacks", e)
                df_0 = pd.DataFrame()
                df_1 = pd.DataFrame()
            
except Exception as e:
    print('Failed to read data:', e)
    df_0 = pd.DataFrame()
    df_1 = pd.DataFrame()

conn.close()

def test_rows(df, actual_status):
    if df.empty:
        print(f'No data found for loanStatus={actual_status}')
        return
    
    import sys, os
    sys.path.insert(0, r"z:\CreditPathAI")
    from recommendation_engine.recommend_engine import load_model, recommend

    model, preprocessor, feature_names = load_model()

    result_str = f'\n=== Testing {len(df)} rows with ACTUAL loanStatus = {actual_status} ===\n'
    print(result_str.strip())
    with open('test_results.txt', 'a') as f:
        f.write(result_str)
        
    correct = 0
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            res = recommend(row_dict, model, preprocessor, feature_names)
            pred = 1 if res['predicted_default'] else 0
            prob = res['default_probability']
            res_str = "PASS" if pred == actual_status else "FAIL"
            out = f"Row {idx+1}: {res_str} | Pred={pred} (Prob: {prob:.4f}) | Band={res['risk_band']}\n"
            print(out.strip())
            with open('test_results.txt', 'a') as f:
                f.write(out)
            if pred == actual_status: correct += 1
        except Exception as e:
            err = f"Error on Row {idx+1}: {e}\n"
            print(err.strip())
            with open('test_results.txt', 'a') as f:
                f.write(err)
    
    summary = f"--> Class {actual_status} Accuracy: {correct}/{len(df)} ({(correct/len(df))*100:.1f}%)\n"
    print(summary.strip())
    with open('test_results.txt', 'a') as f:
        f.write(summary)

# Clear file before run
open('test_results.txt', 'w').close()
test_rows(df_0, 0)
test_rows(df_1, 1)

