import pandas as pd

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    n_trials = 100

    # rewrite this file with 1s and 0s using n trials
    data = pd.read_csv(r"C:\Users\44756\Documents\VR_behaviour_power\Figure3_C_0100.csv")
    print("hello")

    rewritten = pd.DataFrame(columns=['Mouse','track_length','correct','condition_b1_p2'])
    rewritten = pd.DataFrame()


    for i in range(len(data)):
        row = data.iloc[i]

        Mouse = row["Mouse"]
        track_length = row["Reward_zone"]
        correct_trials = row["correction trials"]
        condition_b1_p2 = row["condition_b1_p2"]

        for j in range(n_trials):
            if (correct_trials >= j):
                df = pd.DataFrame({"Mouse":[Mouse], "track_length":[track_length], "correct": [1.0], "condition_b1_p2": [condition_b1_p2]})
                rewritten = rewritten.append(df)
            else:
                df = pd.DataFrame({"Mouse":[Mouse], "track_length":[track_length], "correct": [0.0], "condition_b1_p2": [condition_b1_p2]})
                rewritten = rewritten.append(df)

    rewritten.to_csv(r"C:\Users\44756\Documents\VR_behaviour_power\Figure3_C_0100_long_format.csv")
    print("hello there")

if __name__ == '__main__':
    main()