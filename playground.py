import pandas as pd

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    path = r"Z:\ActiveProjects\Harry\MouseOF\data\Cue_conditioned_cohort1_190902\M3_D10_2019-09-30_13-59-37\MountainSort\DataFrames\spatial_firing.pkl"
    spatial_firing = pd.read_pickle(path)
    print("look now")

if __name__ == '__main__':
    main()