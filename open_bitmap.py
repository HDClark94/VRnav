import pandas as pd
import os
import imageio
import numpy as np

def read_bitmaps(path):
    a = [f.path for f in os.scandir(path)]

    ps = []
    for i in range(len(a)):
        if a[i].split(".")[-1] == "png":
            im = imageio.imread(a[i])
            im = np.sum(im, axis=2)
            im[im<500] = 0
            im[im>500] = 1

            p = np.mean(im)
            ps.append(p)

    ps = np.array(ps)
    x = np.mean(ps)

    print("Thanks we're done here")

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    path = r"C:\Users\44756\Google Drive\PhD Edinburgh\vr_oculus\bitmaps"

    read_bitmaps(path)
    print("look now")

if __name__ == '__main__':
    main()