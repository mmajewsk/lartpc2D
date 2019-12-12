from scripts.classic_conv import conv_net_gdata_generator
import numpy as np
import data

if __name__ == "__main__":
    data_generator = data.LartpcData.from_path('../dump')
    Y = np.array([0.,0.,0.])
    for i,(x,y) in enumerate(conv_net_gdata_generator(data_generator)):
        if i==1000:
            break
        sm = np.squeeze(y).sum(axis=0).sum(axis=0)
        Y+=sm
    print(Y)
    print(sum(Y))
    frac = Y/sum(Y)
    print(frac)
    factor = 1./(Y/sum(Y))[0]
    perc = frac*factor
    print(perc)
    weights = 1/perc
    print(weights)
    # [1.         1.72747848 3.25204277]
    # [1.         1.77344934 3.15416369]
    # [1.         1.78342537 3.08500835]
    # [1.         1.79482279 3.01759431]
    # [1.         1.75350529 3.19063798]
    # avg :       1.76       3.13
    # as list: [1,1.76,3.13]
