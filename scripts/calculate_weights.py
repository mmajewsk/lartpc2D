from scripts.classic_conv import conv_net_gdata_generator
from common_configs import ClassicConfConfig
import numpy as np
import data

if __name__ == "__main__":
    network_config = ClassicConfConfig()
    data_generator = data.LartpcData.from_path('../dump')
    Y = np.array([0.,0.,0.])
    for i,(x,y) in enumerate(conv_net_gdata_generator(data_generator, network_config)):
        if i==1000:
            print(x[0])
            print(y)
            break
        sm = np.squeeze(y)
        sm = sm.sum(axis=0)
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
    #
    #
    # with extended neighbours (3x3):
    # [1.         3.87610333 7.22036896]
    # [1.         4.04957124 6.59997143]
    # [1.         3.98504054 6.79593967]
    # [1.         3.98144402 6.81146786]
    # [1.         4.04999677 6.56913416]
    # avg: [1.        , 3.98, 6.79])
    #
    #
    # (5x5)
    # [1.         3.92143281 6.96875   ]
    # [1.         3.92742873 7.06346667]
    # [1.         4.11465123 6.4191551 ]
    # [1.         3.97702802 6.82774016]
    # [1.         3.9691315  6.89783113]
    # 3.98 6.83
