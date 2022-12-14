import utils as ut


def test_qubit_qracs(seeds):
    for j in range(2, 4):
        p = ut.find_QRAC_value(j, 2, seeds)
        print(p)


def test_higher_dim_qracs(seeds):
    for j in range(3, 5):
        p = ut.find_QRAC_value(2, j, seeds)
        print(p)


if __name__ == "__main__":
    seeds = 5
    # test_qubit_qracs(seeds)
    # test_higher_dim_qracs(seeds)
    p = ut.find_QRAC_value(2, 5, seeds)
    print(p)
