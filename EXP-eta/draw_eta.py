import matplotlib.pyplot as plt
import numpy as np

def draw():
    x = [0.001, 0.1, 0.5, 0.8, 1, 1.5]
    Test_acc = [95.23, 95.20, 95.38, 95.14, 95.04, 94.56]
    Params = [8.9, 5.9, 6.2, 5.3, 4.1, 3.8]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, Test_acc, 'k--', label='Test-acc')
    ax1.set_xlabel(r'$\eta$')
    ax1.set_ylabel('Test-acc(%)')

    ax2 = ax1.twinx()
    ax2.plot(x, Params, 'k-', label='Params')
    ax2.set_ylabel('Params(MB)')

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.87, 0.9))

    ax1.set_title(r'The impact of $\eta$ on Params and Test-acc')

    plt.show()

    plt.savefig('eta_res')

draw()
