from base import *
from kernels import *

if __name__ == "__main__":
    Explorer(lambert, -5, 2, -2, 2, 1000, 1000, interpolation='bilinear')
    plt.show()









    # create_image(lambert, -5, 2, -2, 2, 3000, 2000, path='lambert.png')

