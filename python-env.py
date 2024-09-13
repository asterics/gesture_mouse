import os
print(f"PATH: {os.environ['PATH']}")

#os.environ['PYTHONPATH']

import sys
print(f"sys.prefix: {sys.prefix}")

print("------- show sklearn -----")
import sklearn
sklearn.show_versions()


print("----- Importing numpy and show config ------")
import numpy

numpy.show_config()
#print(numpy.inf)