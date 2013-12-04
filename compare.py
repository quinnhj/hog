import numpy as np
import sys
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter

if len(sys.argv) < 2:
    print "Too few arguments"
    sys.exit(1)

THRESHOLD = 5
count = 0
seen = False;
index = 0;
with open ('output/compare_results.txt', 'w') as out:
    with open(sys.argv[1], 'r') as f1:
        with open(sys.argv[2], 'r') as f2:
            for line1 in f1:
                index += 1
                line2 = f2.readline()
                if float(line1) == float(line2):
                    out.write('0\n')

                else:
                    dif = abs(float(line1) - float(line2))
                    half_sum = 0.5 * (float(line1) + float(line2))
                    pct_dif = 100 * (dif / half_sum)
                    out.write(str(pct_dif) + '\n')
                    if pct_dif > THRESHOLD and dif > 0.001:
                        if not seen:
                            seen = True
#print 'Error at ' + str(index)
                            
                        count += 1  

print count



