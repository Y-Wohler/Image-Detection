#!/usr/bin/env python3
# This is template for your program, for you to expand with all the correct
# functionality.

import sys


background = [51, 108, 23, 255, 252, 0]  ## H_min, H_max, S_min, S_max, V_min, V_max
redBall = [153, 179, 37, 255, 188, 255]
RED_POINTER_HSV_VALUES = [153, 179, 37, 255, 188, 255]
RED_POINTER_HSV_VALUES = [176, 179, 70, 255, 0, 255]

#-------------------------------------------------------------------------------
# Main program.

# Ensure we were invoked with a single argument.

if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    exit (1)

print ("The filename to work on is %s." % sys.argv[1])
xblue = 0.5
yblue = 0.5
xred  = 0.25
yred  = 0.75
hdg = 45.1

# Output the position and bearing in the form required by the test harness.
print ("BLUE %.3f %.3f" % (xblue, yblue))
print ("RED  %.3f %.3f" % (xred,  yred))
print ("BEARING %.1f" % hdg)

#-------------------------------------------------------------------------------

