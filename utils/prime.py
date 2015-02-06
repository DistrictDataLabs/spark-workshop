# utilities.prime
# Comptues prime numbers
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Fri Feb 06 11:53:25 2015 -0500
#
# Copyright (C) 2014 Bengfort.com
# For license information, see LICENSE.txt
#
# ID: prime.py [] benjamin@bengfort.com $

"""
Comptues prime numbers
"""

##########################################################################
## Functions
##########################################################################

def isprime(num):
    n = abs(int(num))         # n is positive
    if n < 2: return False    # 0 and 1 are not prime
    if n == 2: return True    # 2 is the only even prime
    if not n & 1:             # All even nums are not
        return False
    for x in xrange(3, int(n**0.5)+1, 2):
        if n % x == 0:
            return False
    return True
