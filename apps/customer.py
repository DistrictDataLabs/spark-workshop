import csv

from datetime import datetime
from StringIO import StringIO
from collections import namedtuple

## Helper constants
DATE_FMT = "%Y-%m-%d %H:%M:%S" # 2013-09-16 12:23:33

## Named Tuple
Customer = namedtuple('Customer', ('id', 'name', 'email', 'gender', 'registered', 'city', 'state', 'zip'))

def parse(row):
    """
    Parses a row and returns a named tuple.
    """
    row[0] = int(row[0]) # Parse ID to an integer
    row[4] = datetime.strptime(row[4], DATE_FMT)
    return Customer(*row)

def split(line):
    """
    Operator function for splitting a line on a delimiter.
    """
    reader = csv.reader(StringIO(line))
    return reader.next()
