#!/usr/bin/env python3

import datetime

start_date = datetime.date(2013, 11, 1)
end_date   = datetime.date.today() - datetime.timedelta(days=2)

dates = [ start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days))]

for d in reversed(dates):
    print(d)
