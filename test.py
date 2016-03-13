import csv
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rf

companies = []
exit = []

status = {
	'acquired' : 0,
	'ipo' : 1,
	'operating' : 2,
	'closed' : 3,
}

inv_map = {v: k for k, v in status.items()}

with open('crunchbase-companies.csv', 'rU') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')

	
	for row in reader:
		timeBase = row[10]
		if row[2] == 'software' and timeBase.strip() != '' and int(timeBase[0:4]) > 1980:
			
			pattern = '%Y-%m-%d'
			base = int(time.mktime(time.strptime(timeBase, pattern)))

			time2013 = "2013-10"
			pattern2013 = "%Y-%m"
			base2013 = int(time.mktime(time.strptime(time2013, pattern2013)))

			totalTime = base2013 - base
			assert totalTime > 0
			try:
				lst = np.array([int(row[3]), int(row[9]), totalTime])
				exit.append(np.array(status[row[4]]))
				companies.append(lst)
			except ValueError:
				continue

	c = np.array(companies)[0:1700]
	e = np.array(exit)[0:1700]
	c1 = np.array(companies)[1700:]
	e1 = np.array(exit)[1700:]

	tree = rf(max_depth=5)
	tree.fit(c, e)
	print tree.score(c1, e1)

	testBase = "2021-03"
	testPattern = "%Y-%m"
	testCurrent = int(time.mktime(time.strptime(testBase, testPattern)))

	testBase = "2014-01"
	testPattern = "%Y-%m"
	testStart = int(time.mktime(time.strptime(testBase, testPattern)))
	test_point = np.array([2400000, 1, testCurrent - testStart]).reshape(1, -1)
	print inv_map[tree.predict(test_point)[0]]
	print "status = {'acquired' : 0, 'ipo' : 1,'operating' : 2, 'closed' : 3}:\n " + ' , '.join(list(map(str, tree.predict_proba(test_point)[0])))

