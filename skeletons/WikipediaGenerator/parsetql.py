from rdflib import Graph
import re
import os

datasetfiles = os.listdir('.')
datasetfiles = [files for files in datasetfiles if ( not files.endswith('.py') )  and not os.path.isdir(files)]
print datasetfiles
for file in datasetfiles:
	g = Graph()
	g.parse(file, format='nt')
	dirName = re.search('(?<=\_)[a-z]+(?=\.nt)', file)
	dirName = dirName.group(0)
	print dirName
	if not os.path.exists('../WikipediaAbstracts/'+dirName):
		os.mkdir('../WikipediaAbstracts/' + dirName)
	counter = 0
	for sub, pred, obj in g:
		openFile = open('../WikipediaAbstracts/' + dirName + '/' + dirName + str(counter) + '.txt', 'wb')
		print dirName + '/' + dirName + str(counter) + '.txt'
		print openFile.write(obj.encode('UTF-8'))
		counter += 1
		if counter > 100:
			openFile.close()
			break 
		openFile.close()


