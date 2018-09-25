import time
import os, sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="save folder", default=".", type=str)
args = parser.parse_args()

tic = time.time()
print args.i
if args.i == '.' or not os.path.isfile(args.i):
	toc = int(time.time()-tic)
	print "###"+str(toc).rjust(5)+" ### "+ "no such file"
	raise Exception('Exit')

PreContent = ['']
while True:
	f = open(args.i)
	content = f.readlines();
	f.close()

	Ready = False;
	for each in content:
		if each not in PreContent:
			toc = int(time.time()-tic)
			print "###"+str(toc).rjust(5)+" ### "+ each
		if 'Turnaround' in each:
			Ready = True
	PreContent = content[:]
	if Ready: break
	time.sleep(1)